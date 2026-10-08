"""Exact binomial statistics for lesson validation and shadow→enforce promotion (N3).

Pure Python (no scipy): Clopper-Pearson bounds are found by bisection on the exact binomial
CDF, evaluated in log space (``math.lgamma`` + log-sum-exp) so large ``n`` stays stable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from ci_lab.rulespec import (
    PROMOTE_FP_UCB,
    PROMOTE_MIN_ADJUDICATED,
    PROMOTE_MIN_FIRES,
    PROMOTE_MIN_OPPORTUNITIES,
)

__all__ = [
    "binom_cdf",
    "binom_sf",
    "clopper_pearson",
    "cp_lower",
    "cp_upper",
    "fp_negatives",
    "promotion_ok",
    "promotion_ok_stratified",
]

_TOL = 1e-13
_MAX_ITER = 200


def _check(k: int, n: int) -> None:
    if not (isinstance(k, int) and isinstance(n, int)) or n < 0 or k < 0 or k > n:
        raise ValueError(f"need integers 0 <= k <= n, got k={k!r} n={n!r}")


def _log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_comb + k * math.log(p) + (n - k) * math.log1p(-p)


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    _check(k, n)
    if k == n:
        return 1.0
    return min(1.0, math.exp(_logsumexp([_log_pmf(i, n, p) for i in range(k + 1)])))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    _check(k, n)
    if k == 0:
        return 1.0
    return min(1.0, math.exp(_logsumexp([_log_pmf(i, n, p) for i in range(k, n + 1)])))


def _bisect(f: Callable[[float], float], target: float, *, increasing: bool) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(_MAX_ITER):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOL:
            break
    return (lo + hi) / 2


def _alpha(confidence: float, one_sided: bool) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    a = 1.0 - confidence
    return a if one_sided else a / 2


def cp_upper(k: int, n: int, confidence: float = 0.95, *, one_sided: bool = False) -> float:
    """Exact Clopper-Pearson upper bound: largest p with P(X <= k; n, p) >= alpha.

    ``one_sided=False`` (default) returns the upper end of the two-sided ``confidence`` interval
    (alpha/2 per tail) — the conservative choice used for promotion and replay.
    """
    _check(k, n)
    a = _alpha(confidence, one_sided)
    if n == 0 or k == n:
        return 1.0
    if k == 0:
        return 1.0 - a ** (1.0 / n)
    return _bisect(lambda p: binom_cdf(k, n, p), a, increasing=False)


def cp_lower(k: int, n: int, confidence: float = 0.95, *, one_sided: bool = False) -> float:
    """Exact Clopper-Pearson lower bound: smallest p with P(X >= k; n, p) >= alpha."""
    _check(k, n)
    a = _alpha(confidence, one_sided)
    if k == 0:
        return 0.0
    if k == n:
        return a ** (1.0 / n)
    return _bisect(lambda p: binom_sf(k, n, p), a, increasing=True)


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided exact Clopper-Pearson interval for ``k`` successes in ``n`` trials."""
    return cp_lower(k, n, confidence), cp_upper(k, n, confidence)


def fp_negatives(opportunities: int, adjudicated_positives: int) -> int:
    """Denominator of the FP rate: opportunities that were not adjudicated true violations."""
    return max(0, opportunities - adjudicated_positives)


def promotion_ok(
    opportunities: int,
    fires: int,
    adjudicated_positives: int,
    false_positives: int,
    *,
    epsilon: float = PROMOTE_FP_UCB,
    confidence: float = 0.95,
) -> tuple[bool, list[str]]:
    """Shadow→enforce gate (N3). Returns ``(ok, reasons)``; ``reasons`` lists every failed check.

    FP rate = false_positives / (opportunities - adjudicated_positives); its exact two-sided
    Clopper-Pearson ``confidence`` upper bound must be <= ``epsilon``.
    """
    for name, v in (("opportunities", opportunities), ("fires", fires),
                    ("adjudicated_positives", adjudicated_positives), ("false_positives", false_positives)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return False, [f"invalid {name}={v!r}"]
    reasons: list[str] = []
    if fires > opportunities:
        reasons.append(f"inconsistent: fires {fires} > opportunities {opportunities}")
    if adjudicated_positives + false_positives > fires:
        reasons.append(f"inconsistent: adjudicated_positives+false_positives "
                       f"{adjudicated_positives + false_positives} > fires {fires}")
    if reasons:
        return False, reasons
    if opportunities < PROMOTE_MIN_OPPORTUNITIES:
        reasons.append(f"opportunities {opportunities} < {PROMOTE_MIN_OPPORTUNITIES}")
    if fires < PROMOTE_MIN_FIRES:
        reasons.append(f"fires {fires} < {PROMOTE_MIN_FIRES}")
    if adjudicated_positives < PROMOTE_MIN_ADJUDICATED:
        reasons.append(f"adjudicated_positives {adjudicated_positives} < {PROMOTE_MIN_ADJUDICATED}")
    neg = fp_negatives(opportunities, adjudicated_positives)
    if neg == 0:
        reasons.append("no negatives to bound the FP rate")
    else:
        ucb = cp_upper(false_positives, neg, confidence)
        if ucb > epsilon:
            reasons.append(f"fp_ucb {ucb:.4f} > {epsilon} ({false_positives}/{neg})")
    return not reasons, reasons


def promotion_ok_stratified(
    strata: Mapping[str, tuple[int, int, int, int]],
    *,
    epsilon: float = PROMOTE_FP_UCB,
    confidence: float = 0.95,
) -> tuple[bool, list[str]]:
    """N3 "stratified by intent": the aggregate gate over all strata AND, for each stratum
    ``(opportunities, fires, adjudicated_positives, false_positives)``, FP UCB <= ``epsilon``.

    A stratum too sparse to bound its FP rate fails with an explicit reason (add targeted
    synthetic coverage from the evolve generator, per N3).
    """
    if not strata:
        return False, ["no strata"]
    totals = [sum(s[i] for s in strata.values()) for i in range(4)]
    _, agg = promotion_ok(*totals, epsilon=epsilon, confidence=confidence)
    reasons = [f"aggregate: {r}" for r in agg]
    for name in sorted(strata):
        opp, _fires, adj, fp = strata[name]
        neg = fp_negatives(opp, adj)
        ucb = cp_upper(fp, neg, confidence) if neg else 1.0
        if ucb > epsilon:
            reasons.append(f"stratum {name}: fp_ucb {ucb:.4f} > {epsilon} ({fp}/{neg})")
    return not reasons, reasons
