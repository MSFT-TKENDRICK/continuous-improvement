"""Statistics for RRSI: missing=0 aggregation, paired task-level bootstrap, A/A delta,
one-sided confirmation and safety non-inferiority (design v1 §6, v2 §9 C8/C11/C16).

Everything is deterministic given a seed (``numpy.random.default_rng``). Resampling is
at the *case* level: the k trials of a case are correlated, so trial-level resampling
would understate the interval (paper Table 6).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from ci_lab.contracts import EvalResult, TaskScore
from ci_lab.metrics.runtime import percentile

Key = tuple[str, int]  # (case_id, trial)
SeedLike = int | Sequence[int]


def derive_seed(base: int, *parts: Any) -> list[int]:
    """Stable per-(round, arm, ...) seed; never uses ``hash()`` (randomised per process)."""
    out = [int(base) & 0xFFFFFFFF]
    for p in parts:
        out.append(p & 0xFFFFFFFF if isinstance(p, int) else zlib.crc32(str(p).encode("utf-8")))
    return out


# ---------------------------------------------------------------- aggregation (missing = 0)

def index_scores(scores: Iterable[TaskScore]) -> dict[Key, TaskScore]:
    idx: dict[Key, TaskScore] = {}
    for s in scores:
        key = (s.case_id, s.trial)
        if key in idx:
            raise ValueError(f"duplicate task score for case {s.case_id!r} trial {s.trial}")
        idx[key] = s
    return idx


def universe(evals: Iterable[EvalResult | None], *, cases: Sequence[str] | None = None,
             k: int | None = None) -> list[Key]:
    """The (case, trial) keys every arm is scored over. With ``cases`` (and ``k``) the
    universe is fixed (absent trials count as missing); otherwise it is the union of
    keys observed across all evaluations."""
    if cases is not None:
        if k is None:
            raise ValueError("k is required when cases are given")
        return sorted((c, t) for c in set(cases) for t in range(k))
    keys: set[Key] = set()
    for ev in evals:
        if ev is not None:
            keys.update((s.case_id, s.trial) for s in ev.scores)
    return sorted(keys)


def mean_missing_zero(values: Iterable[float | None], n_expected: int | None = None) -> float:
    """Mean where ``None`` (and absent values up to ``n_expected``) count as 0."""
    vals = [0.0 if v is None else float(v) for v in values]
    n = max(len(vals), n_expected or 0)
    return sum(vals) / n if n else 0.0


def _score(idx: Mapping[Key, TaskScore], key: Key) -> float:
    s = idx.get(key)
    return 0.0 if s is None or s.score is None else float(s.score)


def case_means(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> dict[str, float]:
    """Per-case mean over the case's trials in ``keys`` (missing = 0)."""
    acc: dict[str, list[float]] = {}
    for key in keys:
        acc.setdefault(key[0], []).append(_score(idx, key))
    return {c: sum(v) / len(v) for c, v in acc.items()}


def task_score(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> float:
    """S-hat: mean over cases of the per-case mean (missing = 0)."""
    cm = case_means(idx, keys)
    return sum(cm.values()) / len(cm) if cm else 0.0


def missing_rate(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> float:
    if not keys:
        return 0.0
    missing = sum(1 for k in keys if idx.get(k) is None or idx[k].score is None)
    return missing / len(keys)


def mean_cost(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> float:
    """C-hat: mean tokens (in + out) per *completed* trial."""
    vals = [idx[k].tokens_in + idx[k].tokens_out for k in keys if k in idx and idx[k].score is not None]
    return sum(vals) / len(vals) if vals else 0.0


def relative_cost(c_new: float, c_old: float) -> float:
    """Delta-C = (C' - C) / C; +inf when the baseline cost is 0 and the new cost is not."""
    if c_old == 0:
        return 0.0 if c_new == 0 else math.inf
    return (c_new - c_old) / c_old


def critical_count(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> int:
    return sum(1 for k in keys if k in idx for v in idx[k].violations if v.severity == "critical")


# ---------------------------------------------------------------- runtime resources (A14; never part of S)

def completed(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> list[TaskScore]:
    """Completed trials (``score`` not None) among ``keys``."""
    return [idx[k] for k in keys if k in idx and idx[k].score is not None]


def runtime_observed(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> bool:
    """True when some completed trial carries runtime metrics (wall ms or LLM/tool calls)."""
    return any(s.wall_ms > 0 or s.llm_calls > 0 or s.tool_calls > 0 for s in completed(idx, keys))


def mean_calls(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> float:
    """Mean ``llm_calls + tool_calls`` per completed trial (0 when none)."""
    vals = [s.llm_calls + s.tool_calls for s in completed(idx, keys)]
    return sum(vals) / len(vals) if vals else 0.0


def median_wall_ms(idx: Mapping[Key, TaskScore], keys: Sequence[Key]) -> float:
    """Median ``wall_ms`` over completed trials (0 when none)."""
    return percentile([float(s.wall_ms) for s in completed(idx, keys)], 50)


# ---------------------------------------------------------------- bootstrap

def paired_case_deltas(cand: Mapping[str, float], inc: Mapping[str, float]) -> np.ndarray:
    """Per-case paired differences over the union of cases (absent case = 0)."""
    cases = sorted(set(cand) | set(inc))
    return np.array([cand.get(c, 0.0) - inc.get(c, 0.0) for c in cases], dtype=float)


def bootstrap_means(deltas: np.ndarray, *, n_resamples: int, seed: SeedLike) -> np.ndarray:
    d = np.asarray(deltas, dtype=float)
    if d.size == 0:
        return np.zeros(n_resamples)
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples)
    chunk = max(1, 2_000_000 // d.size)
    for start in range(0, n_resamples, chunk):
        stop = min(n_resamples, start + chunk)
        out[start:stop] = d[rng.integers(0, d.size, size=(stop - start, d.size))].mean(axis=1)
    return out


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    level: float
    n_cases: int
    n_resamples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_bootstrap(cand: Mapping[str, float], inc: Mapping[str, float], *, n_resamples: int = 10_000,
                     level: float = 0.95, seed: SeedLike = 0) -> BootstrapCI:
    """Two-sided percentile CI for the mean paired case-level difference cand - inc."""
    d = paired_case_deltas(cand, inc)
    boot = bootstrap_means(d, n_resamples=n_resamples, seed=seed)
    tail = (1.0 - level) / 2.0
    return BootstrapCI(mean=float(d.mean()) if d.size else 0.0, lower=float(np.quantile(boot, tail)),
                       upper=float(np.quantile(boot, 1.0 - tail)), level=level, n_cases=int(d.size),
                       n_resamples=n_resamples)


# ---------------------------------------------------------------- A/A noise band

@dataclass(frozen=True)
class AADelta:
    delta: float
    quantile_value: float
    floor: float
    bound_by: str          # "quantile" | "floor"
    q: float
    repeats: int
    pairs: int
    n_cases: int
    observed_abs: tuple[float, ...]  # |dS| of each repeat pair
    n_resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["observed_abs"] = list(self.observed_abs)
        return d


def aa_delta(repeats: Sequence[Mapping[str, float]], *, q: float = 0.95, n_resamples: int = 10_000,
             seed: int = 0, n_cases: int | None = None) -> AADelta:
    """delta = max(q-quantile of |dS| over paired bootstrap of every A/A repeat pair, 1/M).

    ``repeats`` are per-case mean scores of R >= 2 repeated evaluations of the *unchanged*
    harness. Bootstrap samples of every pair (i < j) are pooled before taking the quantile.
    ``n_cases`` (M) sets the granularity floor; default = number of distinct cases.
    """
    if len(repeats) < 2:
        raise ValueError("A/A calibration needs at least 2 repeats")
    cases = set().union(*(set(r) for r in repeats))
    m = n_cases if n_cases is not None else len(cases)
    if m < 1:
        raise ValueError("A/A calibration needs at least one case")
    pooled, observed = [], []
    for i in range(len(repeats)):
        for j in range(i + 1, len(repeats)):
            d = paired_case_deltas(repeats[j], repeats[i])
            observed.append(abs(float(d.mean())) if d.size else 0.0)
            pooled.append(np.abs(bootstrap_means(d, n_resamples=n_resamples, seed=derive_seed(seed, "aa", i, j))))
    qv = float(np.quantile(np.concatenate(pooled), q))
    floor = 1.0 / m
    return AADelta(delta=max(qv, floor), quantile_value=qv, floor=floor,
                   bound_by="quantile" if qv >= floor else "floor", q=q, repeats=len(repeats),
                   pairs=len(observed), n_cases=m, observed_abs=tuple(observed), n_resamples=n_resamples, seed=seed)


def aa_repeats_for_precision(target_halfwidth: float, pilot: Sequence[float], *, level: float = 0.95,
                             minimum: int = 5, maximum: int | None = None) -> int:
    """C8: number of A/A repeats R so the CI half-width of the incumbent's mean score,
    z * s / sqrt(R), is <= ``target_halfwidth``; ``s`` is the sd of pilot repeat scores
    (S-hat of >= 2 pilot repeats). Never below ``minimum`` (default 5)."""
    if target_halfwidth <= 0:
        raise ValueError("target_halfwidth must be > 0")
    if len(pilot) < 2:
        raise ValueError("pilot needs at least 2 repeat scores")
    s = float(np.std(np.asarray(pilot, dtype=float), ddof=1))
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    r = math.ceil(round((z * s / target_halfwidth) ** 2, 9))
    r = max(minimum, r)
    return min(r, maximum) if maximum is not None else r


# ---------------------------------------------------------------- confirmatory tests

@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class

    name: str
    estimate: float
    bound: float          # one-sided lower bound at level 1 - alpha
    threshold: float      # null boundary the bound must exceed
    p_value: float
    alpha: float
    passed: bool
    n_cases: int
    n_resamples: int

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


def _one_sided(name: str, d: np.ndarray, threshold: float, *, alpha: float, n_resamples: int,
               seed: SeedLike) -> TestResult:
    boot = bootstrap_means(d, n_resamples=n_resamples, seed=seed)
    bound = float(np.quantile(boot, alpha))
    return TestResult(name=name, estimate=float(d.mean()) if d.size else 0.0, bound=bound, threshold=threshold,
                      p_value=float(np.mean(boot <= threshold)), alpha=alpha, passed=bound > threshold,
                      n_cases=int(d.size), n_resamples=n_resamples)


def confirm_test(cand: Mapping[str, float], inc: Mapping[str, float], *, alpha: float = 0.05,
                 n_resamples: int = 10_000, seed: SeedLike = 0) -> TestResult:
    """One-sided H1: mean(cand - inc) > 0 (paired, case-level bootstrap). Passes iff the
    alpha-quantile of the bootstrap distribution is > 0."""
    return _one_sided("confirm", paired_case_deltas(cand, inc), 0.0, alpha=alpha, n_resamples=n_resamples,
                      seed=seed)


def non_inferiority(cand: Mapping[str, float], inc: Mapping[str, float], *, margin: float, alpha: float = 0.05,
                    higher_is_better: bool = True, n_resamples: int = 10_000, seed: SeedLike = 0) -> TestResult:
    """Safety non-inferiority (C11): H1: (cand - inc) > -margin in the "better" direction.
    For violation counts/rates pass ``higher_is_better=False``."""
    if margin < 0:
        raise ValueError("margin must be >= 0")
    d = paired_case_deltas(cand, inc)
    if not higher_is_better:
        d = -d
    return _one_sided("non_inferiority", d, -margin, alpha=alpha, n_resamples=n_resamples, seed=seed)
