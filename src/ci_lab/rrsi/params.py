"""RRSI hyper-parameters (paper Table 5) and campaign profiles (design v1 §7, v2 §9)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

from ci_lab.contracts import STRATEGIES

CiThreshold = Literal["zero", "delta"]

# Contracts v2.4 added the "guard" strategy (lessons arm; writes harness/guards/*.yaml only).
# It is opt-in via ``Hyperparams.strategies``; the default allocation stays over the text strategies.
DEFAULT_STRATEGIES: tuple[str, ...] = tuple(s for s in STRATEGIES if s != "guard")


@dataclass(frozen=True)
class Hyperparams:
    """All knobs consumed by the pure RRSI core.

    Score deltas are fractions of the [0, 1] task score; cost deltas are relative token
    growth ``(C' - C) / C``. ``delta`` is never a tunable: it is measured by A/A
    calibration (``stats.aa_delta``) and stored with the campaign; ``None`` here means
    "not calibrated yet".
    """

    name: str
    T: int                         # rounds (horizon)
    k: int                         # trials per case per evaluation
    n_arms: int                    # candidate arms per round
    b_min: int                     # final-round edit budget
    b_max: int                     # first-round edit budget
    w: int                         # stall window
    m_draft: int                   # exploration slots reserved when stalled
    n_prune: int                   # prune look-back window (rounds)
    beta0: float                   # base relative-cost allowance (Eq. 7)
    beta1: float                   # gain-dependent cost allowance (Eq. 7)
    w_s: float                     # within-band weights (Eq. 17)
    w_c: float
    w_n: float
    M: int | None = None           # evolve cases (None = whatever the domain provides)
    delta: float | None = None     # calibrated noise band (A/A); None until calibrated
    missing_invalid_frac: float = 0.10   # incumbent missing-trial rate above this -> rerun
    n_bootstrap: int = 10_000
    ci_level: float = 0.95         # two-sided percentile CI; lower bound used by C16
    aa_quantile: float = 0.95      # q in delta = max(q-quantile |dS|, 1/M)
    aa_repeats_min: int = 5        # C8 minimum A/A repeats
    alpha: float = 0.05            # one-sided confirm / non-inferiority level
    safety_ni_margin: float = 0.02  # non-inferiority margin for safety scores (C11)
    require_ci_lower: bool = True  # C16: cost-rule acceptance needs CI lower bound > threshold
    ci_lower_threshold: CiThreshold = "zero"
    seed: int = 0
    # v2.2 arm strategies (design §11.2, C23): Thompson allocation over these strategies.
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    strategy_floor_every: int = 3  # K: every strategy gets >= 1 arm in any K consecutive rounds
    strategy_prior: tuple[float, float] = (1.0, 1.0)  # Beta(a0, b0) prior on accepted rate
    strategy_cap: int | None = None  # max arms per strategy per round (None = unlimited)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategies", tuple(self.strategies))
        object.__setattr__(self, "strategy_prior", tuple(float(x) for x in self.strategy_prior))
        if self.T < 1 or self.k < 1 or self.n_arms < 1:
            raise ValueError("T, k and n_arms must be >= 1")
        if not 1 <= self.b_min <= self.b_max:
            raise ValueError("need 1 <= b_min <= b_max")
        if self.w < 1 or self.n_prune < 0 or self.m_draft < 0:
            raise ValueError("w >= 1, n_prune >= 0, m_draft >= 0")
        if not 0.0 < self.ci_level < 1.0 or not 0.0 < self.alpha < 1.0 or not 0.0 < self.aa_quantile < 1.0:
            raise ValueError("ci_level, alpha and aa_quantile must be in (0, 1)")
        if self.ci_lower_threshold not in ("zero", "delta"):
            raise ValueError("ci_lower_threshold must be 'zero' or 'delta'")
        if self.delta is not None and self.delta < 0:
            raise ValueError("delta must be >= 0")
        s = self.strategies
        if not s or len(set(s)) != len(s) or any(x not in STRATEGIES for x in s):
            raise ValueError(f"strategies must be a non-empty, unique subset of {STRATEGIES}")
        if self.strategy_floor_every < 1:
            raise ValueError("strategy_floor_every must be >= 1")
        if len(s) > self.n_arms * self.strategy_floor_every:
            raise ValueError(f"floor infeasible: {len(s)} strategies need >= 1 arm every "
                             f"{self.strategy_floor_every} rounds with only {self.n_arms} arms/round")
        if len(self.strategy_prior) != 2 or min(self.strategy_prior) <= 0:
            raise ValueError("strategy_prior must be (a0, b0) with both > 0")
        if self.strategy_cap is not None and (self.strategy_cap < 1 or self.strategy_cap * len(s) < self.n_arms):
            raise ValueError("strategy_cap must be >= 1 and cap * len(strategies) >= n_arms")

    def with_(self, **changes: Any) -> Hyperparams:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strategies"], d["strategy_prior"] = list(self.strategies), list(self.strategy_prior)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hyperparams:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# Paper Table 5 (Appendix E.1). Within-band weights are only published for coding
# (domains/coding/rrsi.json: w_s=0, w_c=15, w_n=0.5); reused for the other domains.
TABLE5: dict[str, dict[str, Any]] = {
    "coding": dict(T=20, k=2, delta=0.017, b_min=1, b_max=4, w=3, m_draft=1, n_prune=4, beta0=0.10, beta1=44.5),
    "workspace": dict(T=20, k=2, delta=0.004, b_min=1, b_max=3, w=3, m_draft=1, n_prune=4, beta0=0.10, beta1=35.4),
    "engineering": dict(T=40, k=4, delta=0.020, b_min=1, b_max=4, w=3, m_draft=1, n_prune=5, beta0=0.15, beta1=24.4),
}
_WEIGHTS = dict(w_s=0.0, w_c=15.0, w_n=0.5)

PROFILES: dict[str, Hyperparams] = {
    # Wiring tests only (non-inferential): tiny, fast, short windows so every branch can fire.
    "smoke": Hyperparams(name="smoke", T=3, k=1, n_arms=2, M=12, b_min=1, b_max=2, w=1, m_draft=1,
                         n_prune=2, beta0=0.10, beta1=44.5, n_bootstrap=2_000, **_WEIGHTS),
    # Local llama-server budget (~2 h/round): design v1 §7.
    "local": Hyperparams(name="local", T=8, k=2, n_arms=2, M=40, b_min=1, b_max=3, w=2, m_draft=1,
                         n_prune=3, beta0=0.10, beta1=44.5, **_WEIGHTS),
    # Paper Table 5, coding column. delta stays None: it must be re-measured (A/A) per campaign.
    "paper": Hyperparams(name="paper", n_arms=2, **{k: v for k, v in TABLE5["coding"].items() if k != "delta"},
                         **_WEIGHTS),
}


def profile(name: str, **overrides: Any) -> Hyperparams:
    try:
        base = PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown RRSI profile {name!r}; choose from {sorted(PROFILES)}") from None
    return base.with_(**overrides) if overrides else base


def paper_reference(domain: str) -> Hyperparams:
    """Table 5 column for ``domain`` *including* the paper's own delta (reference only)."""
    try:
        row = TABLE5[domain]
    except KeyError:
        raise ValueError(f"unknown Table 5 domain {domain!r}") from None
    return Hyperparams(name=f"paper-{domain}", n_arms=2, **row, **_WEIGHTS)
