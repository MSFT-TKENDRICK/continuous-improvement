"""Arm-strategy allocation (design §11.2, C23): Thompson sampling over ``STRATEGIES``.

Per-strategy evidence is tracked *separately* from per-component evidence: a strategy's
unit is the arm (one measured arm = one trial; success = the arm was elected, paper a = 1).
The posterior of strategy s is Beta(a0 + successes_s, b0 + failures_s).

Each round: strategies that have not had a measured arm in the last K rounds (or never)
are forced first (floor), oldest first; the remaining slots are filled by Thompson draws,
seeded by ``derive_seed(hp.seed, t, "strategy")`` so a plan is a pure function of
(t, hp, history). Selection (Algorithm 2) never sees the strategy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ci_lab.contracts import STRATEGIES

from . import stats
from .history import HistoryRecord
from .params import Hyperparams


@dataclass(frozen=True)
class StrategyStats:
    strategy: str
    tried: int              # measured arms run with this strategy
    accepted: int           # of which elected (a = 1)
    last_round: int | None  # latest round with a measured arm (None = never)
    alpha: float            # posterior Beta parameters
    beta: float

    @property
    def success_rate(self) -> float:
        return self.accepted / self.tried if self.tried else 0.0

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "success_rate": self.success_rate, "posterior_mean": self.posterior_mean}


def strategy_stats(records: Iterable[HistoryRecord], strategies: Sequence[str] = STRATEGIES,
                   prior: tuple[float, float] = (1.0, 1.0)) -> dict[str, StrategyStats]:
    tried: Counter[str] = Counter()
    won: Counter[str] = Counter()
    last: dict[str, int] = {}
    for r in records:
        if not r.measured:
            continue
        tried[r.strategy] += 1
        won[r.strategy] += int(r.accepted)
        last[r.strategy] = max(last.get(r.strategy, r.round), r.round)
    a0, b0 = prior
    return {s: StrategyStats(s, tried[s], won[s], last.get(s), a0 + won[s], b0 + tried[s] - won[s])
            for s in strategies}


def starved(st: dict[str, StrategyStats], t: int, every: int) -> tuple[str, ...]:
    """Strategies owed an arm this round (never run, or last run >= K rounds ago), oldest first."""
    pos = {s: i for i, s in enumerate(st)}
    due = [s for s, x in st.items() if x.last_round is None or t - x.last_round >= every]
    return tuple(sorted(due, key=lambda s: (-1 if st[s].last_round is None else st[s].last_round, pos[s])))


@dataclass(frozen=True)
class StrategyAllocation:
    round: int
    strategies: tuple[str, ...]          # one per arm slot, in arm order
    reasons: tuple[str, ...]             # "floor" | "thompson" per slot
    samples: tuple[dict[str, float], ...]  # Thompson draws per slot ({} for floor slots)
    stats: dict[str, StrategyStats]

    def to_dict(self) -> dict[str, Any]:
        return {"round": self.round, "strategies": list(self.strategies), "reasons": list(self.reasons),
                "samples": [dict(s) for s in self.samples], "stats": {k: v.to_dict() for k, v in self.stats.items()}}


def allocate_strategies(t: int, n_arms: int, history: Iterable[HistoryRecord], hp: Hyperparams) -> StrategyAllocation:
    """Assign a strategy to each of ``n_arms`` slots for round ``t`` (history of rounds < t only)."""
    if n_arms < 1:
        raise ValueError("n_arms must be >= 1")
    records = [r for r in history if r.round < t]
    st = strategy_stats(records, hp.strategies, hp.strategy_prior)
    cap = hp.strategy_cap if hp.strategy_cap is not None else n_arms
    if cap * len(hp.strategies) < n_arms:
        raise ValueError("strategy_cap too small for the number of arms")
    chosen: list[str] = []
    reasons: list[str] = []
    samples: list[dict[str, float]] = []
    for s in starved(st, t, hp.strategy_floor_every)[:n_arms]:
        chosen.append(s)
        reasons.append("floor")
        samples.append({})
    rng = np.random.default_rng(stats.derive_seed(hp.seed, t, "strategy"))
    alphas = np.array([st[s].alpha for s in hp.strategies])
    betas = np.array([st[s].beta for s in hp.strategies])
    while len(chosen) < n_arms:
        theta = rng.beta(alphas, betas)
        used = Counter(chosen)
        eligible = [i for i, s in enumerate(hp.strategies) if used[s] < cap]
        best = max(eligible, key=lambda i: (theta[i], -i))
        chosen.append(hp.strategies[best])
        reasons.append("thompson")
        samples.append({s: float(theta[i]) for i, s in enumerate(hp.strategies)})
    return StrategyAllocation(round=t, strategies=tuple(chosen), reasons=tuple(reasons), samples=tuple(samples),
                              stats=st)
