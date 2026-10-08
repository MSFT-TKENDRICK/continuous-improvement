"""Algorithm 1 schedule: annealed budget b_t, stall flag, untried/prune sets, exploration
slots and one directive per arm (paper Eq. 4, 11, 13, 14)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ci_lab.contracts import COMPONENTS

from .attribution import component_stats
from .history import HistoryRecord, tried_components
from .params import Hyperparams


def edit_budget(t: int, T: int, b_min: int, b_max: int) -> int:
    """Eq. 4: b_t = ceil(b_min + (b_max - b_min) * (1 + cos(pi t / T)) / 2); t clamped to [0, T]."""
    if T < 1:
        raise ValueError("T must be >= 1")
    t = min(max(t, 0), T)
    raw = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))
    return int(math.ceil(round(raw, 9)))  # round() strips float noise such as 1.0000000000000002


def stall_flag(trajectory: Sequence[float], t: int, w: int, delta: float) -> bool:
    """sigma_t = 1[S_t - S_{t-w} <= delta]; ``trajectory[i]`` is the incumbent score at the
    start of round i. False until w rounds of history exist."""
    if t < w or t >= len(trajectory) or w < 1:
        return False
    return trajectory[t] - trajectory[t - w] <= delta


def untried(records: Iterable[HistoryRecord], components: Sequence[str] = COMPONENTS) -> tuple[str, ...]:
    """U_t = K minus T_t, in vocabulary order."""
    seen = tried_components(records)
    return tuple(c for c in components if c not in seen)


def component_yield(records: Iterable[HistoryRecord], t: int, n_prune: int) -> dict[str, float]:
    """g_t(l) = max{dS_i : l_i = l, t - t_i <= n_prune} (max of empty = -inf), for l in T_t."""
    records = list(records)
    out: dict[str, float] = {c: -math.inf for c in tried_components(records)}
    for r in records:
        if r.delta_s is None or t - r.round > n_prune:
            continue
        for c in r.components:
            out[c] = max(out[c], float(r.delta_s))
    return out


def prune_set(records: Iterable[HistoryRecord], t: int, n_prune: int,
              components: Sequence[str] = COMPONENTS) -> tuple[str, ...]:
    """B_t = {l in T_t : g_t(l) <= 0} (Eq. 14), in vocabulary order."""
    g = component_yield(records, t, n_prune)
    order = {c: i for i, c in enumerate(components)}
    return tuple(sorted((c for c, v in g.items() if v <= 0), key=lambda c: (order.get(c, len(order)), c)))


def exploration_slots(stalled: bool, untried_set: Sequence[str], m_draft: int, n_arms: int) -> int:
    """Slots reserved for never-tried components (Eq. 13): m_draft when stalled and U_t is
    non-empty, capped by the number of arms."""
    return min(m_draft, n_arms) if stalled and untried_set else 0


@dataclass(frozen=True)
class Directive:
    arm: str
    round: int
    budget: int                 # max atomic edits (one tagged commit each)
    explore: bool               # reserved exploration slot -> focus is a never-tried component
    focus: str | None           # suggested component (None = proposer's choice)
    avoid: tuple[str, ...]      # prune set B_t: unproductive components, candidates for deletion
    stalled: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["avoid"] = list(self.avoid)
        return d


@dataclass(frozen=True)
class RoundSchedule:
    round: int
    budget: int
    stalled: bool
    untried: tuple[str, ...]
    prune: tuple[str, ...]
    exploration_slots: int
    delta: float
    directives: tuple[Directive, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"round": self.round, "budget": self.budget, "stalled": self.stalled, "untried": list(self.untried),
                "prune": list(self.prune), "exploration_slots": self.exploration_slots, "delta": self.delta,
                "directives": [d.to_dict() for d in self.directives]}


def default_arms(n: int) -> tuple[str, ...]:
    return tuple(f"v{i + 1}" for i in range(n))


def _exploit_order(records: Sequence[HistoryRecord], components: Sequence[str], avoid: Sequence[str]) -> list[str]:
    """Non-pruned components ranked by attribution stats: success rate, then mean dS
    (untried = 0, so they rank above components with negative yield), then vocabulary order."""
    stats = component_stats(records, components)
    pos = {c: i for i, c in enumerate(components)}
    pool = [c for c in components if c not in set(avoid)]
    return sorted(pool, key=lambda c: (-stats[c].success_rate, -stats[c].mean_delta_s, pos[c]))


def plan_round(t: int, hp: Hyperparams, history: Sequence[HistoryRecord], trajectory: Sequence[float],
             delta: float, arms: Sequence[str] | None = None,
             components: Sequence[str] = COMPONENTS) -> RoundSchedule:
    """Everything Algorithm 1 lines 2-7 compute for round t, plus per-arm directives.
    Only history from rounds < t is considered."""
    records = [r for r in history if r.round < t]
    arms = tuple(arms) if arms is not None else default_arms(hp.n_arms)
    if len(set(arms)) != len(arms) or not arms:
        raise ValueError("arms must be non-empty and unique")
    budget = edit_budget(t, hp.T, hp.b_min, hp.b_max)
    stalled = stall_flag(trajectory, t, hp.w, delta)
    u = untried(records, components)
    b = prune_set(records, t, hp.n_prune, components)
    m = exploration_slots(stalled, u, hp.m_draft, len(arms))
    exploit = _exploit_order(records, components, b)
    out = []
    for i, arm in enumerate(arms):
        if i < m:
            focus, explore = u[i % len(u)], True
        else:
            j = i - m
            focus, explore = (exploit[j % len(exploit)] if exploit else None), False
        out.append(Directive(arm=arm, round=t, budget=budget, explore=explore, focus=focus, avoid=b, stalled=stalled))
    return RoundSchedule(round=t, budget=budget, stalled=stalled, untried=u, prune=b, exploration_slots=m,
                         delta=delta, directives=tuple(out))


def directives(t: int, hp: Hyperparams, history: Sequence[HistoryRecord], trajectory: Sequence[float],
               delta: float, arms: Sequence[str] | None = None,
               components: Sequence[str] = COMPONENTS) -> tuple[Directive, ...]:
    """One directive per arm (component focus, budget, explore flag)."""
    return plan_round(t, hp, history, trajectory, delta, arms, components).directives
