"""Agent SRE reliability budgets per campaign arm and per s1 judge pool.

Each key gets an ``agent_sre.SLO`` with a ``TaskSuccessRate`` SLI and an ``ErrorBudget`` sized in
failures (``CIRCUIT_BREAK`` on exhaustion). ``vetoed(key)`` is the circuit break consulted by the
election: it depends only on the failure count, never on wall-clock burn rates.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from agent_sre import SLO
    from agent_sre.slo.indicators import TaskSuccessRate
    from agent_sre.slo.objectives import ErrorBudget, ExhaustionAction

__all__ = ["ArmReliability", "JudgePoolReliability", "ReliabilityBook", "ReliabilityStatus", "arm_vetoes"]


@dataclass(frozen=True)
class ReliabilityStatus:
    key: str
    kind: str
    good: int
    bad: int
    success_rate: float | None
    budget_remaining: float
    status: str
    vetoed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReliabilityBook:
    """Error-budget book keyed by arm or judge-pool name."""

    kind = "arm"

    def __init__(self, *, max_failures: int = 3, target: float = 0.9):
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        if not 0.0 < target < 1.0:
            raise ValueError("target must be in (0, 1)")
        self.max_failures = max_failures
        self.target = target
        self._slos: dict[str, SLO] = {}
        self._counts: dict[str, list[int]] = {}

    def _slo(self, key: str) -> SLO:
        if not key or not isinstance(key, str):
            raise ValueError(f"{self.kind} key must be a non-empty string")
        if key not in self._slos:
            budget = ErrorBudget(total=float(self.max_failures),
                                 exhaustion_action=ExhaustionAction.CIRCUIT_BREAK)
            self._slos[key] = SLO(f"ci.{self.kind}.{key}", [TaskSuccessRate(target=self.target)],
                                  error_budget=budget, labels={"kind": self.kind}, agent_id=key)
            self._counts[key] = [0, 0]
        return self._slos[key]

    def record(self, key: str, ok: bool) -> bool:
        """Record one outcome; returns whether ``key`` is now vetoed."""
        slo = self._slo(key)
        slo.indicators[0].record_task(bool(ok))
        slo.record_event(bool(ok))
        self._counts[key][0 if ok else 1] += 1
        return self.vetoed(key)

    def vetoed(self, key: str) -> bool:
        slo = self._slos.get(key)
        return slo is not None and slo.error_budget.is_exhausted

    def status(self, key: str) -> ReliabilityStatus:
        slo = self._slo(key)
        good, bad = self._counts[key]
        exhausted = slo.error_budget.is_exhausted
        return ReliabilityStatus(
            key=key, kind=self.kind, good=good, bad=bad,
            success_rate=good / (good + bad) if good + bad else None,
            budget_remaining=slo.error_budget.remaining,
            status="exhausted" if exhausted else ("healthy" if bad == 0 else "degraded"),
            vetoed=exhausted)

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {k: self.status(k).to_dict() for k in sorted(self._slos)}

    def vetoes(self) -> list[str]:
        return sorted(k for k in self._slos if self.vetoed(k))

    def reset(self, key: str) -> None:
        self._slos.pop(key, None)
        self._counts.pop(key, None)


class ArmReliability(ReliabilityBook):
    kind = "arm"


class JudgePoolReliability(ReliabilityBook):
    kind = "judge_pool"


def arm_vetoes(history: Iterable[Mapping[str, Any]], *, max_failures: int = 3) -> list[str]:
    """Replay campaign ``history.jsonl`` rows into an ``ArmReliability`` book; return vetoed keys.

    Keyed by strategy (arm slots ``v1..vN`` are reassigned every round). ``evaluated`` counts as good,
    ``failed`` as bad; critic rejections and pending arms are not reliability events. Derived from
    the ledger so resume / readjudicate see the same vetoes.
    """
    book = ArmReliability(max_failures=max_failures)
    for row in history:
        for arm in row.get("arms") or ():
            status = arm.get("status")
            if status in ("evaluated", "failed"):
                book.record(str(arm.get("strategy") or "agent"), status == "evaluated")
    return book.vetoes()
