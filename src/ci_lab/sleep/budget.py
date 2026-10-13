"""Hard ceilings for one sleep night (design C10): tasks, rollouts, tokens/AIU, wall clock.

A :class:`Budget` is charged by the backend before/after every rollout and by the
reflector after every call. The first ceiling crossed raises :class:`BudgetExceeded`
(and is remembered in :attr:`Budget.exceeded`) so the night can record partial
results instead of silently running long.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


class BudgetExceeded(RuntimeError):
    def __init__(self, kind: str, limit: float, used: float) -> None:
        super().__init__(f"sleep budget exceeded: {kind} used {used} > limit {limit}")
        self.kind = kind
        self.limit = limit
        self.used = used

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "limit": self.limit, "used": self.used}


@dataclass(frozen=True)
class BudgetLimits:
    max_tasks: int = 40
    max_rollouts: int = 400
    max_tokens: int = 2_000_000
    max_aiu: float | None = None  # Copilot nano-AIU are converted by the caller; None = unlimited
    max_minutes: float = 60.0

    def __post_init__(self) -> None:
        for name in ("max_tasks", "max_rollouts", "max_tokens"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_minutes <= 0:
            raise ValueError("max_minutes must be positive")
        if self.max_aiu is not None and self.max_aiu <= 0:
            raise ValueError("max_aiu must be positive")


class Budget:
    """Thread-safe counters (SkillOpt replays may run in a thread pool)."""

    def __init__(self, limits: BudgetLimits | None = None, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits or BudgetLimits()
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self.tasks = 0
        self.rollouts = 0
        self.tokens = 0
        self.aiu = 0.0
        self.exceeded: BudgetExceeded | None = None

    # -- helpers
    def _fail(self, kind: str, limit: float, used: float) -> BudgetExceeded:
        exc = BudgetExceeded(kind, limit, used)
        if self.exceeded is None:
            self.exceeded = exc
        return exc

    @property
    def elapsed_minutes(self) -> float:
        return (self._clock() - self._started) / 60.0

    def check_time(self) -> None:
        elapsed = self.elapsed_minutes
        if elapsed > self.limits.max_minutes:
            raise self._fail("minutes", self.limits.max_minutes, round(elapsed, 3))

    # -- charges
    def admit_tasks(self, n: int) -> None:
        with self._lock:
            if n > self.limits.max_tasks:
                raise self._fail("tasks", self.limits.max_tasks, n)
            self.tasks = n

    def begin_rollout(self) -> None:
        """Called BEFORE a rollout: refuses to start one past any ceiling."""
        with self._lock:
            if (prev := self.exceeded) is not None:
                raise BudgetExceeded(prev.kind, prev.limit, prev.used)
            self.check_time()
            if self.rollouts + 1 > self.limits.max_rollouts:
                raise self._fail("rollouts", self.limits.max_rollouts, self.rollouts + 1)
            self.rollouts += 1

    def charge(self, *, tokens: int = 0, aiu: float = 0.0) -> None:
        """Called AFTER a model call with its measured usage."""
        with self._lock:
            self.tokens += max(0, int(tokens))
            self.aiu += max(0.0, float(aiu))
            if self.tokens > self.limits.max_tokens:
                raise self._fail("tokens", self.limits.max_tokens, self.tokens)
            if self.limits.max_aiu is not None and self.aiu > self.limits.max_aiu:
                raise self._fail("aiu", self.limits.max_aiu, round(self.aiu, 6))
            self.check_time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "limits": asdict(self.limits),
                "used": {"tasks": self.tasks, "rollouts": self.rollouts, "tokens": self.tokens,
                         "aiu": round(self.aiu, 6), "minutes": round(self.elapsed_minutes, 3)},
                "exceeded": self.exceeded.to_dict() if self.exceeded else None,
            }
