"""Evaluator-side runtime metering: :class:`RunMeter` around one run, :func:`runtime_summary` over trials."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from ci_lab.contracts import TaskScore

__all__ = ["RunMeter", "percentile", "runtime_summary"]


class RunMeter:
    """Sync context manager timing a run with ``time.perf_counter`` and counting LLM/tool calls.

    ``wall_ms`` is live while the meter is open and frozen on exit. Counters are thread-safe so
    middleware on worker threads may report into one meter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._t1: float | None = None
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.tool_names: Counter[str] = Counter()

    def __enter__(self) -> Self:
        self._t0, self._t1 = time.perf_counter(), None
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self._t1 = time.perf_counter()

    @property
    def wall_ms(self) -> float:
        if self._t0 is None:
            return 0.0
        end = self._t1 if self._t1 is not None else time.perf_counter()
        return (end - self._t0) * 1000.0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def llm_call(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += max(int(tokens_in or 0), 0)
            self.tokens_out += max(int(tokens_out or 0), 0)

    def tool_call(self, name: str) -> None:
        with self._lock:
            self.tool_calls += 1
            self.tool_names[str(name)] += 1

    def as_dict(self) -> dict[str, float]:
        return {"wall_ms": self.wall_ms, "llm_calls": float(self.llm_calls), "tool_calls": float(self.tool_calls),
                "tokens_in": float(self.tokens_in), "tokens_out": float(self.tokens_out),
                "tokens": float(self.tokens)}


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]); 0.0 when empty."""
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * min(max(q, 0.0), 100.0) / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def runtime_summary(scores: Sequence[TaskScore]) -> dict[str, float]:
    """Runtime cost over completed trials (``score is not None``); every value is 0.0 when there are none."""
    done = [s for s in scores if s.score is not None]
    n = len(done)
    if not n:
        return dict.fromkeys(("wall_ms_mean", "wall_ms_p50", "wall_ms_p95", "tokens_per_task",
                              "llm_calls_per_task", "tool_calls_per_task", "n"), 0.0)
    wall = [float(s.wall_ms) for s in done]
    return {
        "wall_ms_mean": sum(wall) / n,
        "wall_ms_p50": percentile(wall, 50),
        "wall_ms_p95": percentile(wall, 95),
        "tokens_per_task": sum(s.tokens_in + s.tokens_out for s in done) / n,
        "llm_calls_per_task": sum(s.llm_calls for s in done) / n,
        "tool_calls_per_task": sum(s.tool_calls for s in done) / n,
        "n": float(n),
    }
