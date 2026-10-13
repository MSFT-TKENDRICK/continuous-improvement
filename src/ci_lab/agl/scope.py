"""``RolloutScope``: one scored agent execution = one AGL rollout attempt (design I5, C2).

Every event id is deterministic — ``op_id(rollout_id, attempt, event_type, name)`` where
``name`` is a caller-supplied logical name or a per-event-type sequence number — so
re-executing a scope after a checkpoint resume journals nothing twice.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any, Literal, Self

from opentelemetry import trace

from ci_lab import obs
from ci_lab.agl.metrics import METRIC, rollout_metrics
from ci_lab.contracts import (
    ATTR_ATTEMPT,
    ATTR_CASE,
    ATTR_EXPERIMENT,
    ATTR_ROLLOUT,
    ATTR_SCORE,
    ATTR_SPLIT,
    ATTR_TRIAL,
    ATTR_VARIANT,
    SPAN_CASE,
    RolloutJournal,
    RolloutKey,
    op_id,
)

log = logging.getLogger(__name__)

current_rollout: ContextVar[RolloutScope | None] = ContextVar("ci_lab_current_rollout", default=None)

REWARD = "reward"
SCORE = "ci.score"
MODEL_REQUEST = "model_request"
ERROR = "ci.error"


class RolloutScope:
    """Sync + async context manager: starts the rollout, records events, finishes on exit.

    On exit: ``succeeded`` unless the body raised an :class:`Exception` or :meth:`fail` was
    called (then ``failed``). ``BaseException`` exits (cancellation, KeyboardInterrupt) leave the
    rollout running so a checkpoint resume can complete it.

    Telemetry (design §12.3): the scope runs inside one ``ci.case`` span carrying
    ``agl.rollout_id``/``agl.attempt_id`` (plus case/trial/experiment/variant/split). If the
    caller (e.g. the ASSERT runner) already opened a ``ci.case`` span, that span is annotated
    instead of nesting a second one. No-op without a tracer provider (``ci_lab.obs``).
    """

    def __init__(self, journal: RolloutJournal, key: RolloutKey, input: Mapping[str, Any] | None = None, *,
                 import_proxy_events: bool = False) -> None:
        self.journal = journal
        self.key = key
        self.input = dict(input or {})
        self.import_proxy_events = import_proxy_events
        self.outcome: Literal["succeeded", "failed"] | None = None
        self._seq: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._tokens: list[Token[RolloutScope | None]] = []
        self._spans: list[AbstractContextManager[Any] | None] = []
        self.span: trace.Span = trace.INVALID_SPAN  # the ci.case span while entered
        self._started = False
        self._entered_at: float | None = None

    def __repr__(self) -> str:
        return f"RolloutScope({self.key.rollout_id!r}, attempt={self.key.attempt_id!r})"

    @property
    def rollout_id(self) -> str:
        return self.key.rollout_id

    @property
    def attempt_id(self) -> str:
        return self.key.attempt_id

    # ------------------------------------------------------------ events

    def event_id(self, event_type: str, name: str | int) -> str:
        return op_id(self.key.rollout_id, self.key.attempt_id, event_type, name)

    def emit(self, event_type: str, data: Mapping[str, Any], *, name: str | int | None = None) -> str:
        """Journal an event; ``name`` defaults to the next sequence number for ``event_type``."""
        if name is None:
            with self._lock:
                name = f"#{self._seq[event_type]}"
                self._seq[event_type] += 1
        eid = self.event_id(event_type, name)
        self.journal.event(self.key, event_type, data, event_id=eid)
        return eid

    def record_model_request(self, data: Mapping[str, Any], *, name: str | int | None = None) -> str:
        """Journal a ``model_request`` event (AGL proxy field set; see ``mirror.model_request_data``)."""
        return self.emit(MODEL_REQUEST, data, name=name)

    def record_tool_call(self, tool: str, *, wall_ms: float = 0.0,
                         name: str | int | None = None) -> str:
        """Journal one typed tool-call delta without recording arguments or output."""
        return self.emit("ci.tool_call", {"tool": str(tool), "wall_ms": float(wall_ms)}, name=name)

    def reward(self, value: float, *, source: str | None = None, reason: str | None = None,
               message: str | None = None, name: str = "reward") -> str:
        """Scalar AGL ``RewardData`` event (telemetry; authoritative scores come from ASSERT)."""
        data = {"value": float(value), "message": message, "source": source, "reason": reason}
        if name == "reward":
            self.span.set_attribute(ATTR_SCORE, float(value))
        return self.emit(REWARD, data, name=name)

    def score(self, name: str, value: float | None, **attrs: Any) -> str:
        """Named ``ci.score`` event (multi-objective / ASSERT / oracle results).

        Conventional attrs read by :mod:`ci_lab.agl.export`: ``suite``, ``category``,
        ``rule_ids``, ``rubric_scores``, ``violations`` (``[{rule_id, severity, detail}]``),
        ``excerpt``.
        """
        data = {"name": name, "value": None if value is None else float(value), **attrs}
        return self.emit(SCORE, data, name=name)

    def fail(self) -> None:
        """Mark the rollout failed on exit without raising."""
        self.outcome = "failed"

    # ------------------------------------------------------------ lifecycle

    def span_attributes(self) -> dict[str, Any]:
        return {ATTR_ROLLOUT: self.key.rollout_id, ATTR_ATTEMPT: self.key.attempt_id,
                ATTR_CASE: self.key.case_id, ATTR_TRIAL: self.key.trial,
                ATTR_EXPERIMENT: self.key.experiment_id, ATTR_VARIANT: self.key.variant,
                ATTR_SPLIT: self.input.get("split")}

    def _open_span(self) -> None:
        current = trace.get_current_span()
        if current.is_recording() and getattr(current, "name", None) == SPAN_CASE:
            obs.annotate(self.span_attributes())
            self._spans.append(None)
            self.span = current
            return
        cm = obs.span(SPAN_CASE, self.span_attributes())
        self.span = cm.__enter__()
        self._spans.append(cm)

    def _close_span(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                    tb: TracebackType | None) -> None:
        cm = self._spans.pop() if self._spans else None
        if not self._spans:
            self.span = trace.INVALID_SPAN
        if cm is not None:
            cm.__exit__(exc_type, exc, tb)

    def __enter__(self) -> Self:
        self._open_span()
        try:
            if not self._started:
                self.journal.start(self.key, self.input)
                self._started = True
        except BaseException as exc:
            self._close_span(type(exc), exc, exc.__traceback__)
            raise
        self._entered_at = time.perf_counter()
        self._tokens.append(current_rollout.set(self))
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        try:
            if exc is not None and not isinstance(exc, Exception):
                log.info("rollout %s interrupted (%s); left running", self.rollout_id, exc_type.__name__ if exc_type else "")
                return
            if exc is not None:
                self.emit(ERROR, {"type": type(exc).__name__}, name="exit")
            if self.import_proxy_events:
                importer = getattr(self.journal, "import_server_events", None)
                if importer is not None:
                    importer(self.key)
            metrics = rollout_metrics(self.journal.events(self.key))
            if self._entered_at is not None:
                metrics["wall_ms"] = max(float(metrics["wall_ms"]),
                                         (time.perf_counter() - self._entered_at) * 1000.0)
            self.emit(METRIC, metrics, name="finish")
            status: Literal["succeeded", "failed"] = "failed" if exc is not None else (self.outcome or "succeeded")
            self.journal.finish(self.key, status)
            if exc is None and status == "failed":
                self.span.set_status(trace.Status(trace.StatusCode.ERROR, "rollout failed"))
        finally:
            if self._tokens:
                current_rollout.reset(self._tokens.pop())
            self._close_span(exc_type, exc, tb)

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                        tb: TracebackType | None) -> None:
        self.__exit__(exc_type, exc, tb)

    # ------------------------------------------------------------ proxy

    def proxy_base_url(self, base: Any, mode: Literal["train", "val"] = "val") -> str:
        """OpenAI base URL through the AGL proxy; ``base`` is an AglServer, AglClient or base URL."""
        from ci_lab.agl.client import proxy_base_url

        url = base if isinstance(base, str) else base.base_url
        return proxy_base_url(url, self.key, mode)
