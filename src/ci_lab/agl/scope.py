"""``RolloutScope``: one scored agent execution = one AGL rollout attempt (design I5, C2).

Every event id is deterministic — ``op_id(rollout_id, attempt, event_type, name)`` where
``name`` is a caller-supplied logical name or a per-event-type sequence number — so
re-executing a scope after a checkpoint resume journals nothing twice.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Mapping
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any, Literal

from ci_lab.contracts import RolloutJournal, RolloutKey, op_id

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
        self._started = False

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

    def reward(self, value: float, *, source: str | None = None, reason: str | None = None,
               message: str | None = None, name: str = "reward") -> str:
        """Scalar AGL ``RewardData`` event (telemetry; authoritative scores come from ASSERT)."""
        data = {"value": float(value), "message": message, "source": source, "reason": reason}
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

    def __enter__(self) -> RolloutScope:
        if not self._started:
            self.journal.start(self.key, self.input)
            self._started = True
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
            status: Literal["succeeded", "failed"] = "failed" if exc is not None else (self.outcome or "succeeded")
            self.journal.finish(self.key, status)
        finally:
            if self._tokens:
                current_rollout.reset(self._tokens.pop())

    async def __aenter__(self) -> RolloutScope:
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
