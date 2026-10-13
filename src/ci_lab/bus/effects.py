"""Exactly-once-observable effects on the bus (bus contract v2 §4).

``async with effect(bus, topic, action, key, author) as eff:`` follows
``ledger.outbox.FileOutbox.arun_once``: a dedicated cross-process lock for ``(topic, key)`` is
held for the whole effect. A completed (``ok``) outcome for ``key`` short-circuits
(``eff.skipped``; ``eff.result`` = its detail). Otherwise an ``intent`` is appended (or the
in-flight one left by a crashed holder is reused) and the optional ``reconcile(key)`` hook may
report the detail of an effect that already happened (recorded, body skipped). Leaving the block
appends ``outcome{ok: true, detail: eff.result}``; an exception appends ``outcome{ok: false}``
and propagates. ``async with`` cannot skip its body, so callers test ``eff.skipped``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ci_lab.bus.types import Author, Entry, IntentBody, OutcomeBody
from ci_lab.ledger.lock import lock_for

if TYPE_CHECKING:
    from ci_lab.bus.wal import AgentBus

__all__ = ["Effect", "Reconcile", "effect", "in_flight"]

Reconcile = Callable[[str], Awaitable[Mapping[str, Any] | None]]
_MAX_ERROR = 2000


@dataclass
class Effect:
    topic: str
    action: str
    key: str
    intent: Entry | None = None
    skipped: bool = False
    reconciled: bool = False
    result: Mapping[str, Any] = field(default_factory=dict)

    def set_result(self, detail: Mapping[str, Any]) -> None:
        """JSON detail recorded in the ``ok`` outcome."""
        if self.skipped:
            raise RuntimeError(f"effect {self.key!r} already completed; nothing to record")
        self.result = detail


def in_flight(bus: AgentBus, topic: str) -> tuple[Entry, ...]:
    """Intents without an outcome (running elsewhere, or crashed mid-effect)."""
    return bus.state(topic).intents_without_outcome()


async def _outcome(bus: AgentBus, eff: Effect, author: Author, ok: bool, detail: Mapping[str, Any]) -> None:
    assert eff.intent is not None
    await bus.append(eff.topic, "outcome", author, OutcomeBody(intent_seq=eff.intent.seq, ok=ok, detail=detail),
                     ref=eff.intent.seq)


@asynccontextmanager
async def effect(bus: AgentBus, topic: str, action: str, key: str, author: Author,
                 detail: Mapping[str, Any] | None = None, *, attempt: str | None = None,
                 reconcile: Reconcile | None = None, lock_timeout: float = 3600.0) -> AsyncIterator[Effect]:
    eff = Effect(topic=topic, action=action, key=key)
    async with lock_for(bus.wal_path(topic), suffix=f"effect|{key}", timeout=lock_timeout, lock_dir=bus.lock_dir):
        st = bus.state(topic)
        if (done := st.outcome_for(key)) is not None:
            outcome: Any = done.body
            eff.skipped, eff.intent, eff.result = True, st.entries[outcome.intent_seq], outcome.detail
            yield eff
            return
        eff.intent = next((i for i in st.intents_without_outcome() if getattr(i.body, "key", None) == key), None)
        if eff.intent is None:
            eff.intent = await bus.append(topic, "intent", author, IntentBody(
                action=action, key=key, attempt=attempt, detail=detail or {}))
        found = await reconcile(key) if reconcile is not None else None
        if found is not None:
            await _outcome(bus, eff, author, True, found)
            eff.skipped = eff.reconciled = True
            eff.result = found
            yield eff
            return
        try:
            yield eff
        except BaseException as exc:
            await _outcome(bus, eff, author, False, {"error": f"{type(exc).__name__}: {exc}"[:_MAX_ERROR]})
            raise
        await _outcome(bus, eff, author, True, eff.result)
