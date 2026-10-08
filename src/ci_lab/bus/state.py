"""Bus fold state and append-time invariants I0-I6 (bus contract v2 §2).

``BusState`` is the per-topic fold over entries; ``apply`` is pure and deterministic and
``check`` must pass BEFORE an entry is appended (``prev``/``hash`` may still be empty).
Derived views are computed lazily from ``entries`` so replaying the same entries always
yields an equal state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any

from ci_lab.bus import ids
from ci_lab.bus.types import Entry, Kind, ManifestBody, Role, StudentCorrection

__all__ = ["TERMINAL_KINDS", "WRITERS", "BusInvariantError", "BusState", "apply", "check", "is_run_topic"]

WRITERS: Mapping[Kind, frozenset[Role]] = MappingProxyType({
    "verdict": frozenset({"judge"}), "commit": frozenset({"judge"}), "reject": frozenset({"judge"}),
    "vote": frozenset({"voter"}), "proposal": frozenset({"student", "adversary"}),
    "rubric_patch": frozenset({"hardener"}),
    **{k: frozenset({"orchestrator"}) for k in ("manifest", "intent", "outcome", "abort", "exploit", "note")},
})
TERMINAL_KINDS: frozenset[Kind] = frozenset({"commit", "reject", "abort"})


class BusInvariantError(ValueError):
    """Append rejected: ``code`` is ``"I0"``..``"I6"`` or ``"SEQ"`` (dense seq / topic mismatch)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _b(entry: Entry) -> Any:
    """Body with an ``Any`` view (kind already dispatched)."""
    return entry.body


def is_run_topic(topic: str) -> bool:
    """``<run>/_run`` (run-level manifest/plan topic)."""
    return ids.topic_task(topic) is None


@dataclass(frozen=True)
class BusState:
    topic: str | None = None
    entries: tuple[Entry, ...] = ()

    @property
    def next_seq(self) -> int:
        return len(self.entries)

    @property
    def head(self) -> str:
        return self.entries[-1].hash if self.entries else ""

    def _of(self, kind: Kind) -> Mapping[int, Entry]:
        return MappingProxyType({e.seq: e for e in self.entries if e.kind == kind})

    @cached_property
    def manifest(self) -> ManifestBody | None:
        m = self._of("manifest")
        return _b(next(iter(m.values()))) if m else None

    @cached_property
    def proposals(self) -> Mapping[int, Entry]:
        return self._of("proposal")

    @cached_property
    def verdicts(self) -> Mapping[int, Entry]:
        return self._of("verdict")

    @cached_property
    def intents(self) -> Mapping[int, Entry]:
        return self._of("intent")

    @cached_property
    def outcomes(self) -> Mapping[int, Entry]:
        """intent seq -> its outcome entry."""
        return MappingProxyType({e.ref: e for e in self.entries if e.kind == "outcome"})

    @cached_property
    def terminal(self) -> Entry | None:
        return next((e for e in self.entries if e.kind in TERMINAL_KINDS), None)

    @cached_property
    def commit(self) -> Entry | None:
        return next((e for e in self.entries if e.kind == "commit"), None)

    def votes_for(self, proposal_seq: int) -> tuple[Entry, ...]:
        return tuple(e for e in self.entries if e.kind == "vote" and e.ref == proposal_seq)

    def proposal_by_id(self, proposal: str) -> Entry | None:
        return next((e for e in self.proposals.values() if _b(e).proposal == proposal), None)

    def intents_without_outcome(self) -> tuple[Entry, ...]:
        return tuple(e for s, e in self.intents.items() if s not in self.outcomes)

    def outcome_for(self, key: str) -> Entry | None:
        """The successful outcome recorded for an effect ``key``, if any."""
        return next((o for s, o in self.outcomes.items()
                     if _b(o).ok and _b(self.entries[s]).key == key), None)

    def latest_correction(self) -> StudentCorrection | None:
        for e in reversed(self.verdicts.values()):
            prop = self.entries[e.ref]
            if _b(e).correction is not None and prop.author.role == "student":
                return _b(e).correction
        return None

    def max_proposed_attempt(self) -> int:
        return max((ids.parse_attempt(_b(e).attempt)[1] for e in self.proposals.values()),
                   default=0)


def apply(state: BusState, entry: Entry) -> BusState:
    """Pure fold step (structure only; invariants are :func:`check`'s job)."""
    _structure(state, entry)
    return BusState(topic=entry.topic, entries=(*state.entries, entry))


def _structure(state: BusState, entry: Entry) -> None:
    if entry.seq != state.next_seq:
        raise BusInvariantError("SEQ", f"expected seq {state.next_seq}, got {entry.seq}")
    if state.topic is not None and entry.topic != state.topic:
        raise BusInvariantError("SEQ", f"entry topic {entry.topic!r} != state topic {state.topic!r}")


def _ref(state: BusState, entry: Entry, kind: Kind, code: str) -> Entry:
    if entry.ref is None or entry.ref >= state.next_seq or state.entries[entry.ref].kind != kind:
        raise BusInvariantError(code, f"{entry.kind}.ref must be a {kind} seq, got {entry.ref!r}")
    return state.entries[entry.ref]


def _fail(code: str, cond: bool, message: str) -> None:
    if cond:
        raise BusInvariantError(code, message)


def _manifest(state: BusState, e: Entry, _: ManifestBody | None) -> None:
    _fail("I0", not is_run_topic(e.topic) or e.seq != 0, "manifest must be seq 0 of the <run>/_run topic")
    _fail("I0", _b(e).run != ids.topic_run(e.topic), "manifest.run does not match the topic run")


def _intent(state: BusState, e: Entry, _: ManifestBody | None) -> None:
    key = _b(e).key
    _fail("I1", state.outcome_for(key) is not None, f"effect key {key!r} already completed")
    _fail("I1", any(_b(i).key == key for i in state.intents_without_outcome()),
          f"effect key {key!r} already in flight")


def _outcome(state: BusState, e: Entry, _: ManifestBody | None) -> None:
    intent = _ref(state, e, "intent", "I1")
    _fail("I1", _b(e).intent_seq != e.ref, "outcome.intent_seq != ref")
    _fail("I1", intent.seq in state.outcomes, f"intent {intent.seq} already has an outcome")


def _proposal(state: BusState, e: Entry, _: ManifestBody | None) -> None:
    body = _b(e)
    expect = ids.proposal_id(body.attempt, e.author.role, e.author.name)
    _fail("I6", body.proposal != expect, f"proposal id {body.proposal!r} != author id {expect!r}")
    _fail("I6", state.proposal_by_id(body.proposal) is not None, f"duplicate proposal {body.proposal!r}")


def _vote(state: BusState, e: Entry, _: ManifestBody | None) -> None:
    prop, body = _b(_ref(state, e, "proposal", "I2")), _b(e)
    _fail("I2", body.proposal != prop.proposal, "vote.proposal != referenced proposal")
    _fail("I2", body.rubric_version != prop.rubric_version, "vote.rubric_version != proposal.rubric_version")
    _fail("I2", any((_b(v).voter, _b(v).criterion) == (body.voter, body.criterion)
                    for v in state.votes_for(e.ref)),
          f"voter {body.voter!r} already voted on criterion {body.criterion!r}")


def _verdict(state: BusState, e: Entry, manifest: ManifestBody | None) -> None:
    prop, body = _b(_ref(state, e, "proposal", "I3")), _b(e)
    for f in ("proposal", "attempt", "rubric_version"):
        _fail("I3", getattr(body, f) != getattr(prop, f), f"verdict.{f} != referenced proposal")
    votes = []
    for s in body.votes:
        v = state.entries[s] if s < state.next_seq else None
        _fail("I3", v is None or v.kind != "vote" or v.ref != e.ref, f"verdict cites {s}, not a vote on {e.ref}")
        votes.append(_b(v))
    if body.decision == "commit":
        m = manifest or state.manifest
        _fail("I3", m is None, "commit verdict needs the run manifest for quorum")
        distinct = {v.voter for v in votes if v.answered}
        _fail("I3", len(distinct) < m.quorum,
              f"quorum {len(distinct)} distinct voters < {m.quorum}")


def _commit(state: BusState, e: Entry, _: ManifestBody | None) -> None:
    verdict = _ref(state, e, "verdict", "I4")
    vb, body = _b(verdict), _b(e)
    _fail("I4", vb.decision != "commit", "commit must reference a verdict with decision commit")
    _fail("I4", body.verdict_seq != e.ref or body.proposal != vb.proposal,
          "commit does not match its verdict")
    prop = state.entries[verdict.ref]
    _fail("I4", prop.author.role != "student", "only student proposals can be committed")
    _fail("I4", body.artifact != _b(prop).artifact, "commit artifact != proposal artifact")
    _fail("I4", state.commit is not None, "task already committed")


def _rubric_patch(state: BusState, e: Entry, _: ManifestBody | None) -> None:
    n, top = _b(e).applies_from_attempt, state.max_proposed_attempt()
    _fail("I5", n is not None and n <= top, f"rubric_patch applies_from_attempt {n} <= proposed attempt {top}")


_CHECKS: Mapping[str, Callable[[BusState, Entry, ManifestBody | None], None]] = MappingProxyType({
    "manifest": _manifest, "intent": _intent, "outcome": _outcome, "proposal": _proposal, "vote": _vote,
    "verdict": _verdict, "commit": _commit, "rubric_patch": _rubric_patch,
})


def check(state: BusState, entry: Entry, *, manifest: ManifestBody | None = None) -> None:
    """Raise :class:`BusInvariantError` unless ``entry`` may be appended to ``state``.

    ``manifest`` supplies the run manifest (from ``<run>/_run``) for quorum checks on task topics.
    """
    _structure(state, entry)
    role, kind = entry.author.role, entry.kind
    _fail("I6", role not in WRITERS[kind], f"role {role!r} may not write {kind!r}")
    _fail("I0", is_run_topic(entry.topic) and entry.seq == 0 and kind != "manifest",
          "first entry of the run topic must be the manifest")
    if state.terminal is not None and kind != "note":
        raise BusInvariantError("I4", f"only notes after terminal {state.terminal.kind} at seq {state.terminal.seq}")
    if (fn := _CHECKS.get(kind)) is not None:
        fn(state, entry, manifest)
