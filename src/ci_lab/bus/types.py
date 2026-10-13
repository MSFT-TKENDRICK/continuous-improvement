"""Typed bus entries (bus contract v2 §2): kinds, roles, strict bodies, ``Entry``, hashing, visibility.

Bodies are frozen dataclasses validated on construction and on ``from_json`` (unknown or missing
fields rejected, types checked, lists -> tuples, mappings -> sorted read-only mappings,
scores/confidences in ``[0, 1]``). Visibility is derived from ``(kind, author.role)`` and never
supplied by callers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import types as _pytypes
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from functools import cache
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ci_lab.bus import ids
from ci_lab.tools.paths import PathRejected, normalize_rel

__all__ = [
    "BODY_TYPES", "GENESIS", "KINDS", "MEASURES", "ROLES", "AbortBody", "ArtifactRef", "Author", "Body",
    "BodyError", "CommitBody", "CriterionResult", "Decision", "Entry", "ExploitBody", "IntentBody", "Kind",
    "ManifestBody", "Measure", "NoteBody", "OutcomeBody", "ProposalBody", "RejectBody", "Role",
    "RubricPatchBody", "SoftPref", "StudentCorrection", "VerdictBody", "VoteBody", "canonical_json",
    "entry_hash", "visibility",
]

Kind = Literal["manifest", "intent", "outcome", "proposal", "vote", "verdict", "commit", "reject", "abort",
               "exploit", "rubric_patch", "note"]
Role = Literal["orchestrator", "planner", "examiner", "student", "voter", "judge", "adversary", "hardener"]
Measure = Literal["deterministic", "assert", "s1", "llm"]
Decision = Literal["commit", "revise", "reject"]
SoftPref = Literal["student", "adversary", "tie"]
KINDS: tuple[Kind, ...] = get_args(Kind)
ROLES: tuple[Role, ...] = get_args(Role)
MEASURES: tuple[Measure, ...] = get_args(Measure)
GENESIS = "0" * 64  # conventional ``prev`` of seq 0

_HEX64 = re.compile(r"[0-9a-f]{64}")
_SUMMARY_MAX = 400
_CORRECTION_MAX = 800


class BodyError(ValueError):
    """A bus body or entry failed strict validation."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise BodyError(msg)


def _hex64(value: str | None, what: str) -> None:
    _require(value is None or bool(_HEX64.fullmatch(value)), f"{what}: expected sha256 hex, got {value!r}")


def _nonneg(value: int | None, what: str, minimum: int = 0) -> None:
    _require(value is None or value >= minimum, f"{what}: must be >= {minimum}, got {value!r}")


def _id(fn: Any, value: str | None, what: str) -> Any:
    if value is None:
        return None
    try:
        return fn(value)
    except ids.IdError as exc:
        raise BodyError(f"{what}: {exc}") from None


def _json_value(v: Any, path: str) -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        _require(math.isfinite(v), f"{path}: non-finite float")
        return v
    if isinstance(v, (list, tuple)):
        return [_json_value(x, f"{path}[{i}]") for i, x in enumerate(v)]
    if isinstance(v, Mapping):
        _require(all(isinstance(k, str) for k in v), f"{path}: mapping keys must be str")
        return {k: _json_value(v[k], f"{path}.{k}") for k in sorted(v)}
    raise BodyError(f"{path}: not JSON-serializable ({type(v).__name__})")


def _coerce(tp: Any, v: Any, path: str) -> Any:
    if tp is Any:
        return _json_value(v, path)
    origin, args = get_origin(tp), get_args(tp)
    if origin is Literal:
        _require(any(type(v) is type(a) and v == a for a in args), f"{path}: {v!r} not in {args}")
        return v
    if origin in (Union, _pytypes.UnionType):
        if v is None:
            _require(type(None) in args, f"{path}: None not allowed")
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(inner, v, path)
    if origin is tuple:
        _require(isinstance(v, (list, tuple)), f"{path}: expected list, got {type(v).__name__}")
        return tuple(_coerce(args[0], x, f"{path}[{i}]") for i, x in enumerate(v))
    if origin is Mapping:
        _require(isinstance(v, Mapping), f"{path}: expected object, got {type(v).__name__}")
        _require(all(isinstance(k, str) for k in v), f"{path}: keys must be str")
        return _pytypes.MappingProxyType({k: _coerce(args[1], v[k], f"{path}.{k}") for k in sorted(v)})
    if tp is float:
        _require(isinstance(v, (int, float)) and not isinstance(v, bool), f"{path}: expected number, got {v!r}")
        _require(math.isfinite(v), f"{path}: non-finite float")
        return float(v)
    if tp in (bool, int, str):
        _require(isinstance(v, tp) and (tp is bool or not isinstance(v, bool)),
                 f"{path}: expected {tp.__name__}, got {type(v).__name__}")
        return v
    if isinstance(tp, type) and is_dataclass(tp):
        if isinstance(v, tp):
            return v
        _require(isinstance(v, Mapping), f"{path}: expected {tp.__name__} object")
        return tp.from_json(v)  # type: ignore[attr-defined]
    raise BodyError(f"{path}: unsupported type {tp!r}")


def _dump(v: Any) -> Any:
    if hasattr(v, "to_json"):
        return v.to_json()
    if isinstance(v, Mapping):
        return {k: _dump(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_dump(x) for x in v]
    return v


@cache
def _hints(cls: type) -> dict[str, Any]:
    hints = get_type_hints(cls)
    return {f.name: hints[f.name] for f in fields(cls)}


class _Strict:
    _UNIT: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        cls = type(self)
        for name, tp in _hints(cls).items():
            object.__setattr__(self, name, _coerce(tp, getattr(self, name), f"{cls.__name__}.{name}"))
        for name in self._UNIT:
            x = getattr(self, name)
            _require(x is None or 0.0 <= x <= 1.0, f"{cls.__name__}.{name}: must be in [0, 1], got {x!r}")
        self._check()

    def _check(self) -> None:
        return None

    def to_json(self) -> dict[str, Any]:
        return {f.name: _dump(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self:
        _require(isinstance(data, Mapping), f"{cls.__name__}: expected object, got {type(data).__name__}")
        fs = fields(cls)  # type: ignore[arg-type]
        unknown = set(data) - {f.name for f in fs}
        _require(not unknown, f"{cls.__name__}: unknown fields {sorted(unknown)}")
        missing = [f.name for f in fs if f.name not in data and f.default is MISSING
                   and f.default_factory is MISSING]
        _require(not missing, f"{cls.__name__}: missing fields {missing}")
        return cls(**data)


@dataclass(frozen=True)
class Author(_Strict):
    role: Role
    name: str
    model: str | None = None

    def _check(self) -> None:
        _id(ids.validate_name, self.name, "Author.name")


@dataclass(frozen=True)
class ArtifactRef(_Strict):
    path: str  # relative to the topic artifact dir; content-addressed
    sha256: str
    bytes: int

    def _check(self) -> None:
        try:
            _require(normalize_rel(self.path) == self.path, f"ArtifactRef.path not normalized: {self.path!r}")
        except PathRejected as exc:
            raise BodyError(f"ArtifactRef.path: {exc}") from None
        _hex64(self.sha256, "ArtifactRef.sha256")
        _nonneg(self.bytes, "ArtifactRef.bytes")


@dataclass(frozen=True)
class StudentCorrection(_Strict):
    text: str  # sanitized (contract §6)
    attempt: str

    def _check(self) -> None:
        _require(len(self.text) <= _CORRECTION_MAX, f"StudentCorrection.text > {_CORRECTION_MAX} chars")
        _id(ids.parse_attempt, self.attempt, "StudentCorrection.attempt")


@dataclass(frozen=True)
class CriterionResult(_Strict):
    passed: bool | None
    score: float | None
    required: bool
    oracle: bool
    votes: int
    _UNIT = ("score",)

    def _check(self) -> None:
        _nonneg(self.votes, "CriterionResult.votes")


@dataclass(frozen=True)
class ManifestBody(_Strict):
    run: str
    created: str
    code_rev: str
    config_sha256: str
    graph_sha256: str | None
    max_parallel: int
    voters: tuple[str, ...]
    quorum: int
    extra: Mapping[str, str] = field(default_factory=dict)

    def _check(self) -> None:
        _id(ids.run_id, self.run, "ManifestBody.run")
        _hex64(self.config_sha256, "ManifestBody.config_sha256")
        _hex64(self.graph_sha256, "ManifestBody.graph_sha256")
        _nonneg(self.max_parallel, "ManifestBody.max_parallel", 1)
        _nonneg(self.quorum, "ManifestBody.quorum", 1)
        for v in self.voters:
            _id(ids.validate_name, v, "ManifestBody.voters")
        _require(len(set(self.voters)) == len(self.voters), "ManifestBody.voters: duplicates")


@dataclass(frozen=True)
class IntentBody(_Strict):
    action: str
    key: str
    attempt: str | None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def _check(self) -> None:
        _require(bool(self.action) and bool(self.key), "IntentBody: action and key must be non-empty")
        _id(ids.parse_attempt, self.attempt, "IntentBody.attempt")


@dataclass(frozen=True)
class OutcomeBody(_Strict):
    intent_seq: int
    ok: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def _check(self) -> None:
        _nonneg(self.intent_seq, "OutcomeBody.intent_seq")


@dataclass(frozen=True)
class ProposalBody(_Strict):
    proposal: str
    attempt: str
    rubric_version: str
    artifact: ArtifactRef
    summary: str

    def _check(self) -> None:
        attempt, _, _ = _id(ids.parse_proposal, self.proposal, "ProposalBody.proposal")
        _require(attempt == self.attempt, f"ProposalBody: proposal {self.proposal!r} not of attempt {self.attempt!r}")
        _id(ids.parse_rubric_version, self.rubric_version, "ProposalBody.rubric_version")
        _require(len(self.summary) <= _SUMMARY_MAX, f"ProposalBody.summary > {_SUMMARY_MAX} chars")


@dataclass(frozen=True)
class VoteBody(_Strict):
    proposal: str
    rubric_version: str
    voter: str
    measure: Measure
    criterion: str | None
    passed: bool | None
    score: float | None
    confidence: float | None
    reasons: tuple[str, ...] = ()
    _UNIT = ("score", "confidence")

    def _check(self) -> None:
        _id(ids.parse_proposal, self.proposal, "VoteBody.proposal")
        _id(ids.parse_rubric_version, self.rubric_version, "VoteBody.rubric_version")
        _id(ids.validate_name, self.voter, "VoteBody.voter")
        _require(self.criterion is None or bool(self.criterion), "VoteBody.criterion: empty")

    @property
    def answered(self) -> bool:
        """False for an abstention (no pass/fail and no score)."""
        return self.passed is not None or self.score is not None


@dataclass(frozen=True)
class VerdictBody(_Strict):
    proposal: str
    attempt: str
    rubric_version: str
    decision: Decision
    score: float
    criteria: Mapping[str, CriterionResult]
    votes: tuple[int, ...]
    correction: StudentCorrection | None
    escalated: bool
    _UNIT = ("score",)

    def _check(self) -> None:
        attempt, _, _ = _id(ids.parse_proposal, self.proposal, "VerdictBody.proposal")
        _require(attempt == self.attempt, f"VerdictBody: proposal {self.proposal!r} not of attempt {self.attempt!r}")
        _id(ids.parse_rubric_version, self.rubric_version, "VerdictBody.rubric_version")
        _require(all(s >= 0 for s in self.votes), "VerdictBody.votes: negative seq")
        _require(len(set(self.votes)) == len(self.votes), "VerdictBody.votes: duplicates")


@dataclass(frozen=True)
class CommitBody(_Strict):
    proposal: str
    verdict_seq: int
    artifact: ArtifactRef

    def _check(self) -> None:
        _id(ids.parse_proposal, self.proposal, "CommitBody.proposal")
        _nonneg(self.verdict_seq, "CommitBody.verdict_seq")


@dataclass(frozen=True)
class RejectBody(_Strict):
    attempt: str
    reason: str

    def _check(self) -> None:
        _id(ids.parse_attempt, self.attempt, "RejectBody.attempt")


@dataclass(frozen=True)
class AbortBody(_Strict):
    attempt: str | None
    reason: str

    def _check(self) -> None:
        _id(ids.parse_attempt, self.attempt, "AbortBody.attempt")


@dataclass(frozen=True)
class ExploitBody(_Strict):
    attempt: str
    adversary_proposal: str
    student_proposal: str | None
    gamer: str
    soft_pref: SoftPref
    soft_pass_adversary: bool
    oracle_invalid: tuple[str, ...] = ()

    def _check(self) -> None:
        _id(ids.parse_attempt, self.attempt, "ExploitBody.attempt")
        for what, value, role in (("adversary_proposal", self.adversary_proposal, "adversary"),
                                  ("student_proposal", self.student_proposal, "student")):
            parsed = _id(ids.parse_proposal, value, f"ExploitBody.{what}")
            _require(parsed is None or parsed[1] == role, f"ExploitBody.{what}: role must be {role}")
        _require(bool(self.gamer), "ExploitBody.gamer: empty")


@dataclass(frozen=True)
class RubricPatchBody(_Strict):
    rubric_id: str
    from_version: str
    to_version: str
    applies_from_attempt: int | None
    applies_from_epoch: int | None
    diff_sha256: str
    metrics: Mapping[str, float]
    accepted: bool

    def _check(self) -> None:
        for what in ("from_version", "to_version"):
            rid, _ = _id(ids.parse_rubric_version, getattr(self, what), f"RubricPatchBody.{what}")
            _require(rid == self.rubric_id, f"RubricPatchBody.{what}: not a version of {self.rubric_id!r}")
        _nonneg(self.applies_from_attempt, "RubricPatchBody.applies_from_attempt", 1)
        _nonneg(self.applies_from_epoch, "RubricPatchBody.applies_from_epoch")
        _hex64(self.diff_sha256, "RubricPatchBody.diff_sha256")


@dataclass(frozen=True)
class NoteBody(_Strict):
    text: str
    data: Mapping[str, Any] = field(default_factory=dict)


Body = (ManifestBody | IntentBody | OutcomeBody | ProposalBody | VoteBody | VerdictBody | CommitBody | RejectBody
        | AbortBody | ExploitBody | RubricPatchBody | NoteBody)

BODY_TYPES: Mapping[Kind, type[_Strict]] = _pytypes.MappingProxyType({
    "manifest": ManifestBody, "intent": IntentBody, "outcome": OutcomeBody, "proposal": ProposalBody,
    "vote": VoteBody, "verdict": VerdictBody, "commit": CommitBody, "reject": RejectBody, "abort": AbortBody,
    "exploit": ExploitBody, "rubric_patch": RubricPatchBody, "note": NoteBody,
})

_ENTRY_FIELDS = ("seq", "topic", "kind", "author", "ref", "body", "ts", "prev", "hash")


@dataclass(frozen=True)
class Entry:
    """One WAL line. ``prev``/``hash`` may be ``""`` before sealing (append-time ``check``)."""

    seq: int
    topic: str
    kind: Kind
    author: Author
    ref: int | None
    body: Body
    ts: str
    prev: str = ""
    hash: str = ""

    def __post_init__(self) -> None:
        _require(type(self.seq) is int and self.seq >= 0, f"Entry.seq: bad {self.seq!r}")
        _id(ids.validate_topic, self.topic, "Entry.topic")
        _require(self.kind in KINDS, f"Entry.kind: unknown {self.kind!r}")
        _require(isinstance(self.author, Author), "Entry.author: expected Author")
        _require(self.ref is None or (type(self.ref) is int and 0 <= self.ref < self.seq),
                 f"Entry.ref: must be an earlier seq, got {self.ref!r}")
        _require(type(self.body) is BODY_TYPES[self.kind],
                 f"Entry.body: {self.kind} needs {BODY_TYPES[self.kind].__name__}, got {type(self.body).__name__}")
        _require(isinstance(self.ts, str), "Entry.ts: expected str")
        for what in ("prev", "hash"):
            value = getattr(self, what)
            _require(isinstance(value, str) and (value == "" or bool(_HEX64.fullmatch(value))),
                     f"Entry.{what}: expected '' or sha256 hex")

    def to_json(self) -> dict[str, Any]:
        return {"seq": self.seq, "topic": self.topic, "kind": self.kind, "author": self.author.to_json(),
                "ref": self.ref, "body": self.body.to_json(), "ts": self.ts, "prev": self.prev, "hash": self.hash}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Entry:
        _require(isinstance(data, Mapping), "Entry: expected object")
        _require(set(data) == set(_ENTRY_FIELDS), f"Entry: fields must be exactly {_ENTRY_FIELDS}, got {sorted(data)}")
        kind = data["kind"]
        _require(kind in KINDS, f"Entry.kind: unknown {kind!r}")
        return cls(seq=data["seq"], topic=data["topic"], kind=kind, author=Author.from_json(data["author"]),
                   ref=data["ref"], body=BODY_TYPES[kind].from_json(data["body"]),  # type: ignore[arg-type]
                   ts=data["ts"], prev=data["prev"], hash=data["hash"])

    def sealed(self) -> Entry:
        """Copy with ``hash`` = :func:`entry_hash` of this entry (``prev`` must already be set)."""
        return replace(self, hash=entry_hash(self))

    def hash_ok(self) -> bool:
        return self.hash == entry_hash(self)


def canonical_json(obj: Any) -> str:
    """Sorted keys, ``(",", ":")`` separators, ``ensure_ascii=False``; NaN/Infinity rejected."""
    return json.dumps(_dump(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def entry_hash(entry: Entry | Mapping[str, Any]) -> str:
    """sha256 hex of the canonical JSON of the entry dict without its ``hash`` field."""
    data = entry.to_json() if isinstance(entry, Entry) else dict(entry)
    data.pop("hash", None)
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


_CONTROL: frozenset[Role] = frozenset({"orchestrator", "hardener", "judge"})
_PROPOSAL: frozenset[Role] = frozenset({"orchestrator", "voter", "judge", "hardener"})
_VISIBILITY: Mapping[Kind, frozenset[Role]] = _pytypes.MappingProxyType({
    "manifest": _CONTROL, "intent": _CONTROL, "outcome": _CONTROL, "note": _CONTROL,
    "proposal": _PROPOSAL, "vote": _CONTROL, "verdict": _CONTROL, "commit": frozenset(ROLES),
    "reject": _CONTROL | {"planner"}, "abort": _CONTROL | {"planner"},
    "exploit": _CONTROL, "rubric_patch": _CONTROL,
})


def visibility(entry: Entry) -> frozenset[Role]:
    """Roles that may see ``entry`` (contract §2). Proposals by student and adversary alike are
    hidden from students (they see committed outputs only via ``commit``); the student's
    ``StudentCorrection`` reaches it only through projection, never via the verdict entry."""
    return _VISIBILITY[entry.kind]
