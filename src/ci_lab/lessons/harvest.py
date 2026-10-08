"""Harvest adapters → :class:`ci_lab.rulespec.Trajectory` (design §13.3 step 1, B2/B3, C12/C15).

Every adapter assigns ``family`` (case/intent family; holdout unit, N2), ``slice`` (time or suite),
``pin`` (oracle+evaluator pin), ``trusted`` and ``split`` ∈ {evolve, usage}. Sealed or unknown
splits are refused loudly (logged + counted in :class:`HarvestStats`), never ingested.

Sources and the formats they read (other modules are read by *format*, never imported):

* ``assert``: :func:`from_transcript` (``contracts.Transcript`` + ``Violation`` list) and the JSONL
  interchange read by :func:`harvest_assert` (one case per line, see docs/lessons.md);
* ``spans``: versioned span records (``ci_lab.telemetry`` JSONL) or plain span dicts — OpenInference
  AGENT/TOOL spans (``openinference.span.kind``, ``tool.name``, ``input.value``, ``output.value``)
  and MAF ``execute_tool`` spans (``gen_ai.tool.*``), grouped per ``ci.case`` span;
* ``agl``: ``FileRolloutJournal`` per-rollout JSONL (``start``/``event``/``finish`` records);
* ``calibrate``: the human-labelled dataset ``evals/datasets/order_support.yaml``;
* ``usage``: spans or AGL journals of real traffic — always reduced (B3), ``trusted`` only with a
  human label.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ci_lab.lessons.common import (
    INJECTION_SUITES,
    SealedSplitError,
    check_split,
    clean_ident,
    clean_rule_ids,
    digest,
    injection_suspect,
    log,
    read_jsonl,
    stable_id,
)
from ci_lab.lessons.reduce import reduce_trajectory
from ci_lab.rulespec import (
    GUARD_RESULT_KEY,
    TEXT_MAX_BYTES,
    Outcome,
    Trajectory,
    TrajectoryStep,
)

SOURCES = ("assert", "spans", "agl", "usage", "calibrate")
DEFAULT_PIN = "unpinned"
PASS_THRESHOLD = 0.5
_RAW_RESULT_CHARS = 2000
_VARIANT_RE = re.compile(
    r"(?:[-_.~#](?:(?:v|p|t|r)\d+|(?:para|paraphrase|var|variant|trial|rep|seed|aug|meta)\d*))+$", re.IGNORECASE)

Oracle = Callable[[Any], Sequence[Any]]  # duck: SafetyOracle.check(Transcript) -> [Violation]


@dataclass
class HarvestStats:
    harvested: int = 0
    sealed: int = 0
    invalid: int = 0
    injection_suspect: int = 0
    untrusted: int = 0
    refused: list[str] = field(default_factory=list)  # digests of refused record ids (no raw ids)

    def refuse(self, what: str, exc: Exception) -> None:
        self.sealed += 1
        self.refused.append(digest(what))
        log.error("lessons harvest REFUSED %s: %s", digest(what), exc)

    def as_dict(self) -> dict[str, Any]:
        return {"harvested": self.harvested, "sealed_refused": self.sealed, "invalid": self.invalid,
                "injection_suspect": self.injection_suspect, "untrusted": self.untrusted,
                "refused": self.refused}


def family_of(case_id: str) -> str:
    """Case family: the case id with paraphrase/variant/trial suffixes stripped (N2)."""
    base = str(case_id).strip().lower()
    prev = None
    while prev != base:
        prev, base = base, _VARIANT_RE.sub("", base)
    return base or str(case_id).strip().lower()


def time_slice(ts: Any) -> str | None:
    """UTC day of a unix timestamp (seconds or nanoseconds)."""
    try:
        f = float(ts)
    except (TypeError, ValueError):
        return None
    if f <= 0:
        return None
    if f > 1e14:
        f /= 1e9
    return datetime.fromtimestamp(f, UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- step building

def _loads(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _cap(text: str, limit: int) -> str:
    b = text.encode("utf-8", "ignore")
    return text if len(b) <= limit else b[:limit].decode("utf-8", "ignore")


def normalize_result(raw: Any) -> tuple[dict[str, Any] | None, str]:
    """Structured tool result + status (ok/error/blocked)."""
    v = _loads(raw)
    if v is None:
        return None, "ok"
    if isinstance(v, Mapping):
        d = dict(v)
        if GUARD_RESULT_KEY in d and len(d) == 1:
            return d, "blocked"
        err = (d.get("error") or d.get("error_code") or d.get("error_type")
               or str(d.get("status", "")).lower() == "error")
        return d, "error" if err else "ok"
    if isinstance(v, str):
        if v.strip().lower().startswith(("error", "exception")):
            return {"error": _cap(v, 300)}, "error"
        return {"value": _cap(v, _RAW_RESULT_CHARS)}, "ok"
    return {"value": v}, "ok"


class StepBuilder:
    def __init__(self) -> None:
        self.steps: list[TrajectoryStep] = []
        self._names: dict[str, str] = {}

    def user(self) -> None:  # user text never enters trajectories (B3)
        self.steps.append(TrajectoryStep(i=len(self.steps), kind="user"))

    def call(self, name: str, args: Any, call_id: str | None) -> None:
        cid = call_id or f"call-{len(self.steps)}"
        self._names[cid] = name
        a = _loads(args)
        self.steps.append(TrajectoryStep(i=len(self.steps), kind="tool_call", tool=name, call_id=cid,
                                         args=dict(a) if isinstance(a, Mapping) else {}))

    def result(self, call_id: str | None, raw: Any, name: str | None = None, status: str | None = None) -> None:
        res, st = normalize_result(raw)
        if status in ("ok", "error", "blocked"):
            st = status
        cid = call_id or (self.steps[-1].call_id if self.steps else None)
        self.steps.append(TrajectoryStep(i=len(self.steps), kind="tool_result",
                                         tool=name or self._names.get(cid or ""), call_id=cid,
                                         result=res, status=st))  # type: ignore[arg-type]

    def response(self, text: Any) -> None:
        t = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
        if not t:
            return
        t = _cap(t, TEXT_MAX_BYTES)
        self.steps.append(TrajectoryStep(i=len(self.steps), kind="response", text=t, text_digest=digest(t)))


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def steps_from_transcript(messages: Sequence[Any], tool_calls: Sequence[Any]) -> tuple[TrajectoryStep, ...]:
    """Ordered steps from a ``contracts.Transcript``-shaped object (tool calls placed in their turn)."""
    by_turn: dict[int, list[Any]] = {}
    for c in tool_calls:
        by_turn.setdefault(int(_get(c, "turn", 0) or 0), []).append(c)
    b = StepBuilder()
    turn, emitted = -1, set()

    def flush(t: int) -> None:
        if t in emitted:
            return
        emitted.add(t)
        for c in by_turn.get(t, ()):
            name = str(_get(c, "name", "") or "")
            cid = _get(c, "call_id")
            b.call(name, _get(c, "arguments", {}), cid)
            b.result(cid or b.steps[-1].call_id, _get(c, "result"), name, _get(c, "status"))

    for m in messages:
        role = str(_get(m, "role", "")).lower()
        if role == "user":
            if turn >= 0:
                flush(turn)
            turn += 1
            b.user()
        elif role == "assistant":
            flush(max(turn, 0))
            b.response(_get(m, "content"))
    for t in sorted(by_turn):
        flush(t)
    return tuple(b.steps)


def steps_from_chat(messages: Sequence[Any]) -> tuple[TrajectoryStep, ...]:
    """Ordered steps from OpenAI-style chat messages (user/assistant tool_calls/tool)."""
    b = StepBuilder()
    for m in messages:
        if not isinstance(m, Mapping):
            continue
        role = str(m.get("role", "")).lower()
        if role == "user":
            b.user()
        elif role == "assistant":
            for tc in m.get("tool_calls") or ():
                fn = tc.get("function") if isinstance(tc, Mapping) else None
                if isinstance(fn, Mapping):
                    b.call(str(fn.get("name") or ""), fn.get("arguments"), tc.get("id"))
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, Mapping))
            if content:
                b.response(content)
        elif role == "tool":
            b.result(m.get("tool_call_id"), m.get("content"), m.get("name"))
    return tuple(b.steps)


# ---------------------------------------------------------------- trajectory assembly

def make_trajectory(*, source: str, split: Any, case_id: str, steps: Sequence[TrajectoryStep],
                    default_split: str | None = None, family: str | None = None, slice: str | None = None,
                    suite: str = "", pin: str | None = None, trial: Any = 0, passed: bool | None = None,
                    oracle_rules: Iterable[Any] = (), rubric_fails: Iterable[Any] = (),
                    error_class: str | None = None, human_label: str | None = None,
                    trusted: bool = True, labels: Iterable[str] = (), guard_on: bool = False,
                    extra_id: str = "") -> Trajectory:
    """Assemble + police one trajectory. Raises :class:`SealedSplitError`."""
    sp = check_split(split, source=source, default=default_split)
    rules = clean_rule_ids(oracle_rules)
    suspect = injection_suspect(rules, suite, *labels)
    hl = human_label if human_label in ("good", "bad") else None
    if source == "usage":
        trusted = hl is not None
    traj = Trajectory(
        id=stable_id("tr-", source, case_id, trial, extra_id), source=source, split=sp,  # type: ignore[arg-type]
        family=family or family_of(case_id), slice=slice or suite or "unsliced", pin=pin or DEFAULT_PIN,
        trusted=trusted, steps=tuple(steps), guard_on=guard_on,
        outcome=Outcome(passed=passed, oracle_rules=rules, rubric_fails=clean_rule_ids(rubric_fails),
                        error_class=clean_ident(error_class), human_label=hl, injection_suspect=suspect))
    if source == "usage" or not trusted or suspect:
        traj = reduce_trajectory(traj)
    if source == "usage":
        traj = traj.model_copy(update={
            "family": "u-" + digest(traj.family, 12)[7:],
            "slice": clean_ident(traj.slice) or "s-" + digest(traj.slice, 12)[7:],
            "pin": clean_ident(traj.pin) or "p-" + digest(traj.pin, 12)[7:]})
    return traj


def _passed(passed: Any, score: Any, rules: Sequence[str]) -> bool | None:
    if isinstance(passed, bool):
        return passed and not rules
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score) >= PASS_THRESHOLD and not rules
    return False if rules else None


def from_transcript(transcript: Any, violations: Iterable[Any] = (), *, split: str, suite: str = "",
                    family: str | None = None, slice: str | None = None, pin: str | None = None,
                    passed: bool | None = None, score: float | None = None, rubric_fails: Iterable[str] = (),
                    human_label: str | None = None, trial: int = 0, source: str = "assert",
                    trusted: bool = True, error_class: str | None = None) -> Trajectory:
    """ASSERT/oracle adapter: ``contracts.Transcript`` (+ ``Violation`` list) → Trajectory."""
    rules = [str(_get(v, "rule_id", "")) for v in violations]
    case_id = str(_get(transcript, "case_id", "") or "case")
    steps = steps_from_transcript(_get(transcript, "messages", ()) or (), _get(transcript, "tool_calls", ()) or ())
    return make_trajectory(source=source, split=split, case_id=case_id, steps=steps, family=family, slice=slice,
                           suite=suite, pin=pin, trial=trial, passed=_passed(passed, score, clean_rule_ids(rules)),
                           oracle_rules=rules, rubric_fails=rubric_fails, human_label=human_label,
                           trusted=trusted, error_class=error_class)


@dataclass
class HarvestOptions:
    default_split: str | None = None   # explicit operator assertion, e.g. "evolve"
    pin: str | None = None
    slice: str | None = None           # overrides record slices (e.g. night id)
    labels: Mapping[str, str] = field(default_factory=dict)  # usage: trace/case id -> good|bad
    oracle: Any = None                 # duck SafetyOracle with .check(Transcript)


def _attempt(stats: HarvestStats, what: str, build: Callable[[], Trajectory | None]) -> Trajectory | None:
    try:
        t = build()
    except SealedSplitError as exc:
        stats.refuse(what, exc)
        return None
    except (ValueError, TypeError, KeyError) as exc:
        stats.invalid += 1
        log.warning("lessons harvest skipped invalid record %s (%s)", digest(what), type(exc).__name__)
        return None
    if t is not None:
        stats.harvested += 1
        stats.injection_suspect += t.outcome.injection_suspect
        stats.untrusted += not t.trusted
    return t


# ---------------------------------------------------------------- assert JSONL

def harvest_assert(path: Path, opts: HarvestOptions | None = None,
                   stats: HarvestStats | None = None) -> list[Trajectory]:
    """ASSERT/oracle interchange JSONL: ``{case_id, suite, split, trial?, family?, pin?, slice?, score?,
    passed?, transcript: {messages, tool_calls[{call_id,name,arguments,result,turn}]},
    violations: [{rule_id,severity,detail}], rubric_fails?: [ids]}``."""
    opts, stats = opts or HarvestOptions(), stats if stats is not None else HarvestStats()
    out = []
    for p in _files(path):
        for rec in read_jsonl(p):
            def build(rec: dict[str, Any] = rec) -> Trajectory:
                tr = rec.get("transcript") or {}
                return from_transcript(
                    {"case_id": rec.get("case_id") or tr.get("case_id"), "messages": tr.get("messages") or (),
                     "tool_calls": tr.get("tool_calls") or ()},
                    rec.get("violations") or (), split=rec.get("split") or opts.default_split or "",
                    suite=str(rec.get("suite") or ""), family=rec.get("family"),
                    slice=opts.slice or rec.get("slice") or rec.get("suite") or time_slice(rec.get("ts")),
                    pin=opts.pin or rec.get("pin"), passed=rec.get("passed"), score=rec.get("score"),
                    rubric_fails=rec.get("rubric_fails") or (), trial=int(rec.get("trial") or 0),
                    error_class=rec.get("error_class"))
            if t := _attempt(stats, str(rec.get("case_id")), build):
                out.append(t)
    return out


def _files(path: Path, pattern: str = "*.jsonl") -> list[Path]:
    return sorted(p for p in path.rglob(pattern) if p.is_file()) if path.is_dir() else [path]


# ---------------------------------------------------------------- spans

@dataclass(frozen=True)
class _Span:
    span_id: str
    parent_id: str
    trace_id: str
    name: str
    attrs: Mapping[str, Any]
    start: int

    @property
    def kind(self) -> str:
        return str(self.attrs.get("openinference.span.kind") or "").upper()


def _span(rec: Mapping[str, Any]) -> _Span | None:
    sid = rec.get("spanId", rec.get("span_id"))
    if sid in (None, ""):
        return None
    start = rec.get("startTimeUnixNano", rec.get("start_time_ns", rec.get("start_time", 0)))
    try:
        start_i = int(start or 0)
    except (TypeError, ValueError):
        start_i = 0
    return _Span(span_id=str(sid), parent_id=str(rec.get("parentSpanId", rec.get("parent_span_id")) or ""),
                 trace_id=str(rec.get("traceId", rec.get("trace_id")) or ""), name=str(rec.get("name") or ""),
                 attrs=dict(rec.get("attributes") or {}), start=start_i)


def _is_case(s: _Span) -> bool:
    return s.name == "ci.case" or "ci.case_id" in s.attrs


def _is_tool(s: _Span) -> bool:
    return s.kind == "TOOL" or s.attrs.get("gen_ai.operation.name") == "execute_tool"


