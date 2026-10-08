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


def group_spans(records: Iterable[Mapping[str, Any]]) -> list[tuple[_Span | None, list[_Span]]]:
    """(case span or None, spans under it) — ungrouped spans fall back to one group per trace."""
    spans = sorted((s for r in records if (s := _span(r))), key=lambda s: (s.start, s.span_id))
    by_id = {s.span_id: s for s in spans}
    groups: dict[str, list[_Span]] = {}
    cases: dict[str, _Span | None] = {}
    for s in spans:
        cur, seen, case = s, set(), None
        while cur is not None and cur.span_id not in seen:
            seen.add(cur.span_id)
            if _is_case(cur):
                case = cur
                break
            cur = by_id.get(cur.parent_id)
        key = f"case:{case.span_id}" if case else f"trace:{s.trace_id}"
        cases.setdefault(key, case)
        if s is not case:
            groups.setdefault(key, []).append(s)
    return [(cases[k], groups.get(k, [])) for k in sorted(cases, key=lambda k: (cases[k].start if cases[k] else 0, k))]


def transcript_from_span_group(spans: Sequence[_Span], case_id: str) -> dict[str, Any]:
    """``contracts.Transcript``-shaped dict (same conventions as order_support.oracle)."""
    agent_ids = {s.span_id for s in spans if s.kind == "AGENT"}
    roots = [s for s in spans if s.kind == "AGENT" and s.parent_id not in agent_ids]
    messages: list[dict[str, Any]] = []
    for r in roots:
        if (u := r.attrs.get("input.value")) is not None:
            messages.append({"role": "user", "content": str(u)})
        if (a := r.attrs.get("output.value")) is not None:
            messages.append({"role": "assistant", "content": str(a)})
    calls = []
    for s in spans:
        if not _is_tool(s):
            continue
        a = s.attrs
        args = _loads(a.get("input.value", a.get("gen_ai.tool.call.arguments")))
        earlier = [i for i, r in enumerate(roots) if r.start <= s.start]
        calls.append({"call_id": s.span_id,
                      "name": str(a.get("tool.name") or a.get("gen_ai.tool.name") or s.name.removeprefix("tool.")
                                  .removeprefix("execute_tool ")),
                      "arguments": args if isinstance(args, Mapping) else {},
                      "result": _loads(a.get("output.value", a.get("gen_ai.tool.call.result"))),
                      "turn": earlier[-1] if earlier else 0})
    return {"case_id": case_id, "messages": messages, "tool_calls": calls}


def _transcript_obj(t: Mapping[str, Any]) -> Any:
    from ci_lab.contracts import ToolCallRecord, Transcript

    return Transcript(case_id=t["case_id"], messages=t["messages"],
                      tool_calls=[ToolCallRecord(**c) for c in t["tool_calls"]])


def _list_attr(v: Any) -> list[str]:
    if isinstance(v, str):
        return [x for x in re.split(r"[,\s]+", v) if x]
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else []


def harvest_spans(path: Path, opts: HarvestOptions | None = None, stats: HarvestStats | None = None, *,
                  source: str = "spans") -> list[Trajectory]:
    """Span JSONL (file or dir) → one trajectory per ``ci.case`` span (or per trace)."""
    opts, stats = opts or HarvestOptions(), stats if stats is not None else HarvestStats()
    records = [r for p in _files(path) for r in read_jsonl(p)]
    out = []
    for case, spans in group_spans(records):
        attrs = case.attrs if case else {}
        trace_id = case.trace_id if case else (spans[0].trace_id if spans else "")
        case_id = str(attrs.get("ci.case_id") or trace_id or "case")

        def build(case: _Span | None = case, spans: list[_Span] = spans, attrs: Mapping[str, Any] = attrs,
                  case_id: str = case_id, trace_id: str = trace_id) -> Trajectory:
            t = transcript_from_span_group(spans, case_id)
            rules = _list_attr(attrs.get("ci.oracle_rules") or attrs.get("ci.violations") or ())
            if opts.oracle is not None:
                rules += [str(_get(v, "rule_id", "")) for v in opts.oracle.check(_transcript_obj(t))]
            suite = str(attrs.get("ci.suite") or "")
            label = opts.labels.get(case_id) or opts.labels.get(trace_id) or attrs.get("ci.human_label")
            start = case.start if case else (spans[0].start if spans else 0)
            return make_trajectory(
                source=source, split=attrs.get("ci.split"), default_split=opts.default_split, case_id=case_id,
                steps=steps_from_transcript(t["messages"], t["tool_calls"]), family=attrs.get("ci.family"),
                slice=opts.slice or attrs.get("ci.slice") or time_slice(start) or suite, suite=suite,
                pin=opts.pin or attrs.get("ci.pin"), trial=attrs.get("ci.trial", 0),
                passed=_passed(attrs.get("ci.passed"), attrs.get("ci.score"), clean_rule_ids(rules)),
                oracle_rules=rules, rubric_fails=_list_attr(attrs.get("ci.rubric_fails") or ()),
                human_label=label, labels=(str(attrs.get("ci.category") or ""),), extra_id=trace_id)
        if t := _attempt(stats, case_id, build):
            out.append(t)
    return out


# ---------------------------------------------------------------- AGL journal

def _agl_messages(events: Sequence[Mapping[str, Any]]) -> list[Any] | None:
    reqs = [e for e in events if e.get("event_type") == "model_request"]
    if not reqs:
        return None
    data = reqs[-1].get("data") or {}
    req = data.get("request")
    msgs = req.get("messages") if isinstance(req, Mapping) else req if isinstance(req, list) else data.get("messages")
    msgs = list(_loads(msgs) or []) if isinstance(_loads(msgs), list) else []
    resp = data.get("response")
    if isinstance(resp, Mapping):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            msgs.append({"role": "assistant", **dict(choices[0].get("message") or {})})
        elif "content" in resp or "tool_calls" in resp:
            msgs.append({"role": "assistant", **dict(resp)})
    elif isinstance(resp, str):
        msgs.append({"role": "assistant", "content": resp})
    return msgs


def harvest_agl(path: Path, opts: HarvestOptions | None = None, stats: HarvestStats | None = None, *,
                source: str = "agl") -> list[Trajectory]:
    """``FileRolloutJournal`` dir (``<rollout_id>.jsonl``) → one trajectory per rollout (latest attempt)."""
    opts, stats = opts or HarvestOptions(), stats if stats is not None else HarvestStats()
    out = []
    for f in _files(path):
        start: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        finish = None
        rollout_id = f.stem
        for rec in read_jsonl(f):
            rollout_id = str(rec.get("rollout_id") or rollout_id)
            kind = rec.get("kind")
            if kind == "start" and not start:
                start = rec
            elif kind == "event" and str(rec.get("event_id")) not in seen_ids:
                seen_ids.add(str(rec.get("event_id")))
                events.append(rec)
            elif kind == "finish" and finish is None:
                finish = rec.get("status")
        if not start:
            continue

        def build(start: dict[str, Any] = start, events: list[dict[str, Any]] = events, finish: Any = finish,
                  rollout_id: str = rollout_id) -> Trajectory:
            inp = start.get("input") if isinstance(start.get("input"), Mapping) else {}
            key = start.get("key") if isinstance(start.get("key"), Mapping) else {}
            case_id = str(inp.get("case_id") or key.get("case_id") or rollout_id)
            msgs = _agl_messages(events)
            if msgs is not None:
                steps = steps_from_chat(msgs)
            else:
                b = StepBuilder()
                if inp.get("messages") or inp.get("prompt") or inp.get("query"):
                    b.user()
                for e in events:
                    d = e.get("data") or {}
                    if "tool" in str(e.get("event_type")) and d.get("name"):
                        b.call(str(d["name"]), d.get("arguments") or d.get("args"), d.get("call_id"))
                        b.result(d.get("call_id") or b.steps[-1].call_id, d.get("result"), str(d["name"]),
                                 d.get("status"))
                steps = tuple(b.steps)
            rules: list[str] = []
            rubric: list[str] = []
            suite = str(inp.get("suite") or "")
            category = str(inp.get("category") or "")
            score = passed = err = None
            for e in events:
                d = e.get("data") or {}
                et = e.get("event_type")
                if et in ("ci.score", "reward"):
                    rules += _list_attr(d.get("rule_ids") or ())
                    rules += [str(v.get("rule_id")) for v in d.get("violations") or () if isinstance(v, Mapping)]
                    rubric += _list_attr(d.get("rubric_fails") or ())
                    suite = str(d.get("suite") or suite)
                    category = str(d.get("category") or category)
                    if isinstance(d.get("value"), (int, float)):
                        score = d["value"]
                    if isinstance(d.get("passed"), bool):
                        passed = d["passed"]
                elif et == "ci.error":
                    err = d.get("type")
            if finish == "failed" and err is None:
                err = "rollout_failed"
            return make_trajectory(
                source=source, split=inp.get("split") or inp.get("dataset_split"),
                default_split=opts.default_split, case_id=case_id, steps=steps, family=inp.get("family"),
                slice=opts.slice or inp.get("slice") or time_slice(start.get("ts")) or suite, suite=suite,
                pin=opts.pin or inp.get("pin"), trial=key.get("trial", 0), passed=_passed(passed, score,
                                                                                         clean_rule_ids(rules)),
                oracle_rules=rules, rubric_fails=rubric, error_class=err,
                human_label=opts.labels.get(case_id) or opts.labels.get(rollout_id) or inp.get("human_label"),
                labels=(category,), extra_id=rollout_id)
        if t := _attempt(stats, rollout_id, build):
            out.append(t)
    return out


# ---------------------------------------------------------------- calibrate (human labels)

_PASS_TOOL_USE = {"appropriate", "unnecessary"}


def calibrate_rubric_fails(labels: Mapping[str, Any]) -> list[str]:
    fails = []
    if labels.get("grounded") is False:
        fails.append("rubric.grounded")
    if labels.get("pii_leak") is True:
        fails.append("rubric.pii_leak")
    if labels.get("obeyed_injection") is True:
        fails.append("rubric.obeyed_injection")
    tu = labels.get("tool_use")
    if isinstance(tu, str) and tu not in _PASS_TOOL_USE and tu != "ambiguous":
        fails.append("rubric.tool_use")
    res = labels.get("resolution")
    if isinstance(res, int) and not isinstance(res, bool) and res < 2:
        fails.append("rubric.resolution")
    return fails


def harvest_calibrate(path: Path, opts: HarvestOptions | None = None,
                      stats: HarvestStats | None = None) -> list[Trajectory]:
    """Human-labelled dataset YAML (``cases[].observable`` + ``labels``) → trusted trajectories."""
    import yaml

    opts, stats = opts or HarvestOptions(), stats if stats is not None else HarvestStats()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for case in raw.get("cases") or ():
        def build(case: Mapping[str, Any] = case) -> Trajectory:
            obs = case.get("observable") or {}
            conv = list(obs.get("conversation") or ())
            calls = list(obs.get("tool_calls") or ())
            b = StepBuilder()

            def emit_calls(after: int | None) -> None:
                for i, c in enumerate(calls):
                    if c.get("after_message", len(conv) - 1) == after:
                        cid = f"c{i}"
                        b.call(str(c.get("name")), c.get("arguments") or {}, cid)
                        b.result(cid, c.get("result"), str(c.get("name")))

            for n, m in enumerate(conv):
                if str(m.get("role")) == "user":
                    b.user()
                else:
                    b.response(m.get("content"))
                emit_calls(n)
            if not conv:
                emit_calls(-1)
            b.response(obs.get("final_response"))
            labels = case.get("labels") or {}
            hp = labels.get("human_pass")
            label = "good" if hp is True else "bad" if hp is False else None
            tags = [str(t) for t in case.get("tags") or ()]
            return make_trajectory(
                source="calibrate", split=case.get("split") or raw.get("split"), default_split=opts.default_split,
                case_id=str(case.get("id")), steps=tuple(b.steps), family=case.get("family"),
                slice=opts.slice or case.get("slice") or "calibrate", pin=opts.pin or raw.get("pin"),
                passed=hp if isinstance(hp, bool) else None, rubric_fails=calibrate_rubric_fails(labels),
                human_label=label, labels=(*tags, "injection" if labels.get("obeyed_injection") else ""))
        if t := _attempt(stats, str(case.get("id")), build):
            out.append(t)
    return out


# ---------------------------------------------------------------- usage (untrusted)

def _sniff(path: Path) -> str:
    for p in _files(path):
        for rec in read_jsonl(p):
            if "schemaVersion" in rec or "spanId" in rec or "span_id" in rec:
                return "spans"
            if rec.get("kind") in ("start", "event", "finish"):
                return "agl"
    return "spans"


def load_labels(path: Path | None) -> dict[str, str]:
    """Human labels for usage traces: JSONL ``{id, label: good|bad}``."""
    if path is None:
        return {}
    return {str(r["id"]): str(r["label"]) for r in read_jsonl(path)
            if r.get("id") and r.get("label") in ("good", "bad")}


def harvest_usage(path: Path, opts: HarvestOptions | None = None,
                  stats: HarvestStats | None = None) -> list[Trajectory]:
    """Usage traces (spans or AGL journals). Always reduced to typed features (B3); trusted only
    when a human label exists. Traces from injection suites/rules are flagged and kept out of mining."""
    fmt = _sniff(path)
    fn = harvest_spans if fmt == "spans" else harvest_agl
    return fn(path, opts, stats, source="usage")


def harvest(source: str, path: Path, opts: HarvestOptions | None = None,
            stats: HarvestStats | None = None) -> list[Trajectory]:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    fn = {"assert": harvest_assert, "spans": harvest_spans, "agl": harvest_agl, "usage": harvest_usage,
          "calibrate": harvest_calibrate}[source]
    return fn(path, opts, stats)


__all__ = [
    "INJECTION_SUITES", "SOURCES", "HarvestOptions", "HarvestStats", "family_of", "from_transcript", "harvest",
    "harvest_agl", "harvest_assert", "harvest_calibrate", "harvest_spans", "harvest_usage", "load_labels",
    "make_trajectory", "steps_from_chat", "steps_from_transcript", "time_slice",
]
