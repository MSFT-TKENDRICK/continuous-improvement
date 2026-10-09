"""Usage-trace harvest (design §11.3, C22, C29): trace sources -> redaction -> prompt-injection
filter -> intent clusters -> PENDING task records (``reviewed: false``).

Pending tasks are proposals for humans: they are written to
``experiments/sleep/tasks.pending.jsonl`` on a draft PR (``exp/usage-<date>/tasks``); a reviewer
authors reference/judge, flips ``reviewed: true`` and moves them into ``tasks.jsonl``. Nothing
here feeds SkillOpt or a reflection prompt directly — ``harvest.load_reviewed_tasks`` refuses
any unreviewed row.

Sources (all read-only, best-effort parsers of sibling formats):
* :class:`AglJournalSource` — AGL rollout journal ``<root>/<rollout_id>.jsonl``
  (``{"v":1,"kind":"start"|"event"|"finish",...}`` records);
* :class:`SpansJsonlSource` — ``<run_dir>/telemetry/spans-*.jsonl`` (versioned span records);
* :class:`ArtifactsDirSource` — a downloaded GitHub Actions artifacts directory holding either.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

PENDING_FORMAT = "skillopt_sleep.tasks.v1"
_MAX_TEXT = 2000
_MAX_LINE = 2 * 1024 * 1024


# ------------------------------------------------------------------ trace model

@dataclass
class UsageTrace:
    source: str
    trace_id: str
    intent: str = ""
    target: str | None = None
    tools: list[str] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)  # injection screening only; never emitted
    violations: list[str] = field(default_factory=list)
    started: float | None = None  # unix seconds
    split: str | None = None


@runtime_checkable
class TraceSource(Protocol):
    name: str

    def traces(self) -> Iterator[UsageTrace]: ...


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "text", "value"):
            if key in value:
                return _text(value[key])
        parts = value.get("parts")
        if isinstance(parts, list):
            return " ".join(_text(p) for p in parts)
        return ""
    if isinstance(value, list):
        return " ".join(_text(v) for v in value)
    return "" if value is None else str(value)


def _first_user(messages: Any) -> str:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except ValueError:
            return messages
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, Mapping) and str(m.get("role", "")).lower() == "user":
                return _text(m)
    return ""


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "rb") as fh:
        for raw in fh:
            if len(raw) > _MAX_LINE or not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue  # truncated crash line / foreign content
            if isinstance(rec, dict):
                yield rec


class AglJournalSource:
    """Rollouts from an AGL journal directory (or explicit ``*.jsonl`` files)."""

    name = "agl"

    def __init__(self, paths: Path | Sequence[Path]) -> None:
        self.paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]

    def _files(self) -> list[Path]:
        out: list[Path] = []
        for p in self.paths:
            out += sorted(p.glob("*.jsonl")) if p.is_dir() else [p] if p.is_file() else []
        return out

    def traces(self) -> Iterator[UsageTrace]:
        for f in self._files():
            tr = UsageTrace(source=self.name, trace_id=f.stem)
            seen = False
            for rec in _jsonl(f):
                kind = rec.get("kind")
                if rec.get("rollout_id"):
                    tr.trace_id = str(rec["rollout_id"])
                if kind == "start":
                    seen = True
                    inp = rec.get("input") if isinstance(rec.get("input"), Mapping) else {}
                    tr.started = rec.get("ts") if isinstance(rec.get("ts"), (int, float)) else tr.started
                    tr.intent = tr.intent or next(
                        (_text(inp[k]) for k in ("intent", "prompt", "query", "user", "input", "message")
                         if inp.get(k)), "") or _first_user(inp.get("messages"))
                    tr.target = tr.target or next((str(inp[k]) for k in ("target", "agent", "project")
                                                   if inp.get(k)), None)
                    split = inp.get("dataset_split") or inp.get("split")
                    tr.split = str(split) if split else tr.split
                elif kind == "event":
                    et = str(rec.get("event_type") or "")
                    data = rec.get("data") if isinstance(rec.get("data"), Mapping) else {}
                    if "tool" in et:
                        if data.get("name"):
                            tr.tools.append(str(data["name"]))
                        if data.get("result") is not None:
                            tr.tool_outputs.append(_text(data["result"])[:_MAX_TEXT])
                    if "violation" in et and data.get("rule_id"):
                        tr.violations.append(str(data["rule_id"]))
                    for v in data.get("violations") or []:
                        rid = v.get("rule_id") if isinstance(v, Mapping) else v
                        if rid:
                            tr.violations.append(str(rid))
                    if not tr.intent and et == "model_request":
                        tr.intent = _first_user(data.get("messages"))
            if seen:
                yield tr


class SpansJsonlSource:
    """Traces from ``<run_dir>/telemetry/spans-*.jsonl`` (grouped by ``traceId``). With the
    default non-sensitive exporter (C29) prompts are absent, so such traces only count."""

    name = "spans"

    def __init__(self, run_dir_or_files: Path | Sequence[Path]) -> None:
        p = run_dir_or_files
        self.paths = [Path(p)] if isinstance(p, (str, Path)) else [Path(x) for x in p]

    def files(self) -> list[Path]:
        out: list[Path] = []
        for p in self.paths:
            if p.is_dir():
                d = p / "telemetry" if (p / "telemetry").is_dir() else p
                out += sorted(d.glob("spans-*.jsonl*"))
            elif p.is_file():
                out.append(p)
        return out

    def traces(self) -> Iterator[UsageTrace]:
        by_trace: dict[str, UsageTrace] = {}
        for f in self.files():
            for rec in _jsonl(f):
                tid = str(rec.get("traceId") or "")
                if not tid:
                    continue
                tr = by_trace.setdefault(tid, UsageTrace(source=self.name, trace_id=tid))
                attrs = rec.get("attributes") if isinstance(rec.get("attributes"), Mapping) else {}
                try:
                    start = int(rec.get("startTimeUnixNano") or 0) / 1e9
                except (TypeError, ValueError):
                    start = 0.0
                if start and (tr.started is None or start < tr.started):
                    tr.started = start
                tool = attrs.get("gen_ai.tool.name")
                if tool:
                    tr.tools.append(str(tool))
                    if attrs.get("gen_ai.tool.call.result") is not None:
                        tr.tool_outputs.append(_text(attrs["gen_ai.tool.call.result"])[:_MAX_TEXT])
                if not tr.intent and attrs.get("gen_ai.input.messages"):
                    tr.intent = _first_user(attrs["gen_ai.input.messages"])
                tr.target = tr.target or (str(attrs["gen_ai.agent.name"]) if attrs.get("gen_ai.agent.name")
                                          else None)
                split = attrs.get("ci.split")
                if split and not tr.split:
                    tr.split = str(split)
                for ev in rec.get("events") or []:
                    if isinstance(ev, Mapping) and "violation" in str(ev.get("name", "")):
                        ea = ev.get("attributes") if isinstance(ev.get("attributes"), Mapping) else {}
                        if ea.get("rule_id"):
                            tr.violations.append(str(ea["rule_id"]))
        yield from by_trace.values()


class ArtifactsDirSource:
    """A downloaded GitHub Actions artifacts directory: span files (``spans-*.jsonl``) and AGL
    journal files (any other ``*.jsonl`` with journal records) anywhere below it."""

    name = "artifacts"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def traces(self) -> Iterator[UsageTrace]:
        spans = sorted(self.root.rglob("spans-*.jsonl*"))
        journals = [p for p in sorted(self.root.rglob("*.jsonl")) if not p.name.startswith("spans-")
                    and not p.name.startswith("tasks")]
        if spans:
            yield from SpansJsonlSource(spans).traces()
        if journals:
            yield from AglJournalSource(journals).traces()


def parse_source(spec: str) -> TraceSource:
    """``agl:PATH`` | ``spans:RUN_DIR`` | ``artifacts:DIR`` (CLI ``--source``)."""
    kind, sep, path = spec.partition(":")
    if not sep or not path:
        raise ValueError(f"bad trace source {spec!r} (want agl:PATH, spans:RUN_DIR or artifacts:DIR)")
    if kind == "agl":
        return AglJournalSource(Path(path))
    if kind == "spans":
        return SpansJsonlSource(Path(path))
    if kind == "artifacts":
        return ArtifactsDirSource(Path(path))
    raise ValueError(f"unknown trace source kind {kind!r}")


# ------------------------------------------------------------------ redaction (PII, secrets)

_SECRET_RES = [
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b(bearer|token|api[_-]?key|password|secret)\b(\s*[:=]\s*|\s+)[^\s,;\"']{6,}"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}"),
]
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ORDER = re.compile(r"\b[A-Z]{2,4}-\d{4,8}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_STREET = (r"St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way|Ct|Court|Pl|Place"
           r"|Pkwy|Parkway|Hwy|Highway|Ter|Terrace")
_ADDRESS = re.compile(rf"\b\d{{1,6}}\s+(?:[A-Z][A-Za-z]+\s+){{1,4}}(?:{_STREET})\b\.?"
                      r"(?:,?\s+(?:Apt|Suite|Unit|#)\s*\w+)?(?:,\s*[A-Z][A-Za-z .]+)?(?:,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?")
_ZIP = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")


def _known_identities() -> list[str]:
    """Names/emails/phones/addresses from ``order_support.data`` when that module is present."""
    try:
        from order_support import data  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional sibling module
        return []
    out: list[str] = []
    for order in (getattr(data, "ORDERS", {}) or {}).values():
        cust = order.get("customer") if isinstance(order, Mapping) else None
        if isinstance(cust, Mapping):
            out += [str(v) for k, v in cust.items() if k in ("name", "email", "phone", "address") and v]
    return out


class Redactor:
    """Deterministic PII/secret scrubber. Order ids become salted hashes (stable, so repeated
    intents still cluster); everything else becomes a typed placeholder."""

    def __init__(self, *, salt: str = "ci-sleep-usage", identities: Iterable[str] | None = None) -> None:
        self.salt = salt
        names = list(_known_identities() if identities is None else identities)
        names = sorted({n for n in names if len(n) >= 3}, key=len, reverse=True)
        self._ident = re.compile("|".join(re.escape(n) for n in names), re.IGNORECASE) if names else None

    def order_token(self, order_id: str) -> str:
        return "order-" + hashlib.sha256(f"{self.salt}|{order_id}".encode()).hexdigest()[:8]

    def text(self, value: str) -> str:
        s = str(value)
        if self._ident is not None:
            s = self._ident.sub("<customer>", s)
        s = _EMAIL.sub("<email>", s)
        for rx in _SECRET_RES:
            s = rx.sub("<secret>", s)
        s = _ORDER.sub(lambda m: self.order_token(m.group(0)), s)
        s = _ADDRESS.sub("<address>", s)
        s = _CARD.sub("<card>", s)
        s = _PHONE.sub("<phone>", s)
        s = _ZIP.sub("<zip>", s)
        return s

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, Mapping):
            return {k: self.value(v) for k, v in value.items()}
        return value


# ------------------------------------------------------------------ prompt-injection filter

_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in (
    (r"\b(ignore|disregard|forget|override)\b.{0,40}"
     r"\b(previous|prior|above|earlier|all|your|system)\b.{0,20}"
     r"\b(instructions?|rules?|polic(y|ies)|prompts?)\b"),
    r"\byou are now\b", r"\bnew (system )?instructions?\b", r"\bsystem prompt\b",
    r"<\|?(im_start|im_end|system)\|?>", r"\[(system|assistant)\]", r"^\s*(system|assistant)\s*:",
    r"\b(do not|don't) (tell|inform|mention)\b.{0,30}\b(user|customer)\b",
    r"\b(as an ai|as the assistant)\b.{0,40}\b(must|should)\b",
    r"\b(issue|approve|process)\b.{0,30}\brefund\b.{0,40}\b(without|skip|no need)\b.{0,30}\bverif",
    r"\bjailbreak\b|\bDAN mode\b|\bdeveloper mode\b",
)]


def injection_reason(trace: UsageTrace) -> str | None:
    """Why a trace must be dropped (oracle ``injection.*`` violation or instruction-like text in
    tool output / user turn), or None."""
    for rid in trace.violations:
        if rid.lower().startswith("injection") or "injection" in rid.lower():
            return f"oracle:{rid}"
    for label, texts in (("tool_output", trace.tool_outputs), ("intent", [trace.intent])):
        for t in texts:
            for rx in _INJECTION_RES:
                if rx.search(t or ""):
                    return f"{label}:{rx.pattern[:32]}"
    return None


# ------------------------------------------------------------------ pending tasks

def _normalize(intent: str) -> str:
    s = intent.lower()
    s = re.sub(r"order-[0-9a-f]{8}", "<order>", s)
    s = re.sub(r"\d+", "<n>", s)
    s = re.sub(r"[^\w<> ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class UsageHarvest:
    tasks: list[dict[str, Any]]
    stats: dict[str, int]


def pending_tasks(sources: Iterable[TraceSource], *, project: str = "harness-editing",
                  aliases: Iterable[str] = (), redactor: Redactor | None = None,
                  since: float | None = None, max_tasks: int = 50,
                  accept_unlabeled: bool = True) -> UsageHarvest:
    """Redact -> filter -> cluster usage traces into ``reviewed: false`` task proposals."""
    red = redactor or Redactor()
    names = {project, *aliases}
    stats = {"traces": 0, "old": 0, "no_intent": 0, "split_dropped": 0, "injection_dropped": 0,
             "other_target": 0, "clusters": 0}
    clusters: dict[str, dict[str, Any]] = {}
    for src in sources:
        for tr in src.traces():
            stats["traces"] += 1
            if since is not None and tr.started is not None and tr.started <= since:
                stats["old"] += 1
                continue
            if tr.split and tr.split not in ("evolve", "usage"):
                stats["split_dropped"] += 1  # never turn heldout/ood/aa cases into tasks (C15)
                continue
            if (tr.target and tr.target not in names) or (not tr.target and not accept_unlabeled):
                stats["other_target"] += 1
                continue
            if injection_reason(tr):
                stats["injection_dropped"] += 1
                continue
            intent = red.text(tr.intent).strip()[:_MAX_TEXT]
            if not intent:
                stats["no_intent"] += 1
                continue
            key = hashlib.sha256(_normalize(intent).encode()).hexdigest()[:12]
            c = clusters.setdefault(key, {"intent": intent, "count": 0, "tools": set(), "sources": set(),
                                          "sessions": [], "first": tr.started, "last": tr.started})
            c["count"] += 1
            c["tools"].update(tr.tools)
            c["sources"].add(tr.source)
            if len(c["sessions"]) < 20:
                c["sessions"].append(hashlib.sha256(f"{red.salt}|{tr.trace_id}".encode()).hexdigest()[:16])
            if tr.started is not None:
                c["first"] = tr.started if c["first"] is None else min(c["first"], tr.started)
                c["last"] = tr.started if c["last"] is None else max(c["last"], tr.started)
    ranked = sorted(clusters.items(), key=lambda kv: (-kv[1]["count"], kv[0]))[:max_tasks]
    stats["clusters"] = len(clusters)
    tasks = [{
        "id": f"usage-{key}", "project": project, "intent": c["intent"],
        "reference": "", "reference_kind": "none", "judge": {},
        "tags": ["origin:usage", *sorted(f"source:{s}" for s in c["sources"])],
        "source_sessions": c["sessions"], "reviewed": False,
        "usage": {"count": c["count"], "tools": sorted(c["tools"]), "first_seen": c["first"],
                  "last_seen": c["last"]},
        "review_note": "Author reference + judge, then set reviewed:true and move to tasks.jsonl (C22).",
    } for key, c in ranked]
    return UsageHarvest(tasks=tasks, stats=stats)


# ------------------------------------------------------------------ spans artifact redaction (C29)

def _sensitive_attr(key: str) -> bool:
    try:
        from ci_lab.telemetry.jsonl import (
            is_sensitive_attr,  # type: ignore[import-not-found]
        )
    except Exception:  # noqa: BLE001 - telemetry (M12) is optional here
        return key.startswith("gen_ai.") and (key.endswith((".messages", ".content", ".arguments", ".result"))
                                              or key.startswith(("gen_ai.prompt", "gen_ai.completion"))
                                              or key == "gen_ai.system_instructions")
    return bool(is_sensitive_attr(key))


def redact_span_record(rec: Mapping[str, Any], red: Redactor) -> dict[str, Any]:
    out = dict(rec)
    attrs = rec.get("attributes") if isinstance(rec.get("attributes"), Mapping) else {}
    out["attributes"] = {k: red.value(v) for k, v in attrs.items() if not _sensitive_attr(k)}
    events = []
    for ev in rec.get("events") or []:
        if not isinstance(ev, Mapping) or str(ev.get("name", "")).startswith("gen_ai."):
            continue
        ea = ev.get("attributes") if isinstance(ev.get("attributes"), Mapping) else {}
        events.append({**ev, "name": red.text(str(ev.get("name", ""))),
                       "attributes": {k: red.value(v) for k, v in ea.items() if not _sensitive_attr(k)}})
    out["events"] = events
    if isinstance(rec.get("status"), Mapping):
        out["status"] = {k: red.value(v) for k, v in rec["status"].items()}
    if isinstance(rec.get("name"), str):
        out["name"] = red.text(rec["name"])
    return out


def redact_spans_jsonl(inputs: Iterable[Path], out_path: Path, *, redactor: Redactor | None = None) -> dict[str, int]:
    """Copy span JSONL files through the redactor into one file (the CI artifact)."""
    red = redactor or Redactor()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_files = n_spans = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for p in inputs:
            n_files += 1
            for rec in _jsonl(Path(p)):
                fh.write(json.dumps(redact_span_record(rec, red), ensure_ascii=False, separators=(",", ":")) + "\n")
                n_spans += 1
    return {"files": n_files, "spans": n_spans}
