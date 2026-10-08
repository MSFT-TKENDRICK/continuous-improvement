"""Task harvest for the nightly sleep cycle (design §6, C12, C15).

Sources:
* the human-reviewed tasks file ``experiments/sleep/tasks.jsonl`` (``skillopt_sleep.tasks.v1``
  header line, then one task per line, every task ``"reviewed": true``);
* Agent Lightning journal exports (``ci_lab.agl.export`` rows). Only the ``evolve`` dataset
  split may feed SkillOpt; any other split is a hard error (C15), never a silent skip.

Output tasks are deduped, split train/val by a stable hash of their id (val gates the
SkillOpt pre-filter), have every rule-judge op validated (upstream silently passes unknown
ops) and carry no raw tool output (C12).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import TaskRecord

TASKS_FORMAT = "skillopt_sleep.tasks.v1"
# skillopt_sleep.judges ops (0.2.0). Anything else is rejected, not passed.
KNOWN_OPS = frozenset({"section_present", "regex", "max_chars", "min_chars", "contains", "tool_called"})
ORDER_SUPPORT_TOOLS = frozenset({"lookup_order", "search_kb", "issue_refund", "escalate_to_human"})
ALLOWED_DATASET_SPLIT = "evolve"
DATASET_SPLITS = frozenset({"evolve", "heldout", "ood", "aa"})
# Internal reference kinds: routes scoring through OrderSupportSleepBackend.judge (oracle +
# validated ops) instead of upstream replay's direct score_rule_judge shortcut.
INTERNAL_KIND = {"rule": "ci_rule", "rubric": "ci_assert", "none": "ci_assert",
                 "ci_rule": "ci_rule", "ci_assert": "ci_assert"}
INJECTION_MARKERS = ("injection", "indirect_prompt_injection")
# Keys in AGL export rows that may hold raw tool output; never copied into a task (C12).
_RAW_KEYS = ("tool_calls", "tool_results", "tool_outputs", "events", "transcript", "messages", "spans")
_MAX_TEXT = 4000


class HarvestError(ValueError):
    """Invalid harvest input (bad schema, unreviewed task, wrong split, ...)."""


class UnknownJudgeOp(HarvestError):
    """A rule judge uses an op SkillOpt 0.2.0 does not implement (upstream would pass it)."""


class SplitViolation(HarvestError):
    """An AGL export row from a non-evolve split reached SkillOpt harvest (C15)."""


@dataclass
class HarvestResult:
    tasks: list[TaskRecord]
    sources: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    capped: int = 0
    excluded_injection: int = 0

    def summary(self) -> dict[str, Any]:
        splits: dict[str, int] = {}
        for t in self.tasks:
            splits[t.split] = splits.get(t.split, 0) + 1
        return {"n_tasks": len(self.tasks), "splits": splits, "sources": dict(self.sources),
                "duplicates": self.duplicates, "capped": self.capped,
                "excluded_injection": self.excluded_injection,
                "task_ids": [t.id for t in self.tasks]}


# ------------------------------------------------------------------ validation

def validate_judge(judge: Mapping[str, Any] | None, *, task_id: str = "?",
                   known_tools: frozenset[str] | None = ORDER_SUPPORT_TOOLS) -> None:
    """Raise :class:`UnknownJudgeOp` / :class:`HarvestError` for anything SkillOpt would misjudge."""
    if not judge:
        return
    if not isinstance(judge, Mapping):
        raise HarvestError(f"task {task_id}: judge must be an object")
    kind = judge.get("kind", "rule")
    if kind != "rule":
        raise HarvestError(f"task {task_id}: judge kind {kind!r} unsupported (only 'rule')")
    checks = judge.get("checks", [])
    if not isinstance(checks, list) or not checks:
        raise HarvestError(f"task {task_id}: rule judge needs a non-empty checks list")
    for check in checks:
        if not isinstance(check, Mapping):
            raise HarvestError(f"task {task_id}: each check must be an object")
        op, arg = check.get("op"), check.get("arg")
        if op not in KNOWN_OPS:
            raise UnknownJudgeOp(f"task {task_id}: unknown rule-judge op {op!r} "
                                 f"(allowed: {sorted(KNOWN_OPS)})")
        if arg is None or (isinstance(arg, str) and not arg.strip()):
            raise HarvestError(f"task {task_id}: op {op} needs an arg")
        if op == "regex":
            try:
                re.compile(str(arg))
            except re.error as exc:
                raise HarvestError(f"task {task_id}: bad regex {arg!r}: {exc}") from exc
        elif op in ("max_chars", "min_chars"):
            if isinstance(arg, bool) or not isinstance(arg, int) or arg < 0:
                raise HarvestError(f"task {task_id}: {op} needs a non-negative int")
        elif op == "tool_called" and known_tools is not None and str(arg) not in known_tools:
            raise HarvestError(f"task {task_id}: tool_called names unknown tool {arg!r}")


def _is_injection(task: TaskRecord, suite: str = "") -> bool:
    labels = [suite, *task.tags]
    return any(m in str(label).lower() for label in labels for m in INJECTION_MARKERS)


def _clean_text(value: Any) -> str:
    return str(redact_secrets(str(value or "")))[:_MAX_TEXT]


def _task_from_dict(raw: Mapping[str, Any], *, task_id: str, strip_context: bool) -> TaskRecord:
    task = TaskRecord.from_dict({k: v for k, v in raw.items() if k not in _RAW_KEYS})
    task.id = task_id
    task.intent = _clean_text(task.intent)
    if not task.intent.strip():
        raise HarvestError(f"task {task_id}: empty intent")
    task.system = _clean_text(task.system)
    task.reference = _clean_text(task.reference)
    task.context_excerpt = "" if strip_context else _clean_text(task.context_excerpt)
    task.attempted_solution = ""  # prior agent output may quote tool results (C12)
    task.tags = [str(t) for t in (task.tags or [])]
    task.origin = "real"
    task.derived_from = ""
    task.source_sessions = [str(s) for s in (task.source_sessions or [])][:20]
    kind = task.reference_kind or "none"
    if kind not in INTERNAL_KIND:
        raise HarvestError(f"task {task_id}: unsupported reference_kind {kind!r}")
    if kind in ("rule", "ci_rule"):
        validate_judge(task.judge, task_id=task_id)
        if not task.judge:
            raise HarvestError(f"task {task_id}: rule task without judge checks")
    elif task.judge:
        raise HarvestError(f"task {task_id}: judge given for non-rule reference_kind {kind!r}")
    task.judge = json.loads(json.dumps(dict(task.judge or {})))
    task.reference_kind = INTERNAL_KIND[kind]
    return task


# ------------------------------------------------------------------ sources

def load_reviewed_tasks(path: Path) -> list[TaskRecord]:
    """Parse the JSONL tasks file. Line 1 is the header; blank and ``//`` lines are ignored."""
    if not path.exists():
        raise HarvestError(f"tasks file not found: {path}")
    lines = [(n, ln) for n, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
             if ln.strip() and not ln.lstrip().startswith("//")]
    if not lines:
        raise HarvestError(f"{path}: empty tasks file")
    try:
        rows = [(n, json.loads(ln)) for n, ln in lines]
    except json.JSONDecodeError as exc:
        raise HarvestError(f"{path}: invalid JSON line: {exc}") from exc
    _, header = rows[0]
    if not isinstance(header, dict) or header.get("format") != TASKS_FORMAT:
        raise HarvestError(f"{path}: first line must be a header with format {TASKS_FORMAT!r}")
    if header.get("reviewed") is not True:
        raise HarvestError(f"{path}: header is not reviewed (requires \"reviewed\": true)")
    out: list[TaskRecord] = []
    seen: set[str] = set()
    for n, row in rows[1:]:
        if not isinstance(row, dict):
            raise HarvestError(f"{path}:{n}: task must be an object")
        tid = str(row.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}", tid):
            raise HarvestError(f"{path}:{n}: bad task id {tid!r}")
        if row.get("reviewed") is not True:
            raise HarvestError(f"{path}:{n}: task {tid} is not reviewed")
        if tid in seen:
            raise HarvestError(f"{path}:{n}: duplicate task id {tid}")
        seen.add(tid)
        out.append(_task_from_dict(row, task_id=tid, strip_context=False))
    return out


def _dataset_split(row: Mapping[str, Any]) -> str | None:
    for key in ("dataset_split", "eval_split", "domain_split"):
        if row.get(key) is not None:
            return str(row[key])
    split = row.get("split")
    if split is not None and str(split) in DATASET_SPLITS:
        return str(split)
    return None


def from_agl_exports(rows: Iterable[Mapping[str, Any]]) -> tuple[list[TaskRecord], int]:
    """TaskRecords from AGL export rows: ``{dataset_split, suite?, task: {...TaskRecord}}``
    or a flat TaskRecord dict carrying ``dataset_split``. Returns ``(tasks, n_injection_excluded)``."""
    out: list[TaskRecord] = []
    excluded = 0
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise HarvestError(f"agl export row {i}: not an object")
        split = _dataset_split(row)
        if split is None:
            raise SplitViolation(f"agl export row {i}: missing dataset split; refusing (C15)")
        if split != ALLOWED_DATASET_SPLIT:
            raise SplitViolation(f"agl export row {i}: dataset split {split!r} may not feed "
                                 f"SkillOpt (only {ALLOWED_DATASET_SPLIT!r}; C15)")
        body = row.get("task") if isinstance(row.get("task"), Mapping) else row
        body = {k: v for k, v in body.items() if k not in ("split", "dataset_split", "eval_split", "domain_split")}
        base = str(body.get("id") or row.get("case_id") or f"row{i}")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:8]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "-", base)[:48]
        task = _task_from_dict(body, task_id=f"agl-{safe}-{digest}", strip_context=True)
        if _is_injection(task, str(row.get("suite", ""))):
            excluded += 1
            continue
        if row.get("suite"):
            task.tags = [f"suite:{row['suite']}", *[t for t in task.tags if not t.startswith("suite:")]]
        out.append(task)
    return out, excluded


# ------------------------------------------------------------------ dedupe / split

def content_key(task: TaskRecord) -> str:
    norm = re.sub(r"\s+", " ", task.intent.strip().lower())
    raw = json.dumps([norm, task.context_excerpt.strip(), task.reference_kind, task.judge], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _hash_unit(task_id: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}|{task_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def assign_stable_splits(tasks: Sequence[TaskRecord], *, val_fraction: float = 0.34,
                         salt: str = "ci-sleep-v1") -> list[TaskRecord]:
    """Stable train/val by id hash; guarantees >=1 of each when there are >=2 tasks."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    units = {t.id: _hash_unit(t.id, salt) for t in tasks}
    for t in tasks:
        t.split = "val" if units[t.id] < val_fraction else "train"
    if len(tasks) >= 2:
        val = [t for t in tasks if t.split == "val"]
        train = [t for t in tasks if t.split == "train"]
        if not val:
            min(train, key=lambda t: units[t.id]).split = "val"
        elif not train:
            max(val, key=lambda t: units[t.id]).split = "train"
    return list(tasks)


def harvest(tasks_file: Path | None, agl_rows: Iterable[Mapping[str, Any]] = (), *,
            max_tasks: int = 40, val_fraction: float = 0.34) -> HarvestResult:
    reviewed = load_reviewed_tasks(tasks_file) if tasks_file is not None else []
    exported, excluded = from_agl_exports(agl_rows)
    reviewed_kept = [t for t in reviewed if not _is_injection(t)]
    excluded += len(reviewed) - len(reviewed_kept)
    seen: set[str] = set()
    ids: set[str] = set()
    tasks: list[TaskRecord] = []
    dupes = 0
    for t in [*reviewed_kept, *exported]:  # reviewed tasks win ties
        key = content_key(t)
        if key in seen or t.id in ids:
            dupes += 1
            continue
        seen.add(key)
        ids.add(t.id)
        tasks.append(t)
    capped = 0
    if len(tasks) > max_tasks:
        # deterministic cap: reviewed first, then by stable hash
        capped = len(tasks) - max_tasks
        n_rev = sum(1 for t in tasks if not t.id.startswith("agl-"))
        rest = sorted(tasks[n_rev:], key=lambda t: _hash_unit(t.id, "cap"))
        tasks = (tasks[:n_rev] + rest)[:max_tasks]
    assign_stable_splits(tasks, val_fraction=val_fraction)
    return HarvestResult(tasks=tasks, sources={"reviewed": len(reviewed_kept), "agl": len(exported)},
                         duplicates=dupes, capped=capped, excluded_injection=excluded)


def read_jsonl_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load AGL export JSONL files (``ci-lab sleep run --agl-export``)."""
    rows: list[dict[str, Any]] = []
    for p in paths:
        for n, line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise HarvestError(f"{p}:{n}: invalid JSON: {exc}") from exc
    return rows
