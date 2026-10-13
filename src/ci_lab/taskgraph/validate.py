"""Task-graph and rubric validation (bus contract v2 §5). Pure: returns problems, never raises.

``validate_rubric`` enforces measurable, non-vague, oracle-backed rubrics; ``validate_graph``
enforces DAG structure, a single output per deliverable, short instructions, and rubric secrecy
(no 8-gram shared with any criterion description/question, no canary/criterion id/suite name)
against the sealed rubric when a vault is given.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ci_lab.taskgraph.model import (
    MEASURES,
    SOFT_MEASURES,
    Criterion,
    Deliverable,
    Rubric,
    SpecError,
    TaskGraph,
)
from ci_lab.taskgraph.vault import SEALED_DIRNAME, RubricVault, VaultError, VaultMissing
from ci_lab.tools.critic_checks import LeakCorpus
from ci_lab.tools.paths import PathRejected, normalize_rel

__all__ = [
    "COMMAND_ALLOWLIST",
    "DETERMINISTIC_KINDS",
    "MAX_INSTRUCTIONS",
    "MAX_QUESTION",
    "VAGUE_WORDS",
    "Problem",
    "rubric_leaks",
    "rubric_material",
    "validate_graph",
    "validate_rubric",
]

TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RUBRIC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
COMMITMENT_RE = re.compile(r"^[0-9a-f]{64}$")
CANARY_RE = re.compile(r"^[0-9a-f]{16}$")
PYTHON_TARGET_RE = re.compile(r"^ci_lab(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
MAX_INSTRUCTIONS = 1200
MAX_QUESTION = 300
LEAK_N = 8
COMMAND_ALLOWLIST = frozenset({"python", "pytest", "ruff", "node", "git"})
DETERMINISTIC_KINDS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {  # kind -> (required, optional)
    "regex": (frozenset({"pattern"}), frozenset({"negate"})),
    "json_schema": (frozenset({"schema"}), frozenset()),
    "file_exists": (frozenset({"path"}), frozenset()),
    "command": (frozenset({"argv"}), frozenset({"timeout_s"})),
    "python": (frozenset({"callable"}), frozenset()),
}
VAGUE_WORDS = ("good", "nice", "appropriate", "high quality", "clean", "robust")
_VAGUE = re.compile(r"\b(?:" + "|".join(w.replace(" ", r"[\s-]+") for w in VAGUE_WORDS) + r")\b", re.IGNORECASE)
_PATH_MENTION = re.compile(r"(?<![\w./-])((?:[\w-]+/)*[\w-]+\.[A-Za-z][A-Za-z0-9]{1,7})(?![\w/-])")


@dataclass(frozen=True)
class Problem:
    code: str
    where: str
    message: str


def _keys(check: Mapping[str, Any], required: frozenset[str], optional: frozenset[str]) -> list[str]:
    errs = [f"missing {k!r}" for k in sorted(required - check.keys())]
    errs += [f"unknown key {k!r}" for k in sorted(check.keys() - required - optional - {"independent"})]
    if not isinstance(check.get("independent", False), bool):
        errs.append("'independent' must be a boolean")
    return errs


def _unit(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and 0 < v <= 1


def _schema_error(schema: Any) -> str | None:
    if not isinstance(schema, Mapping):
        return "schema must be an object"
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return f"invalid JSON schema: {exc.message}"
    return None


def _deterministic(check: Mapping[str, Any]) -> list[str]:
    kind = check.get("kind")
    if kind not in DETERMINISTIC_KINDS:
        return [f"kind must be one of {sorted(DETERMINISTIC_KINDS)}, got {kind!r}"]
    required, optional = DETERMINISTIC_KINDS[kind]
    errs = _keys(check, required, optional | {"kind"})
    if errs:
        return errs
    if kind == "regex":
        try:
            re.compile(check["pattern"])
        except (re.error, TypeError) as exc:
            errs.append(f"bad pattern: {exc}")
        if not isinstance(check.get("negate", False), bool):
            errs.append("'negate' must be a boolean")
    elif kind == "json_schema":
        errs += [e for e in [_schema_error(check["schema"])] if e]
    elif kind == "file_exists":
        try:
            normalize_rel(check["path"])
        except PathRejected as exc:
            errs.append(f"bad path: {exc}")
    elif kind == "command":
        argv = check["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
            errs.append("argv must be a non-empty list of strings (no shell strings)")
        elif argv[0] not in COMMAND_ALLOWLIST:
            errs.append(f"executable {argv[0]!r} not in allowlist {sorted(COMMAND_ALLOWLIST)}")
        if "timeout_s" in check and not (isinstance(check["timeout_s"], (int, float))
                                         and not isinstance(check["timeout_s"], bool) and check["timeout_s"] > 0):
            errs.append("timeout_s must be a positive number")
    elif not isinstance(check["callable"], str) or not PYTHON_TARGET_RE.match(check["callable"]):
        errs.append(f"callable must be a dotted path under ci_lab., got {check['callable']!r}")
    return errs


def _assert(check: Mapping[str, Any]) -> list[str]:
    errs = _keys(check, frozenset({"suite", "split", "min_score"}), frozenset())
    if errs:
        return errs
    errs += [f"{k} must be a non-empty string" for k in ("suite", "split")
             if not (isinstance(check[k], str) and check[k].strip())]
    if not _unit(check["min_score"]):
        errs.append("min_score must be in (0, 1]")
    return errs


def _soft(check: Mapping[str, Any]) -> list[str]:
    errs = _keys(check, frozenset({"question", "type"}), frozenset({"options"}))
    if errs:
        return errs
    q, qtype, options = check["question"], check["type"], check.get("options", [])
    if not isinstance(q, str) or not q.strip() or len(q) > MAX_QUESTION:
        errs.append(f"question must be a non-empty string of at most {MAX_QUESTION} chars")
    if not isinstance(options, list) or not all(isinstance(o, str) and o.strip() for o in options):
        errs.append("options must be a list of non-empty strings")
    elif qtype == "choice":
        if len(set(options)) < 2 or len(set(options)) != len(options):
            errs.append("choice questions need at least two distinct options")
    elif qtype == "noul":
        if options:
            errs.append("noul questions take no options")
    else:
        errs.append(f"type must be 'noul' or 'choice', got {qtype!r}")
    return errs


def _criterion(c: Criterion, where: str) -> list[Problem]:
    out: list[Problem] = []

    def add(code: str, message: str) -> None:
        out.append(Problem(code, where, message))

    if not isinstance(c.id, str) or not RUBRIC_ID_RE.match(c.id):
        add("criterion.id", f"bad criterion id {c.id!r}")
    if not isinstance(c.description, str) or not c.description.strip():
        add("criterion.description", "description is empty")
    if c.measure not in MEASURES:
        add("criterion.measure", f"measure must be one of {MEASURES}, got {c.measure!r}")
        return out
    if not (math.isfinite(c.threshold) and 0 < c.threshold <= 1):
        add("criterion.threshold", f"threshold must be in (0, 1], got {c.threshold}")
    if not (math.isfinite(c.weight) and c.weight > 0):
        add("criterion.weight", f"weight must be > 0, got {c.weight}")
    check = c.to_json()["check"]
    validator = {"deterministic": _deterministic, "assert": _assert}.get(c.measure, _soft)
    errs = validator(check)
    out += [Problem(f"check.{c.measure}", where, e) for e in errs]
    if errs and c.required:
        add("criterion.unmeasurable", "required criterion has no valid check")
    if c.measure in SOFT_MEASURES:
        text = f"{c.description} {check.get('question', '')}"
        if m := _VAGUE.search(text):
            add("criterion.vague", f"soft criterion uses vague wording {m.group(0)!r}; ask a concrete question")
    return out


def validate_rubric(rubric: Rubric) -> list[Problem]:
    where = f"rubric:{rubric.version_id}"
    out: list[Problem] = []

    def add(code: str, message: str) -> None:
        out.append(Problem(code, where, message))

    if not isinstance(rubric.id, str) or not RUBRIC_ID_RE.match(rubric.id):
        add("rubric.id", f"bad rubric id {rubric.id!r}")
    if isinstance(rubric.version, bool) or not isinstance(rubric.version, int) or rubric.version < 1:
        add("rubric.version", f"version must be an integer >= 1, got {rubric.version!r}")
    if not isinstance(rubric.deliverable, str) or not TASK_ID_RE.match(rubric.deliverable):
        add("rubric.deliverable", f"bad deliverable id {rubric.deliverable!r}")
    if not (math.isfinite(rubric.pass_score) and 0 < rubric.pass_score <= 1):
        add("rubric.pass_score", f"pass_score must be in (0, 1], got {rubric.pass_score}")
    if not isinstance(rubric.canary, str) or not CANARY_RE.match(rubric.canary):
        add("rubric.canary", "canary must be 16 lowercase hex chars")
    if not rubric.criteria:
        add("rubric.criteria", "rubric has no criteria")
    elif not rubric.oracles():
        add("rubric.oracle", "rubric needs at least one deterministic or assert criterion")
    seen: set[str] = set()
    for c in rubric.criteria:
        if c.id in seen:
            add("criterion.duplicate", f"duplicate criterion id {c.id!r}")
        seen.add(c.id)
        out += _criterion(c, f"{where}/criterion:{c.id}")
    return out


def rubric_material(rubric: Rubric) -> dict[str, list[str]]:
    """Secret rubric material by category: ``text`` (descriptions/questions), ``canary``, ``criterion_id``, ``suite``."""
    return {
        "text": [c.description for c in rubric.criteria]
        + [q for c in rubric.criteria if isinstance(q := c.check.get("question"), str)],
        "canary": [rubric.canary],
        "criterion_id": [c.id for c in rubric.criteria],
        "suite": [s for c in rubric.criteria if isinstance(s := c.check.get("suite"), str)],
    }


def rubric_leaks(instructions: str, rubric: Rubric) -> list[str]:
    """Rubric material visible in ``instructions``: shared 8-grams, canary, criterion ids, suite names."""
    m = rubric_material(rubric)
    hits = LeakCorpus.build(m["text"], m["canary"] + m["criterion_id"] + m["suite"], n=LEAK_N).screen(instructions)
    if rubric.canary and rubric.canary in instructions and not any(rubric.canary in h for h in hits):
        hits.append("rubric canary")
    return hits


def validate_graph(graph: TaskGraph, *, vault: RubricVault | None = None) -> list[Problem]:
    out: list[Problem] = []

    def add(code: str, where: str, message: str) -> None:
        out.append(Problem(code, where, message))

    if not isinstance(graph.id, str) or not graph.id.strip():
        add("graph.id", "graph", "graph id is empty")
    if not graph.deliverables:
        add("graph.empty", f"graph:{graph.id}", "graph has no deliverables")
    ids = [d.id for d in graph.deliverables]
    by_id = {d.id: d for d in graph.deliverables}
    for i, d in enumerate(graph.deliverables):
        w = f"deliverable:{d.id}"
        if not isinstance(d.id, str) or not TASK_ID_RE.match(d.id):
            add("id.format", w, f"bad deliverable id {d.id!r}")
        elif d.id == SEALED_DIRNAME:
            add("id.reserved", w, f"{SEALED_DIRNAME!r} is reserved for the rubric vault")
        if ids.index(d.id) != i:
            add("id.duplicate", w, f"deliverable id {d.id!r} is used {ids.count(d.id)} times")
        if not d.title.strip():
            add("title.empty", w, "title is empty")
        for dep in sorted(set(d.depends_on)):
            if dep == d.id:
                add("dep.self", w, "deliverable depends on itself")
            elif dep not in by_id:
                add("dep.unknown", w, f"unknown dependency {dep!r}")
        if len(set(d.depends_on)) != len(d.depends_on):
            add("dep.duplicate", w, "duplicate dependency")
        out += _deliverable_body(d, w, by_id)
        b = d.budget
        if b.max_attempts < 1 or not b.timeout_s > 0 or (b.max_tokens is not None and b.max_tokens < 1) \
                or any(not v > 0 for v in b.weight.values()):
            add("budget", w, "max_attempts >= 1, timeout_s > 0, max_tokens >= 1 and weights > 0 required")
        if not COMMITMENT_RE.match(d.rubric_commitment):
            add("commitment.format", w, "rubric_commitment must be 64 lowercase hex chars")
        elif vault is not None:
            out += _sealed(d.id, d.instructions, d.rubric_commitment, w, vault)
    if not any(p.code.startswith(("id.", "dep.")) for p in out):
        try:
            graph.topo_order()
        except SpecError as exc:
            add("graph.cycle", f"graph:{graph.id}", str(exc))
    return out


def _deliverable_body(d: Deliverable, w: str, by_id: Mapping[str, Deliverable]) -> list[Problem]:
    out: list[Problem] = []
    text, o = d.instructions, d.output
    if not text.strip():
        out.append(Problem("instructions.empty", w, "instructions are empty"))
    if len(text) > MAX_INSTRUCTIONS:
        out.append(Problem("instructions.length", w, f"{len(text)} chars > {MAX_INSTRUCTIONS}"))
    if o.kind in ("file", "patch") and not o.path:
        out.append(Problem("output.path", w, f"{o.kind} output needs a path"))
    if o.path:
        try:
            normalize_rel(o.path)
        except PathRejected as exc:
            out.append(Problem("output.path", w, str(exc)))
    if o.schema is not None:
        err = "schema is only allowed for json output" if o.kind != "json" else _schema_error(o.to_json()["schema"])
        if err:
            out.append(Problem("output.schema", w, err))
    inputs = {c.ref for c in d.context if c.kind == "file"}
    inputs |= {by_id[t].output.path for t in d.depends_on if t in by_id and by_id[t].output.path}
    mentioned = {m for m in _PATH_MENTION.findall(text) if m not in inputs}
    if len(mentioned) > 1:
        out.append(Problem("output.ambiguous", w, f"instructions mention {len(mentioned)} output paths: "
                                                  f"{sorted(mentioned)}; a deliverable has exactly one output"))
    for c in d.context:
        if c.kind == "deliverable" and c.ref not in d.depends_on:
            out.append(Problem("context.ref", w, f"context deliverable {c.ref!r} is not a dependency"))
    return out


def _sealed(task_id: str, instructions: str, commitment: str, w: str, vault: RubricVault) -> list[Problem]:
    try:
        rubric = vault.open(commitment, role="examiner")
    except VaultMissing as exc:
        return [Problem("commitment.missing", w, str(exc))]
    except VaultError as exc:
        return [Problem("commitment.tampered", w, str(exc))]
    out = validate_rubric(rubric)
    if rubric.deliverable != task_id:
        out.append(Problem("rubric.deliverable", w, f"sealed rubric is for {rubric.deliverable!r}"))
    out += [Problem("rubric.leak", w, f"instructions leak rubric material: {h}")
            for h in rubric_leaks(instructions, rubric)]
    return out
