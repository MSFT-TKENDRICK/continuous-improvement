"""Task graph + hidden rubric models (bus contract v2 §5).

Frozen dataclasses with strict ``to_json`` / ``from_json`` (unknown or missing fields raise
``ValueError``). A :class:`Deliverable` carries only a ``rubric_commitment`` (sha256 of the
sealed rubric's canonical JSON); rubric content lives in the vault and never reaches students,
whose only view of a deliverable is :class:`StudentSpec`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

import yaml

__all__ = [
    "Budget",
    "ContextRef",
    "Criterion",
    "CriterionRole",
    "Deliverable",
    "Measure",
    "OutputSpec",
    "Rubric",
    "SpecError",
    "StudentSpec",
    "TaskGraph",
    "canonical_json",
    "dump_graph",
    "load_graph",
]

Measure = Literal["deterministic", "assert", "s1", "llm", "metric"]
CriterionRole = Literal["quality", "resource"]
OutputKind = Literal["file", "text", "json", "patch"]
ContextKind = Literal["file", "deliverable", "text"]
MEASURES: tuple[str, ...] = get_args(Measure)
CRITERION_ROLES: tuple[str, ...] = get_args(CriterionRole)
# "metric" is deterministic (evaluator-measured runtime/surface numbers scored by ci_lab.metrics.rubric).
ORACLE_MEASURES = ("deterministic", "assert", "metric")
SOFT_MEASURES = ("s1", "llm")
# Measures whose criteria are always resource criteria (cost/simplicity, never deliverable quality).
RESOURCE_MEASURES = ("metric",)


class SpecError(ValueError):
    """Malformed task graph / rubric: unknown, missing or mistyped field, or bad dependency structure."""


def canonical_json(obj: Any) -> str:
    """Sorted keys, ``(",", ":")`` separators, ``ensure_ascii=False`` (contract §3)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _freeze(v: Any) -> Any:
    if isinstance(v, Mapping):
        return MappingProxyType({str(k): _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def _thaw(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_thaw(x) for x in v]
    return v


def _keys(cls: type, data: Any) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise SpecError(f"{cls.__name__}: expected an object, got {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
    if unknown := sorted(set(data) - names):
        raise SpecError(f"{cls.__name__}: unknown field(s) {unknown}")
    if missing := sorted(required - set(data)):
        raise SpecError(f"{cls.__name__}: missing field(s) {missing}")
    return data


def _str(d: Mapping[str, Any], k: str, *, choices: tuple[str, ...] = (), default: str | None = None) -> str:
    v = d[k] if default is None else d.get(k, default)
    if not isinstance(v, str):
        raise SpecError(f"{k}: expected a string")
    if choices and v not in choices:
        raise SpecError(f"{k}: {v!r} not in {choices}")
    return v


def _num(v: Any, k: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SpecError(f"{k}: expected a number")
    return float(v)


def _int(v: Any, k: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise SpecError(f"{k}: expected an integer")
    return v


def _strs(v: Any, k: str) -> tuple[str, ...]:
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
        raise SpecError(f"{k}: expected a list of strings")
    return tuple(v)


def _list(v: Any, k: str) -> list[Any]:
    if not isinstance(v, (list, tuple)):
        raise SpecError(f"{k}: expected a list")
    return list(v)


def _obj(v: Any, k: str) -> Mapping[str, Any]:
    if not isinstance(v, Mapping):
        raise SpecError(f"{k}: expected an object")
    return v


@dataclass(frozen=True)
class OutputSpec:
    kind: OutputKind
    path: str | None = None
    schema: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", None if self.schema is None else _freeze(self.schema))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "schema": _thaw(self.schema)}

    @classmethod
    def from_json(cls, data: Any) -> OutputSpec:
        d = _keys(cls, data)
        path = d.get("path")
        if path is not None and not isinstance(path, str):
            raise SpecError("path: expected a string or null")
        schema = d.get("schema")
        return cls(_str(d, "kind", choices=get_args(OutputKind)), path,  # type: ignore[arg-type]
                   None if schema is None else _obj(schema, "schema"))


@dataclass(frozen=True)
class ContextRef:
    kind: ContextKind
    ref: str

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref}

    @classmethod
    def from_json(cls, data: Any) -> ContextRef:
        d = _keys(cls, data)
        return cls(_str(d, "kind", choices=get_args(ContextKind)), _str(d, "ref"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Budget:
    max_attempts: int = 3
    timeout_s: float = 600.0
    max_tokens: int | None = None
    weight: Mapping[str, float] = field(default_factory=lambda: {"llm": 1.0})

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeout_s", float(self.timeout_s))
        object.__setattr__(self, "weight", _freeze({k: float(v) for k, v in self.weight.items()}))

    def to_json(self) -> dict[str, Any]:
        return {"max_attempts": self.max_attempts, "timeout_s": self.timeout_s,
                "max_tokens": self.max_tokens, "weight": _thaw(self.weight)}

    @classmethod
    def from_json(cls, data: Any) -> Budget:
        d = _keys(cls, data)
        tokens = d.get("max_tokens")
        weight = _obj(d.get("weight", {"llm": 1.0}), "weight")
        return cls(_int(d.get("max_attempts", 3), "max_attempts"), _num(d.get("timeout_s", 600.0), "timeout_s"),
                   None if tokens is None else _int(tokens, "max_tokens"),
                   {str(k): _num(v, f"weight.{k}") for k, v in weight.items()})


@dataclass(frozen=True)
class Deliverable:
    id: str
    title: str
    instructions: str
    output: OutputSpec
    context: tuple[ContextRef, ...] = ()
    depends_on: tuple[str, ...] = ()
    rubric_commitment: str = ""
    budget: Budget = field(default_factory=Budget)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", tuple(self.context))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "instructions": self.instructions,
                "output": self.output.to_json(), "context": [c.to_json() for c in self.context],
                "depends_on": list(self.depends_on), "rubric_commitment": self.rubric_commitment,
                "budget": self.budget.to_json()}

    @classmethod
    def from_json(cls, data: Any) -> Deliverable:
        d = _keys(cls, data)
        return cls(_str(d, "id"), _str(d, "title"), _str(d, "instructions"), OutputSpec.from_json(d["output"]),
                   tuple(ContextRef.from_json(c) for c in _list(d.get("context", []), "context")),
                   _strs(d.get("depends_on", []), "depends_on"),
                   _str(d, "rubric_commitment", default=""),
                   Budget.from_json(d.get("budget", {})))


@dataclass(frozen=True)
class Criterion:
    id: str
    description: str
    measure: Measure
    check: Mapping[str, Any]
    threshold: float
    weight: float = 1.0
    required: bool = False
    # "quality" counts toward the deliverable score; "resource" (cost/simplicity) is scored separately into
    # subscores. "" resolves to "resource" for metric criteria and "quality" otherwise.
    role: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "check", _freeze(self.check))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "weight", float(self.weight))
        role = self.role or ("resource" if self.measure in RESOURCE_MEASURES else "quality")
        if role not in CRITERION_ROLES:
            raise SpecError(f"role: {role!r} not in {CRITERION_ROLES}")
        if self.measure in RESOURCE_MEASURES and role != "resource":
            raise SpecError(f"role: {self.measure} criteria are always resource criteria, got {role!r}")
        object.__setattr__(self, "role", role)

    @property
    def oracle(self) -> bool:
        return self.measure in ORACLE_MEASURES

    @property
    def resource(self) -> bool:
        return self.role == "resource"

    def to_json(self) -> dict[str, Any]:
        out = {"id": self.id, "description": self.description, "measure": self.measure,
               "check": _thaw(self.check), "threshold": self.threshold, "weight": self.weight,
               "required": self.required}
        if self.resource:  # quality is the default; omitting it keeps pre-role rubric commitments stable
            out["role"] = self.role
        return out

    @classmethod
    def from_json(cls, data: Any) -> Criterion:
        d = _keys(cls, data)
        required = d.get("required", False)
        if not isinstance(required, bool):
            raise SpecError("required: expected a boolean")
        role = _str(d, "role", choices=CRITERION_ROLES) if "role" in d else ""
        return cls(_str(d, "id"), _str(d, "description"), _str(d, "measure", choices=MEASURES),  # type: ignore[arg-type]
                   _obj(d["check"], "check"), _num(d["threshold"], "threshold"),
                   _num(d.get("weight", 1.0), "weight"), required, role)


@dataclass(frozen=True)
class Rubric:
    id: str
    version: int
    deliverable: str
    criteria: tuple[Criterion, ...]
    pass_score: float
    canary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "pass_score", float(self.pass_score))

    def oracles(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if c.measure in ORACLE_MEASURES)

    def soft(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if c.measure in SOFT_MEASURES)

    def quality(self) -> tuple[Criterion, ...]:
        """Criteria that make up the deliverable-quality score."""
        return tuple(c for c in self.criteria if not c.resource)

    def resources(self) -> tuple[Criterion, ...]:
        """Resource (cost/simplicity) criteria, reported as subscores and never folded into quality."""
        return tuple(c for c in self.criteria if c.resource)

    def quality_score(self, scores: Mapping[str, float | None]) -> float:
        """Weighted mean of the scored quality criteria in ``scores`` (criterion id -> score); 0.0 if none."""
        scored = [(c.weight, float(s)) for c in self.quality() if (s := scores.get(c.id)) is not None]
        weight = sum(w for w, _ in scored)
        return sum(w * s for w, s in scored) / weight if weight else 0.0

    def resource_subscores(self, scores: Mapping[str, float | None]) -> dict[str, float]:
        """``resource.<id>`` per resource criterion (unscored counts as 0.0, fail closed) plus their weighted
        mean ``resource_score``; empty when the rubric has no resource criteria."""
        res = self.resources()
        if not res:
            return {}
        out = {f"resource.{c.id}": float(scores.get(c.id) or 0.0) for c in res}
        weight = sum(c.weight for c in res)
        out["resource_score"] = (sum(c.weight * out[f"resource.{c.id}"] for c in res) / weight) if weight else 0.0
        return out

    @property
    def version_id(self) -> str:
        return f"{self.id}@v{self.version}"

    def commitment(self) -> str:
        return hashlib.sha256(canonical_json(self.to_json()).encode("utf-8")).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "deliverable": self.deliverable,
                "criteria": [c.to_json() for c in self.criteria], "pass_score": self.pass_score,
                "canary": self.canary}

    @classmethod
    def from_json(cls, data: Any) -> Rubric:
        d = _keys(cls, data)
        return cls(_str(d, "id"), _int(d["version"], "version"), _str(d, "deliverable"),
                   tuple(Criterion.from_json(c) for c in _list(d["criteria"], "criteria")),
                   _num(d["pass_score"], "pass_score"), _str(d, "canary"))


@dataclass(frozen=True)
class TaskGraph:
    id: str
    goal: str
    deliverables: tuple[Deliverable, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "deliverables", tuple(self.deliverables))

    def deliverable(self, task_id: str) -> Deliverable:
        for d in self.deliverables:
            if d.id == task_id:
                return d
        raise KeyError(task_id)

    def dependents(self, task_id: str) -> tuple[str, ...]:
        """Direct dependents of ``task_id`` in declaration order."""
        return tuple(d.id for d in self.deliverables if task_id in d.depends_on)

    def topo_order(self) -> tuple[str, ...]:
        """Kahn order, ties broken by declaration order. ``ValueError`` on unknown deps or cycles."""
        ids = [d.id for d in self.deliverables]
        if len(set(ids)) != len(ids):
            raise SpecError("duplicate deliverable ids")
        pending = {d.id: set(d.depends_on) for d in self.deliverables}
        for tid, deps in pending.items():
            if unknown := sorted(deps - pending.keys()):
                raise SpecError(f"{tid}: unknown dependencies {unknown}")
        order: list[str] = []
        while pending:
            ready = [t for t in ids if t in pending and not pending[t]]
            if not ready:
                raise SpecError(f"dependency cycle among {sorted(pending)}")
            for t in ready:
                del pending[t]
                order.append(t)
            for deps in pending.values():
                deps.difference_update(ready)
        return tuple(order)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "goal": self.goal, "deliverables": [d.to_json() for d in self.deliverables]}

    @classmethod
    def from_json(cls, data: Any) -> TaskGraph:
        d = _keys(cls, data)
        return cls(_str(d, "id"), _str(d, "goal"),
                   tuple(Deliverable.from_json(x) for x in _list(d["deliverables"], "deliverables")))


@dataclass(frozen=True)
class StudentSpec:
    """The only view of a deliverable that reaches a student: no rubric content, just its commitment."""

    id: str
    title: str
    instructions: str
    output: OutputSpec
    context: tuple[ContextRef, ...]
    depends_on: tuple[str, ...]
    budget: Budget
    rubric_commitment: str

    @classmethod
    def of(cls, deliverable: Deliverable) -> StudentSpec:
        d = deliverable
        return cls(d.id, d.title, d.instructions, d.output, d.context, d.depends_on, d.budget,
                   d.rubric_commitment)

    def render(self) -> str:
        o = self.output
        lines = [f"# Task {self.id}: {self.title}", "", "## Instructions", self.instructions.strip(), "",
                 "## Output", f"- kind: {o.kind}"]
        if o.path is not None:
            lines.append(f"- path: {o.path}")
        if o.schema is not None:
            lines.append(f"- schema: {canonical_json(_thaw(o.schema))}")
        if self.context:
            lines += ["", "## Context", *(f"- {c.kind}: {c.ref}" for c in self.context)]
        if self.depends_on:
            lines += ["", "## Depends on", *(f"- {t}" for t in self.depends_on)]
        b = self.budget
        lines += ["", "## Budget", f"- max_attempts: {b.max_attempts}", f"- timeout_s: {b.timeout_s:g}"]
        if b.max_tokens is not None:
            lines.append(f"- max_tokens: {b.max_tokens}")
        lines += ["", f"Rubric commitment: {self.rubric_commitment}"]
        return "\n".join(lines) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "instructions": self.instructions,
                "output": self.output.to_json(), "context": [c.to_json() for c in self.context],
                "depends_on": list(self.depends_on), "budget": self.budget.to_json(),
                "rubric_commitment": self.rubric_commitment}

    @classmethod
    def from_json(cls, data: Any) -> StudentSpec:
        d = _keys(cls, data)
        return cls(_str(d, "id"), _str(d, "title"), _str(d, "instructions"), OutputSpec.from_json(d["output"]),
                   tuple(ContextRef.from_json(c) for c in _list(d["context"], "context")),
                   _strs(d["depends_on"], "depends_on"), Budget.from_json(d["budget"]),
                   _str(d, "rubric_commitment"))


def load_graph(path: Path | str) -> TaskGraph:
    """Load a :class:`TaskGraph` from ``.json`` or ``.yaml``/``.yml``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    return TaskGraph.from_json(data)


def dump_graph(graph: TaskGraph, path: Path | str) -> Path:
    """Write ``graph`` as JSON or YAML (by suffix); returns the path."""
    p = Path(path)
    data = graph.to_json()
    if p.suffix.lower() in (".yaml", ".yml"):
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    p.write_text(text, encoding="utf-8", newline="\n")
    return p
