"""ACS manifest loading and validation (spec §2, §4, §9, §10, §12) and the §3 path language.

A manifest is YAML or JSON validated against the vendored ``manifest.schema.json`` plus the
normative rules the schema cannot express. File manifests resolve local ``extends`` (confined
to the top-level manifest's directory tree); remote URL ``extends`` are unsupported and fail
closed. Every failure raises :class:`AcsError` carrying a reserved reason.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from ci_lab.governance.acs.canonical import (
    DEFAULT_LIMITS,
    AcsError,
    JsonValue,
    Limits,
    canonical_bytes,
)

SPEC_VERSION = "0.4.0-alpha.1"
INTERVENTION_POINTS = (
    "agent_startup", "input", "pre_model_call", "post_model_call",
    "pre_tool_call", "post_tool_call", "output", "agent_shutdown",
)  # fmt: skip
TOOL_POINTS = frozenset({"pre_tool_call", "post_tool_call"})
REMOVED_FIELDS = ("system_prompt_file", "system_prompt_url", "bundle_url")
INVALID = "runtime_error:manifest_invalid"

_ROOT = re.compile(r"\$(snap|pi|target|tool)?")
_SEG = re.compile(r'\.([^.\[\]"]+)|\[(\d+)\]|\[("(?:[^"\\]|\\.)*")\]')
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


@dataclass(frozen=True, slots=True)
class AcsPath:
    """A parsed manifest path: ``root`` is ``snap`` (also for ``$``/``$.name``), ``pi``, ``target`` or ``tool``."""

    text: str
    root: str
    segments: tuple[str | int, ...]


def parse_path(text: str) -> AcsPath:
    """Parse ``$root(.name|[n]|["name"])*``; raises ``ValueError`` on any grammar violation."""
    m = _ROOT.match(text) if isinstance(text, str) else None
    if m is None:
        raise ValueError(f"path must start with a $ root: {text!r}")
    segments: list[str | int] = []
    pos = m.end()
    while pos < len(text):
        s = _SEG.match(text, pos)
        if s is None:
            raise ValueError(f"bad path segment at {pos}: {text!r}")
        name, index, quoted = s.groups()
        segments.append(
            name
            if name is not None
            else int(index)
            if index is not None
            else json.loads(quoted)
        )
        pos = s.end()
    return AcsPath(text, m.group(1) or "snap", tuple(segments))


def resolve(value: JsonValue, segments: tuple[str | int, ...]) -> JsonValue:
    """Read ``segments`` from ``value`` without coercion (spec §3)."""
    for seg in segments:
        if not isinstance(value, list if isinstance(seg, int) else dict):
            raise AcsError("runtime_error:path_type_mismatch", f"segment {seg!r}")
        if (seg >= len(value)) if isinstance(seg, int) else (seg not in value):
            raise AcsError("runtime_error:path_missing", f"segment {seg!r}")
        value = value[seg]
    return value


@dataclass(frozen=True, slots=True)
class Annotation:
    name: str
    source: AcsPath
    binding: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class PointConfig:
    """A validated intervention point entry with parsed paths; annotations sorted by name (spec §10)."""

    name: str
    policy_target: AcsPath
    policy_target_kind: str | None
    tool_name_from: AcsPath | None
    annotations: tuple[Annotation, ...]
    policy_id: str
    binding: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Manifest:
    """A validated, fully merged manifest."""

    data: Mapping[str, JsonValue]
    points: Mapping[str, PointConfig]

    @property
    def policies(self) -> Mapping[str, Mapping[str, JsonValue]]:
        return self.data["policies"]

    @property
    def tools(self) -> Mapping[str, Mapping[str, JsonValue]]:
        return self.data.get("tools") or {}

    @property
    def annotators(self) -> Mapping[str, Mapping[str, JsonValue]]:
        return self.data.get("annotators") or {}

    @property
    def approval(self) -> Mapping[str, JsonValue]:
        return self.data.get("approval") or {}


@cache
def schema_validator(name: str) -> Draft202012Validator:
    """A draft 2020-12 validator for a vendored schema (``manifest.schema.json``, ``wire/verdict.schema.json``)."""
    base = resources.files(__package__).joinpath("schema")
    names = {name, "manifest.schema.json", "approval.schema.json"}
    docs = {n: json.loads(base.joinpath(n).read_text("utf-8")) for n in names}
    registry = Registry().with_resources(
        (d["$id"], Resource.from_contents(d)) for d in docs.values()
    )
    return Draft202012Validator(docs[name], registry=registry)


def _checked_path(
    text: str, roots: set[str], where: str, *, annotation: bool = False
) -> AcsPath:
    try:
        path = parse_path(text)
    except ValueError as exc:
        raise AcsError(INVALID, f"{where}: {exc}") from None
    if (
        path.root not in roots
        or annotation
        and path.root == "pi"
        and path.segments[:1] == ("annotations",)
    ):
        raise AcsError(INVALID, f"{where}: root not allowed in {text!r}")
    return path


def _compile_point(
    name: str, entry: Mapping[str, Any], data: Mapping[str, Any]
) -> PointConfig:
    where = f"intervention_points.{name}"
    if "policy_target" not in entry or "policy" not in entry:
        raise AcsError(INVALID, f"{where}: policy_target and policy are required")
    binding = entry["policy"]
    policy = data["policies"].get(binding["id"])
    if policy is None:
        raise AcsError(INVALID, f"{where}: binds undefined policy {binding['id']!r}")
    if policy["type"] == "rego" and "query" not in binding and "query" not in policy:
        raise AcsError(INVALID, f"{where}: rego policy needs a query")
    if "tool_name_from" in entry and name not in TOOL_POINTS:
        raise AcsError(INVALID, f"{where}: tool_name_from on a non-tool point")
    declared = data.get("annotators") or {}
    annotations = []
    for ann_name, ann in sorted((entry.get("annotations") or {}).items()):
        if ann_name not in declared:
            raise AcsError(
                INVALID, f"{where}: annotation {ann_name!r} names no declared annotator"
            )
        src = _checked_path(
            ann["from"], {"pi", "target", "tool", "snap"}, where, annotation=True
        )
        annotations.append(Annotation(ann_name, src, ann))
    tool_from = entry.get("tool_name_from")
    return PointConfig(
        name=name,
        policy_target=_checked_path(entry["policy_target"], {"snap"}, where),
        policy_target_kind=entry.get("policy_target_kind"),
        tool_name_from=_checked_path(tool_from, {"snap"}, where)
        if tool_from is not None
        else None,
        annotations=tuple(annotations),
        policy_id=binding["id"],
        binding=binding,
    )


def validate_manifest(data: object, *, limits: Limits = DEFAULT_LIMITS) -> Manifest:
    """Validate an already-merged manifest mapping; non-empty ``extends`` fails closed (spec §2.2)."""
    if not isinstance(data, Mapping):
        raise AcsError(INVALID, "manifest must be an object")
    try:
        size = len(canonical_bytes(data))
    except (TypeError, ValueError) as exc:
        raise AcsError(INVALID, f"not JSON-compatible: {exc}") from None
    if size > limits.max_merged_manifest_bytes:
        raise AcsError(
            "runtime_error:resource_limit_exceeded", "max_merged_manifest_bytes"
        )
    errors = schema_validator("manifest.schema.json").iter_errors(data)
    # §9: tool entries must be objects but their members are unconstrained; the
    # published schema's tighter `security_labels` shape is not normative.
    error = next(
        (
            e
            for e in errors
            if not (len(e.absolute_path) > 2 and e.absolute_path[0] == "tools")
        ),
        None,
    )
    if error is not None:
        raise AcsError(
            INVALID, f"{'/'.join(map(str, error.absolute_path))}: {error.message}"
        )
    if data.get("extends"):
        raise AcsError(
            INVALID, "unresolved extends; load from a file or pass a merged manifest"
        )
    if not data.get("policies") or not data.get("intervention_points"):
        raise AcsError(
            INVALID, "policies and intervention_points are required and non-empty"
        )
    sections = [*data["policies"].values(), *(data.get("annotators") or {}).values()]
    for entry in data["intervention_points"].values():
        sections += [
            entry.get("policy") or {},
            *(entry.get("annotations") or {}).values(),
        ]
    if removed := sorted({f for s in sections for f in REMOVED_FIELDS if f in s}):
        raise AcsError(INVALID, f"removed field(s) declared: {', '.join(removed)}")
    points = {
        n: _compile_point(n, e, data) for n, e in data["intervention_points"].items()
    }
    return Manifest(
        data={k: v for k, v in data.items() if k != "extends"}, points=points
    )


def parse_manifest_text(
    text: str | bytes, *, limits: Limits = DEFAULT_LIMITS
) -> object:
    """Parse YAML (a JSON superset) manifest text."""
    if len(text) > limits.max_merged_manifest_bytes:
        raise AcsError(
            "runtime_error:resource_limit_exceeded", "max_merged_manifest_bytes"
        )
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise AcsError(INVALID, f"unparseable manifest: {exc}") from None


def _merge_ok(path: tuple[str, ...]) -> bool:
    head = path[0]
    return (
        len(path) == 1
        or head == "metadata"
        or head == "intervention_points"
        and (len(path) == 2 or len(path) == 3 and path[2] == "annotations")
        or path == ("approval", "resolvers")
    )


def merge_manifests(
    base: Mapping[str, Any], child: Mapping[str, Any], path: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Additive merge (spec §2.2): identical duplicates are fine, any other conflict fails closed."""
    out = dict(base)
    for key, value in child.items():
        here = (*path, key)
        if key not in out or out[key] == value:
            out[key] = value
        elif (
            isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
            and _merge_ok(here)
        ):
            out[key] = merge_manifests(out[key], value, here)
        else:
            raise AcsError("runtime_error:resolution_merge_conflict", "/".join(here))
    return out


def _load_file(
    path: Path, root: Path, chain: tuple[Path, ...], limits: Limits
) -> dict[str, Any]:
    real = path.resolve()
    if not real.is_relative_to(root):
        raise AcsError("runtime_error:resolution_path_traversal", str(path))
    if real in chain:
        raise AcsError("runtime_error:resolution_cycle", str(path))
    if len(chain) > limits.max_extends_depth:
        raise AcsError("runtime_error:resource_limit_exceeded", "max_extends_depth")
    try:
        data = parse_manifest_text(real.read_bytes(), limits=limits)
    except OSError as exc:
        raise AcsError(INVALID, f"cannot read {path}: {exc.strerror}") from None
    if not isinstance(data, Mapping) or not isinstance(data.get("extends", []), list):
        raise AcsError(
            INVALID, f"{path}: manifest must be an object with an extends array"
        )
    merged: dict[str, Any] = {}
    for ref in data.get("extends", []):
        if not isinstance(ref, str) or not ref or _URL.match(ref):
            raise AcsError(
                INVALID, f"{path}: only local path extends are supported, got {ref!r}"
            )
        merged = merge_manifests(
            merged, _load_file(real.parent / ref, root, (*chain, real), limits)
        )
    return merge_manifests(merged, {k: v for k, v in data.items() if k != "extends"})


def load_manifest(
    source: str | bytes | Path | Mapping[str, Any], *, limits: Limits = DEFAULT_LIMITS
) -> Manifest:
    """Load a manifest from a file :class:`~pathlib.Path` (resolving local ``extends``), text, or a mapping."""
    if isinstance(source, Path):
        root = source.resolve().parent
        return validate_manifest(_load_file(source, root, (), limits), limits=limits)
    data = (
        source
        if isinstance(source, Mapping)
        else parse_manifest_text(source, limits=limits)
    )
    return validate_manifest(data, limits=limits)
