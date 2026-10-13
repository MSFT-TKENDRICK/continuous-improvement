"""Declarative agent specs and the agent manifest (validated, expression-free).

Two YAML shapes are validated here:

* the **agent manifest** (``agents/manifest.yaml``) maps names to contained specs,
  runtimes, purposes, bindings and skill directories.

* a **declarative agent** (MAF ``kind: Prompt``) plus our repo-local ``x-ci``
  extension, which the loader strips and uses to compose ``instructions`` from
  Markdown files that live next to the YAML::

    kind: Prompt
    name: Analyst
    description: ...
    instructions: optional inline preamble
    model: {id: gpt-5-mini, provider: GitHubCopilot, options: {reasoningEffort: low}}
    tools:
      - kind: function
        name: read_file
        description: ...
        bindings: [{name: read_file}]
        parameters: {properties: {path: {kind: string, required: true}}}
    x-ci:
      instructions_files: [prompts/system.md]
      append_text: ""

Security rules enforced (arms are data-only, design §7/C13): model id allowlist,
provider allowlist, no string anywhere starting with ``=`` (PowerFx / ``=Env.``),
tool bindings limited to an allowed set, optional frozen tool parameter schemas
(only descriptions may evolve), ``options`` key allowlist and path containment for
every referenced file (no ``..``, absolute paths, drive letters or symlinks).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ci_lab.contracts import PROVIDER_NAME, Purpose

DEFAULT_ALLOWED_OPTIONS: frozenset[str] = frozenset({"reasoningEffort", "maxOutputTokens", "allowMultipleToolCalls"})
# JSON-schema keys that are documentation only; everything else in a tool's parameter
# schema is frozen when ``frozen_tool_schemas`` is supplied.
DOC_ONLY_SCHEMA_KEYS: frozenset[str] = frozenset({"description", "examples", "example", "title"})
XCI_KEY = "x-ci"

_AGENT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_MAX_INSTRUCTIONS_FILE_BYTES = 512 * 1024

Runtime = Literal["prompt", "harness"]


class SpecError(ValueError):
    """A manifest or declarative agent spec failed validation."""


# ---------------------------------------------------------------- path containment

def resolve_contained(base: Path, rel: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve ``rel`` under ``base``, rejecting anything that could escape it.

    Rejects absolute paths, drive/UNC prefixes, ``..`` components, ``:`` (Windows
    alternate data streams), and any symlink/junction along the way."""
    raw = str(rel)
    if not raw or raw.strip() != raw:
        raise SpecError(f"invalid relative path {raw!r}")
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise SpecError(f"absolute paths are not allowed: {raw!r}")
    parts = [p for p in re.split(r"[\\/]", raw) if p not in ("", ".")]
    if not parts:
        raise SpecError(f"empty relative path {raw!r}")
    for part in parts:
        if part == ".." or ":" in part:
            raise SpecError(f"path escapes its base directory: {raw!r}")
    base_resolved = Path(base).resolve()
    current = base_resolved
    for part in parts:
        current = current / part
        if current.is_symlink() or _is_junction(current):
            raise SpecError(f"symlinks are not allowed: {raw!r}")
    resolved = current.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise SpecError(f"path escapes its base directory: {raw!r}")
    if must_exist and not resolved.exists():
        raise SpecError(f"file not found: {raw!r} (under {base_resolved})")
    return resolved


def _is_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction()) if is_junction is not None else False


# ---------------------------------------------------------------- expression guard

def iter_strings(obj: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    """Yield ``(json_path, string)`` for every string key and value in ``obj``."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str):
                yield f"{path}.<key>", key
            yield from iter_strings(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from iter_strings(value, f"{path}[{i}]")


def find_expressions(obj: Any) -> list[str]:
    """JSON paths of every string starting with ``=`` (PowerFx expression marker)."""
    return [p for p, s in iter_strings(obj) if s.startswith("=")]


def assert_no_expressions(obj: Any, *, what: str = "spec") -> None:
    if found := find_expressions(obj):
        raise SpecError(f"{what} contains PowerFx-style '=' expressions at {', '.join(found[:5])}")


# ---------------------------------------------------------------- declarative agent schema

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolBinding(_Strict):
    name: str = Field(min_length=1)


class ToolSpec(_Strict):
    kind: Literal["function"]
    name: str
    description: str = Field(min_length=1)
    bindings: list[ToolBinding] = Field(min_length=1)
    parameters: dict[str, Any] | list[Any] | None = None
    strict: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _TOOL_NAME_RE.match(v):
            raise ValueError(f"bad tool name {v!r}")
        return v

    @field_validator("bindings", mode="before")
    @classmethod
    def _bindings(cls, v: Any) -> Any:
        # MAF also accepts ``{name: input}`` maps; normalise to the list form.
        if isinstance(v, Mapping):
            return [{"name": k} for k in v]
        return v

    def json_schema(self) -> dict[str, Any] | None:
        """The JSON schema MAF advertises for this tool (same conversion as AgentFactory)."""
        if self.parameters is None:
            return None
        from agent_framework_declarative._models import FunctionTool as DeclFunctionTool

        decl = DeclFunctionTool(name=self.name, description=self.description,
                                parameters=copy.deepcopy(self.parameters))
        return decl.parameters.to_json_schema() if decl.parameters is not None else None


class ModelSpec(_Strict):
    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(_Strict):
    kind: Literal["Prompt"]
    name: str
    description: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    model: ModelSpec
    tools: list[ToolSpec] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    outputSchema: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _AGENT_NAME_RE.match(v):
            raise ValueError(f"bad agent name {v!r}")
        return v

    @field_validator("tools")
    @classmethod
    def _unique_tools(cls, v: list[ToolSpec]) -> list[ToolSpec]:
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tool names: {names}")
        return v

    def binding_names(self) -> set[str]:
        return {b.name for t in self.tools for b in t.bindings}

    def tool_schemas(self) -> dict[str, dict[str, Any] | None]:
        """``{tool name: parameter JSON schema}`` — use this to produce ``frozen_tool_schemas``."""
        return {t.name: t.json_schema() for t in self.tools}

    def declarative_dict(self, *, include_model_identity: bool = False) -> dict[str, Any]:
        """Plain MAF ``kind: Prompt`` dict. Without ``include_model_identity`` the model
        ``id``/``provider`` are dropped so ``AgentFactory(client=...)`` uses the injected
        client instead of constructing one from the provider mapping."""
        data = self.model_dump(mode="json", exclude_none=True)
        model = data.pop("model")
        if not include_model_identity:
            model = {"options": model["options"]} if model.get("options") else {}
        elif not model.get("options"):
            model.pop("options", None)
        if model:
            data["model"] = model
        if not data.get("tools"):
            data.pop("tools", None)
        return data

    def digest(self) -> str:
        """Stable sha256 of the composed spec (provenance)."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


class XCi(_Strict):
    instructions_files: list[str] = Field(default_factory=list)
    append_text: str = ""


@dataclass(frozen=True)
class LoadedSpec:
    spec: AgentSpec
    path: Path | None
    instruction_files: tuple[Path, ...] = ()
    source_sha256: str = ""


def load_agent_spec(path: str | Path, *, allowed_models: Collection[str],
                    allowed_providers: Collection[str] = (), allowed_bindings: Collection[str] | None = None,
                    frozen_tool_schemas: Mapping[str, Any] | None = None,
                    allowed_options: Collection[str] = DEFAULT_ALLOWED_OPTIONS,
                    extra_instructions: str = "") -> LoadedSpec:
    """Read + validate a declarative agent YAML file (``x-ci`` resolved against its dir)."""
    path = Path(path)
    if path.is_symlink():
        raise SpecError(f"agent spec may not be a symlink: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read agent spec {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid YAML in {path}: {exc}") from exc
    loaded = parse_agent_spec(data, base_dir=path.parent, allowed_models=allowed_models,
                              allowed_providers=allowed_providers, allowed_bindings=allowed_bindings,
                              frozen_tool_schemas=frozen_tool_schemas, allowed_options=allowed_options,
                              extra_instructions=extra_instructions)
    return LoadedSpec(spec=loaded.spec, path=path.resolve(), instruction_files=loaded.instruction_files,
                      source_sha256=hashlib.sha256(text.encode()).hexdigest())


def parse_agent_spec(data: Any, *, base_dir: str | Path | None, allowed_models: Collection[str],
                     allowed_providers: Collection[str] = (), allowed_bindings: Collection[str] | None = None,
                     frozen_tool_schemas: Mapping[str, Any] | None = None,
                     allowed_options: Collection[str] = DEFAULT_ALLOWED_OPTIONS,
                     extra_instructions: str = "") -> LoadedSpec:
    """Validate an already-parsed declarative agent mapping. ``base_dir`` anchors
    ``x-ci.instructions_files`` (required when that list is non-empty)."""
    if not isinstance(data, Mapping):
        raise SpecError("agent spec must be a mapping")
    raw = dict(data)
    assert_no_expressions(raw, what="agent spec")
    xci = _parse_xci(raw.pop(XCI_KEY, None))
    files = _resolve_instruction_files(xci, base_dir)
    parts = [str(raw["instructions"]).strip()] if isinstance(raw.get("instructions"), str) else []
    parts += [_read_instructions_file(f) for f in files]
    parts += [s.strip() for s in (xci.append_text, extra_instructions) if s and s.strip()]
    composed = "\n\n".join(p for p in parts if p)
    if composed:
        raw["instructions"] = composed
    try:
        spec = AgentSpec.model_validate(raw)
    except ValidationError as exc:
        raise SpecError(f"invalid agent spec: {exc}") from exc
    assert_no_expressions(spec.model_dump(mode="json"), what="composed agent spec")
    validate_agent_spec(spec, allowed_models=allowed_models, allowed_providers=allowed_providers,
                        allowed_bindings=allowed_bindings, frozen_tool_schemas=frozen_tool_schemas,
                        allowed_options=allowed_options)
    return LoadedSpec(spec=spec, path=None, instruction_files=tuple(files))


def validate_agent_spec(spec: AgentSpec, *, allowed_models: Collection[str],
                        allowed_providers: Collection[str] = (), allowed_bindings: Collection[str] | None = None,
                        frozen_tool_schemas: Mapping[str, Any] | None = None,
                        allowed_options: Collection[str] = DEFAULT_ALLOWED_OPTIONS) -> None:
    if spec.model.id not in set(allowed_models):
        raise SpecError(f"model {spec.model.id!r} is not in the allowlist {sorted(allowed_models)}")
    providers = {PROVIDER_NAME, *allowed_providers}
    if spec.model.provider not in providers:
        raise SpecError(f"provider {spec.model.provider!r} is not allowed (allowed: {sorted(providers)})")
    if bad := set(spec.model.options) - set(allowed_options):
        raise SpecError(f"model options not allowed: {sorted(bad)} (allowed: {sorted(allowed_options)})")
    if allowed_bindings is not None and (bad := spec.binding_names() - set(allowed_bindings)):
        raise SpecError(f"tool bindings not allowed: {sorted(bad)} (allowed: {sorted(allowed_bindings)})")
    if frozen_tool_schemas is not None:
        check_frozen_tool_schemas(spec, frozen_tool_schemas)


def check_frozen_tool_schemas(spec: AgentSpec, frozen: Mapping[str, Any]) -> None:
    """Tool set and parameter schemas must match ``frozen`` modulo documentation keys."""
    actual = spec.tool_schemas()
    if set(actual) != set(frozen):
        raise SpecError(f"tool set changed: expected {sorted(frozen)}, got {sorted(actual)}")
    for name, schema in actual.items():
        if strip_doc_keys(schema) != strip_doc_keys(frozen[name]):
            raise SpecError(f"tool {name!r} parameter schema differs from the frozen schema "
                            "(only descriptions may change)")


def strip_doc_keys(schema: Any) -> Any:
    """Copy of a JSON schema without documentation-only keys (description, examples...).
    Keys under ``properties`` are property names, never stripped."""
    if isinstance(schema, Mapping):
        out: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "properties" and isinstance(value, Mapping):
                out[key] = {k: strip_doc_keys(v) for k, v in value.items()}
            elif key in DOC_ONLY_SCHEMA_KEYS:
                continue
            else:
                out[key] = strip_doc_keys(value)
        return out
    if isinstance(schema, list):
        return [strip_doc_keys(v) for v in schema]
    return schema


def _parse_xci(value: Any) -> XCi:
    if value is None:
        return XCi()
    try:
        return XCi.model_validate(value)
    except ValidationError as exc:
        raise SpecError(f"invalid {XCI_KEY} block: {exc}") from exc


def _resolve_instruction_files(xci: XCi, base_dir: str | Path | None) -> list[Path]:
    if not xci.instructions_files:
        return []
    if base_dir is None:
        raise SpecError(f"{XCI_KEY}.instructions_files requires a base directory")
    files = [resolve_contained(Path(base_dir), rel) for rel in xci.instructions_files]
    for f in files:
        if not f.is_file():
            raise SpecError(f"instructions file is not a regular file: {f}")
    return files


def _read_instructions_file(path: Path) -> str:
    if path.stat().st_size > _MAX_INSTRUCTIONS_FILE_BYTES:
        raise SpecError(f"instructions file too large: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------- manifest

class AgentEntry(_Strict):
    spec: str = Field(min_length=1)
    runtime: Runtime = "prompt"
    purpose: Purpose
    bindings: list[str] = Field(default_factory=list)
    skills_paths: list[str] = Field(default_factory=list)


class AgentManifest(_Strict):
    agents: dict[str, AgentEntry] = Field(min_length=1)

    @field_validator("agents")
    @classmethod
    def _names(cls, v: dict[str, AgentEntry]) -> dict[str, AgentEntry]:
        for name in v:
            if not _AGENT_NAME_RE.match(name):
                raise ValueError(f"bad agent name {name!r}")
        return v


@dataclass(frozen=True)
class ResolvedEntry:
    name: str
    entry: AgentEntry
    spec_path: Path
    skills_paths: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LoadedManifest:
    path: Path
    manifest: AgentManifest
    entries: dict[str, ResolvedEntry]


def load_manifest(path: str | Path) -> LoadedManifest:
    """Read + validate a manifest; every referenced path is contained under its dir."""
    path = Path(path)
    if path.is_symlink():
        raise SpecError(f"manifest may not be a symlink: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"cannot read manifest {path}: {exc}") from exc
    assert_no_expressions(data, what="manifest")
    try:
        manifest = AgentManifest.model_validate(data)
    except ValidationError as exc:
        raise SpecError(f"invalid manifest {path}: {exc}") from exc
    base = path.parent
    entries: dict[str, ResolvedEntry] = {}
    for name, entry in manifest.agents.items():
        spec_path = resolve_contained(base, entry.spec)
        if not spec_path.is_file():
            raise SpecError(f"agent {name}: spec is not a file: {entry.spec!r}")
        skills = tuple(resolve_contained(base, p) for p in entry.skills_paths)
        for s in skills:
            if not s.is_dir():
                raise SpecError(f"agent {name}: skills path is not a directory: {s}")
        entries[name] = ResolvedEntry(name=name, entry=entry, spec_path=spec_path, skills_paths=skills)
    return LoadedManifest(path=path.resolve(), manifest=manifest, entries=entries)
