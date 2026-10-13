"""The repo-root evolvable ``harness/`` tree (contract v3 L3, v3.1 A1/A2/A4).

``harness/`` holds the agent-facing assets of the self-hosted harness: target-agent specs
(``agents/``), their prompts (``prompts/``), skills, target workflows, loop knobs, the client-tool
overlay, the MCP exposure and the guard bundle. Its control plane is frozen: components, owners,
required agents and caps come from :data:`FROZEN_MANIFEST` (``harness/harness.yaml`` must be an
identical copy), never from the candidate tree.

Library code never reads ``$CI_HARNESS_DIR``: callers pass ``harness_dir`` explicitly and
``None`` means :func:`repo_harness_dir` (the ``harness/`` of a source checkout). Only CLI entry
points call :func:`default_root`. Installed wheels do not ship ``harness/``; set
``CI_HARNESS_DIR`` or pass ``--harness-dir``/``--dir``.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ci_lab.gitops.safe_path import (
    UnsafePathError,
    is_link_like,
    match_globs,
    safe_join,
)

__all__ = [
    "ENV_VAR",
    "FORMAT",
    "FROZEN_MANIFEST",
    "LOOPS_FORMAT",
    "LOOP_KNOBS",
    "MANIFEST_NAME",
    "TOOLS_FORMAT",
    "HarnessSnapshot",
    "HarnessTree",
    "HarnessTreeError",
    "default_root",
    "load_manifest",
    "manifest_errors",
    "materialize",
    "repo_harness_dir",
    "snapshot",
    "tree_digest",
]

FORMAT = "ci_lab.harness.v1"
LOOPS_FORMAT = "ci_lab.harness.loops.v1"
TOOLS_FORMAT = "ci_lab.harness.tools.v1"
MANIFEST_NAME = "harness.yaml"
FROZEN_MANIFEST = Path(__file__).resolve().parent / "manifest.yaml"
ENV_VAR = "CI_HARNESS_DIR"
LOOP_KNOBS = ("max_nudges", "max_tool_calls", "max_turns")
CODE_MODE_KNOBS = ("max_runs",)
EVAL_CAPS = ("max_llm_calls", "max_tool_calls", "max_tokens", "timeout_s")
MAX_TOOL_DESCRIPTION = 1024
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SKIP_DIRS = frozenset({"__pycache__"})


class HarnessTreeError(ValueError):
    pass


def repo_harness_dir() -> Path:
    """``<repo>/harness`` of the source checkout this package runs from (no env lookup)."""
    return Path(__file__).resolve().parents[3] / "harness"


def default_root(environ: Mapping[str, str] | None = None) -> Path:
    """CLI default: ``$CI_HARNESS_DIR`` when set, else :func:`repo_harness_dir`."""
    env = os.environ if environ is None else environ
    return Path(env[ENV_VAR]) if env.get(ENV_VAR) else repo_harness_dir()


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HarnessTreeError(f"{path.name}: {exc}") from exc


def _norm(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def manifest_errors(data: Any) -> list[str]:
    """Every way ``data`` is not a valid ``ci_lab.harness.v1`` manifest (empty = valid)."""
    from ci_lab.contracts import COMPONENT_OWNERS, COMPONENTS

    if not isinstance(data, Mapping):
        return ["manifest must be a mapping"]
    errs: list[str] = []
    if data.get("format") != FORMAT:
        errs.append(f"format must be {FORMAT!r}")
    comps = data.get("components")
    if not isinstance(comps, Mapping):
        errs.append("components must map component -> globs")
        comps = {}
    for name, globs in comps.items():
        if name not in COMPONENTS:
            errs.append(f"components.{name}: unknown component (known: {', '.join(COMPONENTS)})")
        if not isinstance(globs, list) or not all(isinstance(g, str) and g for g in globs):
            errs.append(f"components.{name}: globs must be a list of strings")
        elif any(g.startswith(("/", "harness/")) or ".." in g.split("/") for g in globs):
            errs.append(f"components.{name}: globs are relative to harness/")
    owners = data.get("owners")
    if not isinstance(owners, Mapping) or dict(owners) != dict(COMPONENT_OWNERS):
        errs.append("owners must mirror ci_lab.contracts.COMPONENT_OWNERS")
    if not isinstance(data.get("frozen"), list):
        errs.append("frozen must be a list of repo-relative globs")
    req = data.get("required_agents")
    if not isinstance(req, list) or not all(isinstance(r, str) and _NAME.match(r) for r in req):
        errs.append("required_agents must be a list of agent names")
    caps = data.get("caps")
    if not isinstance(caps, Mapping):
        return [*errs, "caps must be a mapping"]
    for section, keys in (("agents", LOOP_KNOBS), ("code_mode", CODE_MODE_KNOBS), ("eval", EVAL_CAPS)):
        sec = caps.get(section)
        if not isinstance(sec, Mapping) or set(sec) != set(keys):
            errs.append(f"caps.{section} must set exactly {', '.join(keys)}")
        elif not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in sec.values()):
            errs.append(f"caps.{section}: values must be non-negative ints")
    return errs


@functools.lru_cache(maxsize=8)
def _load_manifest_cached(path: str, mtime_ns: int) -> dict[str, Any]:
    data = _read_yaml(Path(path))
    if errs := manifest_errors(data):
        raise HarnessTreeError(f"{Path(path).name}: " + "; ".join(errs))
    return data


def load_manifest(path: Path | str = FROZEN_MANIFEST) -> dict[str, Any]:
    """Parse and validate a manifest (default: the frozen one)."""
    p = Path(path)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError as exc:
        raise HarnessTreeError(f"{p}: {exc}") from exc
    return _load_manifest_cached(str(p.resolve()), mtime)


def _is_int(v: Any, low: int = 0) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= low


class HarnessTree:
    """One harness tree on disk, governed by the frozen manifest."""

    def __init__(self, root: Path | str, *, frozen_manifest: Path | str = FROZEN_MANIFEST) -> None:
        self.root = Path(root).resolve()
        self.frozen_manifest = Path(frozen_manifest)

    def __repr__(self) -> str:
        return f"HarnessTree({str(self.root)!r})"

    @property
    def manifest(self) -> dict[str, Any]:
        """The frozen manifest (the authority; the tree's own ``harness.yaml`` must equal it)."""
        return load_manifest(self.frozen_manifest)

    # ------------------------------------------------------------ paths
    def path(self, rel: str | Path) -> Path:
        """``root/rel``, contained under the root, with no symlink/junction on the way."""
        if not self.root.is_dir():
            raise HarnessTreeError(f"harness dir not found: {self.root}")
        try:
            return safe_join(self.root, rel)
        except UnsafePathError as exc:
            raise HarnessTreeError(f"{rel}: {exc}") from exc

    def _named(self, folder: str, name: str, suffix: str) -> Path:
        stem = name.removesuffix(suffix)
        if not _NAME.match(stem):
            raise HarnessTreeError(f"bad {folder} name {name!r}")
        return self.path(f"{folder}/{stem}{suffix}")

    def agent_spec_path(self, name: str) -> Path:
        return self._named("agents", name, ".yaml")

    def prompt_path(self, name: str) -> Path:
        return self._named("prompts", name, ".md")

    def workflow_path(self, name: str) -> Path:
        return self._named("workflows", name, ".yaml")

    def files(self) -> list[str]:
        """Sorted POSIX paths of every file under the root (``__pycache__`` skipped); a symlink fails."""
        if not self.root.is_dir():
            raise HarnessTreeError(f"harness dir not found: {self.root}")
        return sorted(_walk(self.root))

    # ------------------------------------------------------------ components
    def component_globs(self) -> dict[str, tuple[str, ...]]:
        """Repo-relative globs (prefixed ``harness/``) per component, from the frozen manifest."""
        return {c: tuple(f"harness/{g}" for g in globs) for c, globs in self.manifest["components"].items()}

    def component_of(self, rel: str) -> str | None:
        hits = [c for c, globs in self.manifest["components"].items() if match_globs(rel, globs)]
        return hits[0] if hits else None

    def agent_names(self) -> list[str]:
        d = self.root / "agents"
        return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []

    # ------------------------------------------------------------ overlays
    def _overlay(self, rel: str, fmt: str) -> dict[str, Any]:
        p = self.root / rel
        if not p.exists():
            return {}
        data = _read_yaml(self.path(rel))
        if not isinstance(data, Mapping) or data.get("format") != fmt:
            raise HarnessTreeError(f"{rel}: format must be {fmt!r}")
        return dict(data)

    def loop_errors(self, *, caps: bool = True) -> list[str]:
        """Structural problems of ``loops/loops.yaml`` (plus values above the frozen caps)."""
        try:
            data = self._overlay("loops/loops.yaml", LOOPS_FORMAT)
        except HarnessTreeError as exc:
            return [str(exc)]
        if not data:
            return []
        limits = self.manifest["caps"]
        extra = set(data) - {"format", "agents", "code_mode"}
        errs = [f"loops.yaml: unknown keys {sorted(extra)}"] if extra else []
        agents = data.get("agents") or {}
        if not isinstance(agents, Mapping):
            return [*errs, "loops.yaml: agents must be a mapping"]
        known = set(self.agent_names())
        for name, knobs in agents.items():
            where = f"loops.yaml: agents.{name}"
            if name not in known:
                errs.append(f"{where}: no agents/{name}.yaml")
            if not isinstance(knobs, Mapping):
                errs.append(f"{where} must be a mapping")
                continue
            for k, v in knobs.items():
                if k not in LOOP_KNOBS:
                    errs.append(f"{where}.{k}: unknown knob (known: {', '.join(LOOP_KNOBS)})")
                elif not _is_int(v, 0 if k == "max_nudges" else 1):
                    errs.append(f"{where}.{k} must be a {'non-negative' if k == 'max_nudges' else 'positive'} int")
                elif caps and v > limits["agents"][k]:
                    errs.append(f"{where}.{k}={v} exceeds the frozen cap {limits['agents'][k]}")
        cm = data.get("code_mode") or {}
        if not isinstance(cm, Mapping):
            return [*errs, "loops.yaml: code_mode must be a mapping"]
        for k, v in cm.items():
            if k not in CODE_MODE_KNOBS:
                errs.append(f"loops.yaml: code_mode.{k}: unknown knob")
            elif not _is_int(v, 1):
                errs.append(f"loops.yaml: code_mode.{k} must be a positive int")
            elif caps and v > limits["code_mode"][k]:
                errs.append(f"loops.yaml: code_mode.{k}={v} exceeds the frozen cap {limits['code_mode'][k]}")
        return errs

    def loops(self) -> dict[str, Any]:
        """``{"agents": {name: {knob: value}}, "code_mode": {...}}`` clamped to the frozen caps.

        Raises :class:`HarnessTreeError` on a malformed file (values above a cap are clamped)."""
        if errs := self.loop_errors(caps=False):
            raise HarnessTreeError("; ".join(errs))
        data = self._overlay("loops/loops.yaml", LOOPS_FORMAT)
        limits = self.manifest["caps"]
        agents = {n: {k: min(int(v), limits["agents"][k]) for k, v in (knobs or {}).items()}
                  for n, knobs in (data.get("agents") or {}).items()}
        cm = {k: min(int(v), limits["code_mode"][k]) for k, v in (data.get("code_mode") or {}).items()}
        return {"agents": agents, "code_mode": cm}

    def agent_loop(self, name: str) -> dict[str, int]:
        return dict(self.loops()["agents"].get(name) or {})

    def tools(self) -> dict[str, Any]:
        """``{"agents": {name: {tool: description | None}}}`` from ``tools/tools.yaml``.

        Listing an agent restricts it to the listed tools; a ``null`` description keeps the
        bound function's own. Whether each tool is bound by the spec is checked by the spec
        loader (:func:`ci_lab.meta.spec_loader.load_spec`)."""
        data = self._overlay("tools/tools.yaml", TOOLS_FORMAT)
        if not data:
            return {"agents": {}}
        if extra := set(data) - {"format", "agents"}:
            raise HarnessTreeError(f"tools.yaml: unknown keys {sorted(extra)}")
        agents = data.get("agents") or {}
        if not isinstance(agents, Mapping):
            raise HarnessTreeError("tools.yaml: agents must be a mapping")
        out: dict[str, dict[str, str | None]] = {}
        for name, tools in agents.items():
            if not isinstance(tools, Mapping) or not tools:
                raise HarnessTreeError(f"tools.yaml: agents.{name} must map tool -> description (or null)")
            for tool, desc in tools.items():
                if desc is not None and (not isinstance(desc, str) or not desc.strip()
                                         or len(desc) > MAX_TOOL_DESCRIPTION):
                    raise HarnessTreeError(f"tools.yaml: agents.{name}.{tool}: description must be null or a "
                                           f"non-empty string of <= {MAX_TOOL_DESCRIPTION} chars")
            out[str(name)] = {str(t): (d.strip() if isinstance(d, str) else None) for t, d in tools.items()}
        return {"agents": out}

    def agent_tools(self, name: str) -> dict[str, str | None] | None:
        return self.tools()["agents"].get(name)

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Every reason this tree is not a valid harness (empty = valid)."""
        if not self.root.is_dir():
            return [f"harness dir not found: {self.root}"]
        errs = self._manifest_check()
        try:
            files = self.files()
        except HarnessTreeError as exc:
            return [*errs, str(exc)]
        errs += [f"{rel}: not in any component" for rel in files
                 if rel != MANIFEST_NAME and self.component_of(rel) is None]
        names = self.agent_names()
        errs += [f"agents/{a}.yaml: required agent missing" for a in self.manifest["required_agents"]
                 if a not in names]
        specs = self._load_specs(names, errs)
        errs += self.loop_errors()
        try:
            for name in self.tools()["agents"]:
                if name not in names:
                    errs.append(f"tools.yaml: agents.{name}: no agents/{name}.yaml")
        except HarnessTreeError as exc:
            errs.append(str(exc))
        errs += self._exposure_errors()
        errs += self._workflow_errors({s.name for s in specs.values()})
        errs += self._skill_errors()
        return errs

    def _manifest_check(self) -> list[str]:
        declared = self.root / MANIFEST_NAME
        if not declared.is_file():
            return [f"{MANIFEST_NAME}: missing (copy {self.frozen_manifest.name} from ci_lab.harness_tree)"]
        have = hashlib.sha256(_norm(declared.read_bytes())).hexdigest()
        want = hashlib.sha256(_norm(self.frozen_manifest.read_bytes())).hexdigest()
        if have != want:
            return [f"{MANIFEST_NAME}: differs from the frozen manifest (sha256 {have[:12]} != {want[:12]})"]
        return []

    def _load_specs(self, names: list[str], errs: list[str]) -> dict[str, Any]:
        from ci_lab.meta.spec_loader import load_spec

        specs: dict[str, Any] = {}
        for name in names:
            try:
                specs[name] = load_spec(self.agent_spec_path(name), harness_dir=self.root)
            except Exception as exc:  # noqa: BLE001 - every load failure is a validation error
                errs.append(f"agents/{name}.yaml: {exc}")
        return specs

    def _exposure_errors(self) -> list[str]:
        p = self.root / "mcp" / "exposure.yaml"
        if not p.exists():
            return []
        from ci_lab.mcp.registry import McpConfigError, load_registry, validate_exposure

        try:
            data = _read_yaml(self.path("mcp/exposure.yaml"))
            return [f"mcp/exposure.yaml: {e}" for e in validate_exposure(data, load_registry())]
        except (HarnessTreeError, McpConfigError, OSError) as exc:
            return [f"mcp/exposure.yaml: {exc}"]

    def _workflow_errors(self, agent_names: set[str]) -> list[str]:
        from ci_lab.workflows import assert_expression_free

        allowed = set(self.manifest.get("workflow_functions") or ())
        errs: list[str] = []
        for p in sorted((self.root / "workflows").glob("*.yaml")):
            where = f"workflows/{p.name}"
            try:
                doc = assert_expression_free(self.path(f"workflows/{p.name}"))
            except (ValueError, OSError, yaml.YAMLError) as exc:
                errs.append(f"{where}: {exc}")
                continue
            for a in doc["trigger"]["actions"]:
                if a["kind"] == "InvokeAzureAgent" and a["agent"]["name"] not in agent_names:
                    errs.append(f"{where}: {a['id']}: agent {a['agent']['name']!r} is not an agent of this tree")
                if a["kind"] == "InvokeFunctionTool" and a.get("functionName") not in allowed:
                    errs.append(f"{where}: {a['id']}: function {a.get('functionName')!r} is not one of "
                                f"{sorted(allowed)}")
        return errs

    def _skill_errors(self) -> list[str]:
        errs: list[str] = []
        root = self.root / "skills"
        for d in sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []:
            where = f"skills/{d.name}/SKILL.md"
            f = d / "SKILL.md"
            if not f.is_file():
                errs.append(f"{where}: missing")
                continue
            meta = skill_frontmatter(f.read_text(encoding="utf-8"))
            if meta is None:
                errs.append(f"{where}: no YAML frontmatter")
            elif meta.get("name") != d.name or not str(meta.get("description") or "").strip():
                errs.append(f"{where}: frontmatter needs name: {d.name} and a description")
        return errs


def skill_frontmatter(text: str) -> dict[str, Any] | None:
    """The YAML frontmatter mapping of a ``SKILL.md`` (``None`` when absent or malformed)."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
        data = yaml.safe_load("\n".join(lines[1:end]))
    except (StopIteration, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _walk(root: Path) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for d in list(dirnames):
            if d in _SKIP_DIRS:
                dirnames.remove(d)
            elif is_link_like(base / d):
                raise HarnessTreeError(f"{(base / d).relative_to(root).as_posix()}: symlinks are not allowed")
        for f in filenames:
            p = base / f
            if is_link_like(p):
                raise HarnessTreeError(f"{p.relative_to(root).as_posix()}: symlinks are not allowed")
            yield p.relative_to(root).as_posix()


# ---------------------------------------------------------------- snapshots


def tree_digest(root: Path | str) -> str:
    """``sha256:`` over the sorted relative paths and bytes of every file under ``root``."""
    r = Path(root).resolve()
    h = hashlib.sha256()
    for rel in HarnessTree(r).files():
        data = (r / rel).read_bytes()
        h.update(rel.encode("utf-8") + b"\0" + str(len(data)).encode() + b"\0" + data)
    return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class HarnessSnapshot:
    """An incumbent harness tree pinned for one round: its root and :func:`tree_digest`."""

    root: Path
    digest: str

    def verify(self) -> HarnessSnapshot:
        """Raise :class:`HarnessTreeError` when the files under ``root`` no longer match ``digest``."""
        if (now := tree_digest(self.root)) != self.digest:
            raise HarnessTreeError(f"harness snapshot {self.root} changed ({now[:19]} != {self.digest[:19]})")
        return self

    def as_dict(self) -> dict[str, str]:
        return {"root": str(self.root), "digest": self.digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HarnessSnapshot:
        return cls(Path(str(data["root"])), str(data["digest"]))


def snapshot(root: Path | str) -> HarnessSnapshot:
    r = Path(root).resolve()
    return HarnessSnapshot(r, tree_digest(r))


def materialize(src: Path | str, dest: Path | str) -> HarnessSnapshot:
    """Copy the tree at ``src`` to the fresh dir ``dest`` and return the copy's snapshot
    (whose digest equals ``src``'s), so later edits of ``src`` cannot affect users of ``dest``."""
    source, target = Path(src).resolve(), Path(dest)
    if target.exists():
        raise HarnessTreeError(f"{target}: already exists")
    files = HarnessTree(source).files()
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    for rel in files:
        (staging / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, staging / rel)
    staging.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    snap = snapshot(target)
    if snap.digest != tree_digest(source):
        raise HarnessTreeError(f"{source} changed while it was copied to {target}")
    return snap
