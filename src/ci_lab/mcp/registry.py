"""MCP server registry (frozen) and exposure overlay (evolvable).

``src/ci_lab/mcp/servers.yaml`` (``ci_lab.mcp.v1``) is part of the frozen control plane: it fixes each
server's argv, roots, forwarded env vars, the tools it may ever expose, and the code-mode caps.
``harness/mcp/exposure.yaml`` (``ci_lab.mcp.exposure.v1``) is candidate surface and may only *narrow*
that: pick a subset of ``tools_allow`` (with description overrides), a ``mode`` (``direct`` | ``code``),
and code-mode ``timeout_s`` / ``max_output_chars`` / ``imports`` within the frozen caps.
"""

from __future__ import annotations

import keyword
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FORMAT = "ci_lab.mcp.v1"
EXPOSURE_FORMAT = "ci_lab.mcp.exposure.v1"
FROZEN_SERVERS = Path(__file__).with_name("servers.yaml")
MODES = ("direct", "code")
MAX_DESCRIPTION_CHARS = 1000
# Never importable from code mode, even if a registry lists them (A18).
FORBIDDEN_MODULES = frozenset({
    "subprocess", "os", "socket", "ctypes", "importlib", "sys", "builtins", "shutil", "pathlib", "io",
    "multiprocessing", "threading", "signal", "pickle", "marshal", "inspect", "gc", "code", "codeop",
    "runpy", "pty", "asyncio", "urllib", "http", "ssl", "select", "selectors", "mmap", "winreg", "_winapi",
})
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SHELL_META = re.compile(r"[|&;<>`$\n]")


class McpConfigError(ValueError):
    """An invalid registry or exposure file. ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class ServerDef:
    name: str
    command: tuple[str, ...]
    roots: Mapping[str, str] = field(default_factory=dict)
    env_allow: tuple[str, ...] = ()
    tools_allow: tuple[str, ...] = ()

    def resolve_roots(self, overrides: Mapping[str, str | Path] | None = None,
                      base_dir: str | Path | None = None) -> dict[str, Path]:
        """Absolute root dirs: ``overrides`` win, else the registry default relative to ``base_dir``."""
        overrides = dict(overrides or {})
        unknown = sorted(set(overrides) - set(self.roots))
        if unknown:
            raise McpConfigError([f"server {self.name!r}: unknown roots {unknown}"])
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        return {k: Path(overrides.get(k, base / v)).resolve() for k, v in self.roots.items()}

    def argv(self, roots: Mapping[str, Path]) -> list[str]:
        values = {"python": sys.executable, **{k: str(v) for k, v in roots.items()}}
        return [_PLACEHOLDER.sub(lambda m: values[m.group(1)], a) for a in self.command]

    def env(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        environ = os.environ if environ is None else environ
        return {k: environ[k] for k in self.env_allow if k in environ}


@dataclass(frozen=True)
class CodeModeCaps:
    timeout_s_max: float
    max_output_chars_max: int
    allowed_imports: tuple[str, ...]


@dataclass(frozen=True)
class Registry:
    servers: Mapping[str, ServerDef]
    code_mode: CodeModeCaps
    path: Path | None = None


@dataclass(frozen=True)
class CodeModeConfig:
    timeout_s: float
    max_output_chars: int
    imports: tuple[str, ...]

    @classmethod
    def from_caps(cls, caps: CodeModeCaps) -> CodeModeConfig:
        return cls(caps.timeout_s_max, caps.max_output_chars_max, caps.allowed_imports)


@dataclass(frozen=True)
class ServerExposure:
    mode: str
    tools: Mapping[str, str | None]  # tool name -> description override (None keeps the server's)


@dataclass(frozen=True)
class Exposure:
    servers: Mapping[str, ServerExposure]
    code_mode: CodeModeConfig

    @classmethod
    def everything(cls, registry: Registry, mode: str = "direct") -> Exposure:
        """Expose every registered server with all of its ``tools_allow`` (tests, operator tools)."""
        return cls({n: ServerExposure(mode, dict.fromkeys(s.tools_allow)) for n, s in registry.servers.items()},
                   CodeModeConfig.from_caps(registry.code_mode))


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise McpConfigError([f"{path}: {e}"]) from e


def _str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x for x in v)


def _ident(name: Any) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_")


def _num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and v > 0


def parse_registry(data: Any, path: Path | None = None) -> Registry:
    errs: list[str] = []
    if not isinstance(data, dict):
        raise McpConfigError(["registry must be a mapping"])
    if data.get("format") != REGISTRY_FORMAT:
        errs.append(f"format must be {REGISTRY_FORMAT!r}")
    if extra := set(data) - {"format", "servers", "code_mode"}:
        errs.append(f"unknown top-level keys {sorted(extra)}")
    servers: dict[str, ServerDef] = {}
    raw_servers = data.get("servers")
    if not isinstance(raw_servers, dict) or not raw_servers:
        errs.append("servers must be a non-empty mapping")
        raw_servers = {}
    for name, spec in raw_servers.items():
        where = f"servers.{name}"
        if not _ident(name):
            errs.append(f"{where}: name must be a python identifier")
        if not isinstance(spec, dict):
            errs.append(f"{where}: must be a mapping")
            continue
        if extra := set(spec) - {"command", "roots", "env_allow", "tools_allow"}:
            errs.append(f"{where}: unknown keys {sorted(extra)}")
        cmd, roots = spec.get("command"), spec.get("roots") or {}
        if not _str_list(cmd) or not cmd:
            errs.append(f"{where}.command: must be a non-empty argv list (shell strings are rejected)")
            cmd = []
        if not isinstance(roots, dict) or not all(_ident(k) and isinstance(v, str) and v for k, v in roots.items()):
            errs.append(f"{where}.roots: must map identifier -> path")
            roots = {}
        for a in cmd:
            if _SHELL_META.search(a):
                errs.append(f"{where}.command: shell metacharacters in {a!r}")
            for ph in _PLACEHOLDER.findall(a):
                if ph != "python" and ph not in roots:
                    errs.append(f"{where}.command: unknown placeholder {{{ph}}}")
        env_allow, tools = spec.get("env_allow", []), spec.get("tools_allow")
        if not _str_list(env_allow) and env_allow != []:
            errs.append(f"{where}.env_allow: must be a list of names")
            env_allow = []
        if not _str_list(tools) or not tools or not all(_ident(t) for t in tools):
            errs.append(f"{where}.tools_allow: must be a non-empty list of identifiers")
            tools = []
        servers[str(name)] = ServerDef(str(name), tuple(cmd), dict(roots), tuple(env_allow), tuple(tools))
    cm = data.get("code_mode")
    caps = CodeModeCaps(1.0, 1, ())
    if not isinstance(cm, dict):
        errs.append("code_mode must be a mapping")
    else:
        if extra := set(cm) - {"timeout_s_max", "max_output_chars_max", "allowed_imports"}:
            errs.append(f"code_mode: unknown keys {sorted(extra)}")
        t, m, imps = cm.get("timeout_s_max"), cm.get("max_output_chars_max"), cm.get("allowed_imports", [])
        if not _num(t):
            errs.append("code_mode.timeout_s_max must be a positive number")
        if not (_num(m) and isinstance(m, int)):
            errs.append("code_mode.max_output_chars_max must be a positive int")
        if not _str_list(imps) and imps != []:
            errs.append("code_mode.allowed_imports must be a list of module names")
        elif bad := sorted(i for i in imps if i.split(".")[0] in FORBIDDEN_MODULES or i.startswith("_")):
            errs.append(f"code_mode.allowed_imports: forbidden modules {bad}")
        if not errs:
            caps = CodeModeCaps(float(t), int(m), tuple(imps))
    if errs:
        raise McpConfigError(errs)
    return Registry(servers, caps, path)


def load_registry(path: str | Path | None = None) -> Registry:
    """Load and validate the frozen registry (``FROZEN_SERVERS`` by default)."""
    p = Path(path) if path is not None else FROZEN_SERVERS
    return parse_registry(_read_yaml(p), p)


def validate_exposure(data: Any, registry: Registry) -> list[str]:
    """Every way ``data`` exceeds what the frozen ``registry`` allows (empty = valid)."""
    if not isinstance(data, dict):
        return ["exposure must be a mapping"]
    errs: list[str] = []
    if data.get("format") != EXPOSURE_FORMAT:
        errs.append(f"format must be {EXPOSURE_FORMAT!r}")
    if extra := set(data) - {"format", "servers", "code_mode"}:
        errs.append(f"unknown top-level keys {sorted(extra)}")
    servers = data.get("servers", {})
    if not isinstance(servers, dict):
        return [*errs, "servers must be a mapping"]
    for name, spec in servers.items():
        where = f"servers.{name}"
        sdef = registry.servers.get(name)
        if sdef is None:
            errs.append(f"{where}: not in the frozen registry")
            continue
        if not isinstance(spec, dict):
            errs.append(f"{where}: must be a mapping")
            continue
        if extra := set(spec) - {"mode", "tools"}:
            errs.append(f"{where}: unknown keys {sorted(extra)}")
        if spec.get("mode", "direct") not in MODES:
            errs.append(f"{where}.mode must be one of {list(MODES)}")
        tools = spec.get("tools", {})
        if not isinstance(tools, dict):
            errs.append(f"{where}.tools must map tool name -> description (or null)")
            continue
        for tool, desc in tools.items():
            if tool not in sdef.tools_allow:
                errs.append(f"{where}.tools.{tool}: not in frozen tools_allow")
            if desc is not None and (not isinstance(desc, str) or len(desc) > MAX_DESCRIPTION_CHARS):
                errs.append(f"{where}.tools.{tool}: description must be a string of <= {MAX_DESCRIPTION_CHARS} chars")
    cm = data.get("code_mode", {})
    if not isinstance(cm, dict):
        return [*errs, "code_mode must be a mapping"]
    caps = registry.code_mode
    if extra := set(cm) - {"timeout_s", "max_output_chars", "imports"}:
        errs.append(f"code_mode: unknown keys {sorted(extra)}")
    if "timeout_s" in cm and not (_num(cm["timeout_s"]) and cm["timeout_s"] <= caps.timeout_s_max):
        errs.append(f"code_mode.timeout_s must be in (0, {caps.timeout_s_max}]")
    mo = cm.get("max_output_chars")
    if "max_output_chars" in cm and not (_num(mo) and isinstance(mo, int) and mo <= caps.max_output_chars_max):
        errs.append(f"code_mode.max_output_chars must be an int in (0, {caps.max_output_chars_max}]")
    if "imports" in cm:
        imps = cm["imports"]
        if not _str_list(imps) and imps != []:
            errs.append("code_mode.imports must be a list of module names")
        elif bad := sorted(set(imps) - set(caps.allowed_imports)):
            errs.append(f"code_mode.imports exceed the frozen allowed_imports: {bad}")
    return errs


def parse_exposure(data: Any, registry: Registry) -> Exposure:
    if errs := validate_exposure(data, registry):
        raise McpConfigError(errs)
    caps = registry.code_mode
    # An omitted ``tools`` key exposes all of tools_allow; an explicit empty mapping exposes none.
    servers = {n: ServerExposure(s.get("mode", "direct"),
                                 dict(s["tools"] or {}) if "tools" in s
                                 else dict.fromkeys(registry.servers[n].tools_allow))
               for n, s in (data.get("servers") or {}).items()}
    cm = data.get("code_mode") or {}
    cfg = CodeModeConfig(float(cm.get("timeout_s", caps.timeout_s_max)),
                         int(cm.get("max_output_chars", caps.max_output_chars_max)),
                         tuple(cm.get("imports", caps.allowed_imports)))
    return Exposure(servers, cfg)


def load_exposure(path: str | Path, registry: Registry | None = None) -> Exposure:
    """Load ``harness/mcp/exposure.yaml`` and validate it against the frozen registry."""
    return parse_exposure(_read_yaml(Path(path)), registry or load_registry())


def exposure_path(harness_dir: str | Path) -> Path:
    return Path(harness_dir) / "mcp" / "exposure.yaml"
