"""Meta-agent specs: declarative ``kind: Prompt`` YAML + the ``x-ci`` extension block.

``x-ci`` is our extension to the MAF declarative schema (``AgentDefinition`` rejects unknown
top-level keys, so it is popped before validation):

* ``instructions_files``: Markdown files (relative to the spec) appended to ``instructions``
* ``terminal_tool``: the ``submit_*`` tool whose call ends the run
* ``skills_paths``: Agent Skills directories (relative to the spec dir, else the repo root)
* ``documents``: run documents the agent may read with ``read_brief``
* ``purpose``: :data:`ci_lab.contracts.Purpose` used to obtain the chat client

The agents run as MAF harness agents (``runtime: harness`` in ``manifest.yaml``). A builder
(:class:`AgentBuilder`) turns a spec + bound tool functions into a runnable agent; the default
prefers ``ci_lab.maf.loader.build_agent(..., runtime="harness")`` and falls back to
:func:`harness_builder` (``agent_framework.create_harness_agent``).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml
from agent_framework import FunctionMiddleware, MiddlewareTermination

__all__ = [
    "SPECS_DIR",
    "AgentBuilder",
    "MetaAgentSpec",
    "SpecError",
    "TerminalSubmitMiddleware",
    "default_builder",
    "harness_builder",
    "load_manifest",
    "load_spec",
    "loader_builder",
]

SPECS_DIR = Path(__file__).resolve().parent / "specs"
REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS = ("analyst", "proposer", "critic", "reflector")
HARNESS_DEFAULTS: Mapping[str, Any] = {"disable_todo": True, "disable_mode": True, "disable_file_memory": True,
                                       "disable_web_search": True}


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class MetaAgentSpec:
    key: str
    path: Path
    name: str
    description: str
    instructions: str  # spec instructions + instructions_files, composed
    model: str
    provider: str
    purpose: str
    tools: tuple[str, ...]  # binding names, in spec order
    terminal_tool: str
    documents: tuple[str, ...] = ()
    skills_paths: tuple[Path, ...] = ()
    harness: Mapping[str, Any] = field(default_factory=dict)
    runtime: str = "harness"
    max_nudges: int = 2


def load_manifest(path: Path | str = SPECS_DIR / "manifest.yaml") -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("agents"), Mapping):
        raise SpecError(f"{path}: manifest has no agents mapping")
    return data


def _expressions(node: Any, where: str = "") -> list[str]:
    if isinstance(node, str):
        return [where or "<root>"] if node.lstrip().startswith("=") else []
    if isinstance(node, Mapping):
        return [x for k, v in node.items() for x in _expressions(v, f"{where}.{k}" if where else str(k))]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in _expressions(v, f"{where}[{i}]")]
    return []


def _validate_declarative(spec: Mapping[str, Any], path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from agent_framework_declarative._models import AgentDefinition

        try:
            agent = AgentDefinition.from_dict(dict(spec))
        except Exception as exc:
            raise SpecError(f"{path.name}: not a valid declarative agent: {exc}") from exc
    if type(agent).__name__ != "PromptAgent":
        raise SpecError(f"{path.name}: kind must be Prompt")


def _resolve_skill_path(raw: str, spec_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (spec_dir, REPO_ROOT):
        if (base / p).exists():
            return (base / p).resolve()
    return (REPO_ROOT / p).resolve()


def load_spec(key_or_path: str | Path, *, manifest: Mapping[str, Any] | None = None) -> MetaAgentSpec:
    """Load + validate one meta-agent spec by manifest key (``"critic"``) or path."""
    manifest = manifest if manifest is not None else load_manifest()
    entry: Mapping[str, Any] = {}
    if isinstance(key_or_path, str) and key_or_path in manifest["agents"]:
        key = key_or_path
        entry = manifest["agents"][key]
        path = (SPECS_DIR / entry["spec"]).resolve()
    else:
        path = Path(key_or_path).resolve()
        key = path.stem
        entry = manifest["agents"].get(key, {})
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    xci = dict(spec.pop("x-ci", None) or {})
    if bad := _expressions(spec) + _expressions(xci):
        raise SpecError(f"{path.name}: expressions are not allowed (safe_mode): {', '.join(bad)}")
    _validate_declarative(spec, path)

    tools: list[str] = []
    for t in spec.get("tools") or []:
        if t.get("kind") != "function":
            raise SpecError(f"{path.name}: only function tools are allowed, got {t.get('kind')!r}")
        bindings = [b.get("name") for b in t.get("bindings") or []]
        if len(bindings) != 1 or not bindings[0]:
            raise SpecError(f"{path.name}: tool {t.get('name')!r} needs exactly one binding")
        if bindings[0] != t.get("name"):
            raise SpecError(f"{path.name}: tool {t.get('name')!r} must bind a function of the same name")
        tools.append(bindings[0])
    terminal = str(xci.get("terminal_tool") or entry.get("terminal_tool") or "")
    if not terminal.startswith("submit_") or terminal not in tools:
        raise SpecError(f"{path.name}: terminal_tool {terminal!r} must be a submit_* tool bound by the spec")
    if entry.get("terminal_tool") and entry["terminal_tool"] != terminal:
        raise SpecError(f"{path.name}: terminal_tool disagrees with manifest")

    parts = [str(spec.get("instructions") or "").strip()]
    for rel in xci.get("instructions_files") or []:
        f = (path.parent / rel).resolve()
        if not f.is_relative_to(path.parent.resolve()) or not f.is_file():
            raise SpecError(f"{path.name}: instructions file {rel!r} not found under the spec dir")
        parts.append(f.read_text(encoding="utf-8").strip())
    skills = tuple(_resolve_skill_path(str(p), path.parent) for p in xci.get("skills_paths") or [])
    for s in skills:
        if not s.is_dir():
            raise SpecError(f"{path.name}: skills path {s} does not exist")
    model = spec.get("model") or {}
    harness = {**HARNESS_DEFAULTS, **(manifest.get("harness") or {}), **(xci.get("harness") or {})}
    nudges = int(harness.pop("max_nudges", 2))
    return MetaAgentSpec(
        key=key, path=path, name=str(spec.get("name")), description=str(spec.get("description") or ""),
        instructions="\n\n".join(p for p in parts if p), model=str(model.get("id") or manifest.get("model") or ""),
        provider=str(model.get("provider") or ""), purpose=str(xci.get("purpose") or entry.get("purpose") or key),
        tools=tuple(tools), terminal_tool=terminal, documents=tuple(xci.get("documents") or ()),
        skills_paths=skills, harness=harness, runtime=str(entry.get("runtime") or manifest.get("runtime") or "harness"),
        max_nudges=nudges)


# ---------------------------------------------------------------- building


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Sequence):
        return " ".join(str(getattr(c, "text", None) or c) for c in result)
    return str(getattr(result, "text", None) or result)


class TerminalSubmitMiddleware(FunctionMiddleware):
    """Ends the tool loop right after a successful call of the terminal ``submit_*`` tool."""

    def __init__(self, terminal_tool: str) -> None:
        self.terminal_tool = terminal_tool
        self.submitted = False

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        await call_next()
        if getattr(context.function, "name", None) != self.terminal_tool:
            return
        if _result_text(context.result).lstrip().startswith(("ERROR", "Error")):
            return
        self.submitted = True
        raise MiddlewareTermination(result=context.result)


class AgentBuilder(Protocol):
    """Builds a runnable agent (``.run(message, session=...)``, ``.create_session()``)."""

    def __call__(self, spec: MetaAgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
                 middleware: Sequence[Any] = (), loop_should_continue: Callable[..., Any] | None = None,
                 loop_next_message: Callable[..., Any] | None = None) -> Any: ...


def harness_builder(spec: MetaAgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
                    middleware: Sequence[Any] = (), loop_should_continue: Callable[..., Any] | None = None,
                    loop_next_message: Callable[..., Any] | None = None) -> Any:
    """Local runtime=harness build: ``create_harness_agent`` with the spec's tools bound in order."""
    from agent_framework import create_harness_agent

    missing = [t for t in spec.tools if t not in bindings]
    if missing:
        raise SpecError(f"{spec.path.name}: no binding for tool(s) {', '.join(missing)}")
    kwargs: dict[str, Any] = {k: v for k, v in spec.harness.items()
                              if k in inspect.signature(create_harness_agent).parameters}
    if spec.skills_paths:
        kwargs["skills_paths"] = [str(p) for p in spec.skills_paths]
    if loop_should_continue is not None:
        kwargs.update(loop_should_continue=loop_should_continue, loop_next_message=loop_next_message,
                      loop_max_iterations=spec.max_nudges + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MAF harness APIs emit ExperimentalWarning
        return create_harness_agent(client, name=spec.name, description=spec.description,
                                    agent_instructions=spec.instructions,
                                    tools=[bindings[t] for t in spec.tools], middleware=list(middleware), **kwargs)


def loader_builder(build_agent: Callable[..., Any]) -> AgentBuilder:
    """Adapt ``ci_lab.maf.loader.build_agent(spec_path, client=, bindings=, runtime="harness", ...)``.

    Only keyword arguments the loader accepts are passed (it must honor ``x-ci``).
    """
    params = inspect.signature(build_agent).parameters
    open_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def build(spec: MetaAgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
              middleware: Sequence[Any] = (), loop_should_continue: Callable[..., Any] | None = None,
              loop_next_message: Callable[..., Any] | None = None) -> Any:
        extra = {"middleware": list(middleware), "loop_should_continue": loop_should_continue,
                 "loop_next_message": loop_next_message}
        kwargs = {k: v for k, v in extra.items() if v is not None and (open_kwargs or k in params)}
        return build_agent(spec.path, client=client, bindings={t: bindings[t] for t in spec.tools},
                           runtime=spec.runtime, **kwargs)

    return build


def default_builder() -> AgentBuilder:
    """``ci_lab.maf.loader`` when present (built in parallel, M2); else :func:`harness_builder`.

    A loader build error falls back to the local builder (building has no side effects).
    """
    try:
        from ci_lab.maf.loader import build_agent  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return harness_builder
    via_loader = loader_builder(build_agent)

    def build(spec: MetaAgentSpec, **kwargs: Any) -> Any:
        try:
            return via_loader(spec, **kwargs)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"ci_lab.maf.loader could not build {spec.key} ({exc}); using local harness builder",
                          RuntimeWarning, stacklevel=2)
            return harness_builder(spec, **kwargs)

    return build
