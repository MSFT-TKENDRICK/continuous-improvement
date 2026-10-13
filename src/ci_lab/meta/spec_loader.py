"""Meta-agent specs: declarative ``kind: Prompt`` YAML + the ``x-ci`` extension block.

``x-ci`` is our extension to the MAF declarative schema (``AgentDefinition`` rejects unknown
top-level keys, so it is popped before validation):

* ``instructions_files``: Markdown files (relative to the spec) appended to ``instructions``; a
  spec in a harness tree (``harness/agents/*.yaml``) may use any file of its tree
  (``../prompts/common.md``), other specs only files under their own dir
* ``terminal_tool``: the ``submit_*`` tool whose call ends the run
* ``skills_paths``: Agent Skills directories (relative to the spec dir, else the repo root)
* ``documents``: run documents the agent may read with ``read_brief``
* ``purpose``: :data:`ci_lab.contracts.Purpose` used to obtain the chat client
* ``subagents``: MAF background agents this agent may delegate to: names from the manifest's
  ``subagents`` section, or ``*.yaml`` spec files contained in the spec dir
* ``subagent_instructions``: override for MAF's background-agents instructions (may contain
  the ``{background_agents}`` placeholder)
* ``role``: ``agent`` (default) or ``subagent``. A subagent has no terminal tool, answers in
  text, may only use the read-only tools in :data:`SUBAGENT_TOOLS` and runs on its parent's
  chat client.

Subagent references are validated fail-closed at load time (unknown name, cycle, non
read-only tool, model outside the manifest allowlist -> :class:`SpecError`).

Model selection: every agent and subagent uses the manifest ``model`` unless the operator sets
``CI_META_MODEL`` (``ci_lab.maf.models``). The override replaces ``model.id`` before validation
and hashing, so recorded specs and digests name the model that actually ran, and it must be in
``allowed_models``, which only ``CI_ALLOWED_MODELS`` (comma-separated) can extend: arm-editable
data never picks a meta-agent model.

The agents run as MAF harness agents (``runtime: harness`` in ``manifest.yaml``). A builder
(:class:`AgentBuilder`) turns a spec + bound tool functions into a runnable agent; the default
(:func:`validated_harness_builder`) validates the spec with ``ci_lab.maf.specs`` against the
manifest's ``allowed_models`` and builds with :func:`harness_builder`
(``agent_framework.create_harness_agent``).
"""

from __future__ import annotations

import copy
import inspect
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml
from agent_framework import (
    ChatMiddleware,
    ChatResponse,
    FunctionMiddleware,
    Message,
    MiddlewareFailure,
    MiddlewareTermination,
)

__all__ = [
    "SPECS_DIR",
    "SUBAGENT_TOOLS",
    "AgentBuilder",
    "MetaAgentSpec",
    "SpecError",
    "TerminalSubmitMiddleware",
    "ToolBudgetMiddleware",
    "TurnLimitMiddleware",
    "default_builder",
    "evolvable_agents",
    "harness_builder",
    "load_manifest",
    "load_spec",
    "loader_builder",
    "loop_limit_middleware",
    "manifest_allowed_models",
    "subagent_specs",
    "validated_harness_builder",
]

SPECS_DIR = Path(__file__).resolve().parent / "specs"
REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS = ("analyst", "proposer", "critic", "reflector")
HARNESS_DEFAULTS: Mapping[str, Any] = {"disable_todo": True, "disable_mode": True, "disable_file_memory": True,
                                       "disable_web_search": True}
# Read-only tools a subagent may bind (no write, commit, submit or other side effects).
SUBAGENT_TOOLS: frozenset[str] = frozenset({"read_brief", "list_documents", "read_history", "list_files",
                                            "read_file"})
ROLES = ("agent", "subagent")


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
    role: str = "agent"
    subagents: tuple[MetaAgentSpec, ...] = ()
    subagent_instructions: str | None = None
    harness_root: Path | None = None  # the harness tree an evolvable spec was loaded from
    # From the tree's loops/loops.yaml (clamped to the frozen caps; None = no limit) and
    # tools/tools.yaml (descriptions replacing the bound functions' own).
    max_tool_calls: int | None = None
    max_turns: int | None = None
    tool_descriptions: Mapping[str, str] = field(default_factory=dict)

def subagent_specs(spec: MetaAgentSpec) -> list[MetaAgentSpec]:
    """Every subagent below ``spec`` (depth-first, each key once)."""
    out: dict[str, MetaAgentSpec] = {}

    def walk(s: MetaAgentSpec) -> None:
        for sub in s.subagents:
            if sub.key not in out:
                out[sub.key] = sub
                walk(sub)

    walk(spec)
    return list(out.values())


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


def evolvable_agents(manifest: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Keys whose specs live in the harness tree (``<harness_dir>/agents``), not in :data:`SPECS_DIR`."""
    manifest = manifest if manifest is not None else load_manifest()
    return tuple(str(k) for k in manifest.get("evolvable") or ())


def _harness_root(harness_dir: Path | str | None) -> Path:
    from ci_lab.harness_tree import repo_harness_dir

    return Path(harness_dir).resolve() if harness_dir is not None else repo_harness_dir().resolve()


def _tree_root(path: Path, harness_dir: Path | str | None) -> Path | None:
    """The harness tree a path-loaded spec belongs to (``None`` = a standalone/frozen spec)."""
    if harness_dir is not None and path.is_relative_to(root := _harness_root(harness_dir)):
        return root
    from ci_lab.harness_tree import MANIFEST_NAME

    tree = path.parent.parent
    return tree if path.parent.name == "agents" and (tree / MANIFEST_NAME).is_file() else None


def load_spec(key_or_path: str | Path, *, manifest: Mapping[str, Any] | None = None,
              harness_dir: Path | str | None = None) -> MetaAgentSpec:
    """Load + validate one meta-agent spec by manifest key (``"critic"``, or a ``subagents`` key)
    or path, including its subagents (recursively).

    Evolvable keys (manifest ``evolvable``: analyst, proposer, reflector, failure_analyst,
    student) resolve to ``<harness_dir>/agents/<spec>``; ``harness_dir=None`` means the repo-root
    ``harness/`` (:func:`ci_lab.harness_tree.repo_harness_dir`; no env lookup). Frozen specs
    (critic, judge, adversary) ignore ``harness_dir``. A tree spec may read instruction files
    anywhere in its tree (``../prompts/*.md``); other specs only under their own dir."""
    manifest = manifest if manifest is not None else load_manifest()
    subs = manifest.get("subagents") or {}
    if not isinstance(subs, Mapping):
        raise SpecError("manifest: subagents must be a mapping")
    tree: Path | None = None
    if isinstance(key_or_path, str) and (key_or_path in manifest["agents"] or key_or_path in subs
                                         or key_or_path in evolvable_agents(manifest)):
        key = key_or_path
        entry: Mapping[str, Any] = manifest["agents"].get(key) or subs.get(key) or {}
        rel = str(entry.get("spec") or f"{key}.yaml")
        if key in evolvable_agents(manifest):
            from ci_lab.harness_tree import HarnessTree, HarnessTreeError

            tree = _harness_root(harness_dir)
            try:
                path = HarnessTree(tree).agent_spec_path(rel)
            except HarnessTreeError as exc:
                raise SpecError(f"{key}: {exc}") from exc
        else:
            path = (SPECS_DIR / rel).resolve()
    else:
        path = Path(key_or_path).resolve()
        key = path.stem
        entry = manifest["agents"].get(key) or subs.get(key) or {}
        tree = _tree_root(path, harness_dir)
    return _load(key, path, entry, manifest, stack=(), tree=tree)


def _subagent_ref(ref: Any, parent: Path, manifest: Mapping[str, Any]) -> tuple[str, Path, Mapping[str, Any]]:
    """Resolve a subagent reference (manifest ``subagents`` key or ``*.yaml`` path) next to ``parent``."""
    from ci_lab.maf.specs import SpecError as MafSpecError
    from ci_lab.maf.specs import resolve_contained

    subs = manifest.get("subagents") or {}
    if not isinstance(ref, str) or not ref.strip():
        raise SpecError(f"{parent.name}: subagent references must be non-empty strings, got {ref!r}")
    if ref in subs:
        key, rel, entry = ref, str(subs[ref].get("spec") or ""), subs[ref]
    elif ref.endswith((".yaml", ".yml")):
        key, rel = Path(ref).stem, ref
        entry = subs.get(key) or {}
    else:
        raise SpecError(f"{parent.name}: unknown subagent {ref!r} (known: {', '.join(sorted(subs)) or 'none'})")
    try:
        path = resolve_contained(parent.parent, rel)
    except MafSpecError as exc:
        raise SpecError(f"{parent.name}: subagent {ref!r}: {exc}") from exc
    return key, path, entry


def _load(key: str, path: Path, entry: Mapping[str, Any], manifest: Mapping[str, Any], *,
          stack: tuple[Path, ...], as_subagent: bool = False, tree: Path | None = None) -> MetaAgentSpec:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(p.stem for p in (*stack[stack.index(path):], path))
        raise SpecError(f"{stack[-1].name}: subagent cycle {chain}")
    if not path.is_file():
        raise SpecError(f"{path.name}: spec file not found")
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    xci = dict(spec.pop("x-ci", None) or {})
    if bad := _expressions(spec) + _expressions(xci):
        raise SpecError(f"{path.name}: expressions are not allowed (safe_mode): {', '.join(bad)}")
    _validate_declarative(spec, path)
    role = str(xci.get("role") or entry.get("role") or ("subagent" if key in (manifest.get("subagents") or {})
                                                        else "agent"))
    if role not in ROLES:
        raise SpecError(f"{path.name}: role {role!r} must be one of {', '.join(ROLES)}")
    if as_subagent and role != "subagent":
        raise SpecError(f"{path.name}: only role: subagent specs may be used as subagents")

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
    model = spec.get("model") or {}
    model_id = _model_override(path, manifest) or str(model.get("id") or manifest.get("model") or "")
    if role == "subagent":
        terminal = ""
        if xci.get("terminal_tool") or entry.get("terminal_tool"):
            raise SpecError(f"{path.name}: a subagent has no terminal_tool (it answers in text)")
        if bad := [t for t in tools if t not in SUBAGENT_TOOLS]:
            raise SpecError(f"{path.name}: subagent tool(s) {', '.join(bad)} are not read-only "
                            f"(allowed: {', '.join(sorted(SUBAGENT_TOOLS))})")
        if model_id not in manifest_allowed_models(manifest):
            raise SpecError(f"{path.name}: subagent model {model_id!r} is not in the allowlist "
                            f"{list(manifest_allowed_models(manifest))}")
    else:
        terminal = str(xci.get("terminal_tool") or entry.get("terminal_tool") or "")
        if not terminal.startswith("submit_") or terminal not in tools:
            raise SpecError(f"{path.name}: terminal_tool {terminal!r} must be a submit_* tool bound by the spec")
        if entry.get("terminal_tool") and entry["terminal_tool"] != terminal:
            raise SpecError(f"{path.name}: terminal_tool disagrees with manifest")

    loop, descriptions = _tree_overlays(tree, path) if tree is not None else ({}, None)
    if descriptions is not None:
        if unbound := [t for t in descriptions if t not in tools]:
            raise SpecError(f"{path.name}: tools/tools.yaml names tool(s) {', '.join(unbound)} the spec does not bind")
        if terminal and terminal not in descriptions:
            raise SpecError(f"{path.name}: tools/tools.yaml must keep the terminal tool {terminal!r}")
        tools = [t for t in tools if t in descriptions]

    parts = [str(spec.get("instructions") or "").strip()]
    base, where = (tree, "the harness dir") if tree is not None else (path.parent.resolve(), "the spec dir")
    for rel in xci.get("instructions_files") or []:
        f = (path.parent / rel).resolve()
        if not f.is_relative_to(base) or not f.is_file():
            raise SpecError(f"{path.name}: instructions file {rel!r} not found under {where}")
        parts.append(f.read_text(encoding="utf-8").strip())
    skills = tuple(_resolve_skill_path(str(p), path.parent) for p in xci.get("skills_paths") or [])
    for s in skills:
        if not s.is_dir():
            raise SpecError(f"{path.name}: skills path {s} does not exist")

    refs = xci.get("subagents") or []
    if not isinstance(refs, list):
        raise SpecError(f"{path.name}: x-ci.subagents must be a list")
    children = tuple(_load(*_subagent_ref(r, path, manifest), manifest, stack=(*stack, path), as_subagent=True,
                           tree=tree) for r in refs)
    names = [c.name.lower() for c in children]
    if len(set(names)) != len(names):
        raise SpecError(f"{path.name}: subagent names must be unique (case-insensitive)")
    sub_instructions = xci.get("subagent_instructions")
    if sub_instructions is not None and (not isinstance(sub_instructions, str) or not children):
        raise SpecError(f"{path.name}: subagent_instructions must be a string and needs subagents")

    harness = {**HARNESS_DEFAULTS, **(manifest.get("harness") or {}), **(xci.get("harness") or {})}
    nudges = int(harness.pop("max_nudges", 2))
    nudges = int(loop.get("max_nudges", nudges))
    return MetaAgentSpec(
        key=key, path=path, name=str(spec.get("name")), description=str(spec.get("description") or ""),
        instructions="\n\n".join(p for p in parts if p), model=model_id,
        provider=str(model.get("provider") or ""), purpose=str(xci.get("purpose") or entry.get("purpose") or key),
        tools=tuple(tools), terminal_tool=terminal, documents=tuple(xci.get("documents") or ()),
        skills_paths=skills, harness=harness, runtime=str(entry.get("runtime") or manifest.get("runtime") or "harness"),
        max_nudges=nudges, role=role, subagents=children,
        subagent_instructions=sub_instructions.strip() if isinstance(sub_instructions, str) else None,
        harness_root=tree, max_tool_calls=loop.get("max_tool_calls"), max_turns=loop.get("max_turns"),
        tool_descriptions={t: d for t, d in (descriptions or {}).items() if d})


def _tree_overlays(tree: Path, path: Path) -> tuple[dict[str, int], dict[str, str | None] | None]:
    """Agent ``path.stem``'s ``loops/loops.yaml`` knobs (clamped to the frozen caps) and its
    ``tools/tools.yaml`` entry (``None`` = all bound tools, own descriptions)."""
    from ci_lab.harness_tree import HarnessTree, HarnessTreeError

    t = HarnessTree(tree)
    try:
        return t.agent_loop(path.stem), t.agent_tools(path.stem)
    except HarnessTreeError as exc:
        raise SpecError(f"{path.name}: {exc}") from exc


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


class ToolBudgetMiddleware(FunctionMiddleware):
    """``max_tool_calls``: past the budget a tool call is not run and returns an ERROR asking for
    the final answer. The terminal ``submit_*`` tool is never counted or blocked."""

    def __init__(self, max_calls: int, terminal_tool: str = "") -> None:
        self.max_calls, self.terminal_tool, self.calls = max_calls, terminal_tool, 0

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        if not self.terminal_tool or getattr(context.function, "name", None) != self.terminal_tool:
            if self.calls >= self.max_calls:
                then = f"call {self.terminal_tool} now" if self.terminal_tool else "answer now"
                context.result = f"ERROR: tool-call budget ({self.max_calls}) exhausted; {then}"
                return
            self.calls += 1
        await call_next()


class TurnLimitMiddleware(ChatMiddleware):
    """``max_turns``: model calls per agent run (all nudges included). Past the limit the model
    is not called; a fixed assistant reply with no tool calls ends the turn instead (a streaming
    call fails closed with ``MiddlewareFailure``)."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns, self.turns, self.tripped = max_turns, 0, False

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        if self.turns >= self.max_turns:
            self.tripped = True
            text = f"Stopped: the turn limit (max_turns={self.max_turns}) is reached."
            if getattr(context, "stream", False):
                raise MiddlewareFailure(text)
            context.result = ChatResponse(messages=[Message("assistant", [text])], finish_reason="stop")
            return
        self.turns += 1
        await call_next()


def loop_limit_middleware(spec: MetaAgentSpec) -> list[Any]:
    """Fresh middleware enforcing ``spec.max_tool_calls``/``spec.max_turns`` for one run."""
    out: list[Any] = []
    if spec.max_tool_calls is not None:
        out.append(ToolBudgetMiddleware(spec.max_tool_calls, spec.terminal_tool))
    if spec.max_turns is not None:
        out.append(TurnLimitMiddleware(spec.max_turns))
    return out


def _described(fn: Any, name: str, description: str | None) -> Any:
    if not description:
        return fn
    from agent_framework import FunctionTool, tool

    if isinstance(fn, FunctionTool):
        out = copy.copy(fn)
        out.description = description
        return out
    return tool(fn, name=name, description=description)


class AgentBuilder(Protocol):
    """Builds a runnable agent (``.run(message, session=...)``, ``.create_session()``).

    ``subagent_bindings`` maps each subagent key (see :func:`subagent_specs`) to its own
    read-only tool bindings; it is only passed when the spec declares subagents.
    """

    def __call__(self, spec: MetaAgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
                 middleware: Sequence[Any] = (), loop_should_continue: Callable[..., Any] | None = None,
                 loop_next_message: Callable[..., Any] | None = None,
                 subagent_bindings: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None) -> Any: ...


def _build_subagents(spec: MetaAgentSpec, client: Any,
                     subagent_bindings: Mapping[str, Mapping[str, Callable[..., Any]]] | None) -> list[Any]:
    agents = []
    for sub in spec.subagents:
        if sub.role != "subagent" or sub.terminal_tool:
            raise SpecError(f"{sub.path.name}: not a subagent spec")
        if bad := [t for t in sub.tools if t not in SUBAGENT_TOOLS]:
            raise SpecError(f"{sub.path.name}: subagent tool(s) {', '.join(bad)} are not read-only")
        if subagent_bindings is None or sub.key not in subagent_bindings:
            raise SpecError(f"{spec.path.name}: no bindings for subagent {sub.key!r}")
        own = subagent_bindings[sub.key]
        agents.append(harness_builder(sub, client=client, bindings={t: own[t] for t in sub.tools if t in own},
                                      subagent_bindings=subagent_bindings))
    return agents


def harness_builder(spec: MetaAgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
                    middleware: Sequence[Any] = (), loop_should_continue: Callable[..., Any] | None = None,
                    loop_next_message: Callable[..., Any] | None = None,
                    subagent_bindings: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None) -> Any:
    """Local runtime=harness build: ``create_harness_agent`` with the spec's tools bound in order
    (``spec.tool_descriptions`` replace their descriptions) and :func:`loop_limit_middleware`
    appended to ``middleware``.

    Each subagent is built the same way on the same ``client`` with only its own read-only
    bindings and handed to MAF as ``background_agents`` (the parent gets MAF's
    ``background_agents_*`` tools to start/wait for/read delegated tasks).
    """
    from agent_framework import create_harness_agent

    from ci_lab.governance.maf import governed_harness_agent

    missing = [t for t in spec.tools if t not in bindings]
    if missing:
        raise SpecError(f"{spec.path.name}: no binding for tool(s) {', '.join(missing)}")
    params = inspect.signature(create_harness_agent).parameters
    kwargs: dict[str, Any] = {k: v for k, v in spec.harness.items() if k in params}
    if spec.skills_paths:
        kwargs["skills_paths"] = [str(p) for p in spec.skills_paths]
    if loop_should_continue is not None:
        kwargs.update(loop_should_continue=loop_should_continue, loop_next_message=loop_next_message,
                      loop_max_iterations=spec.max_nudges + 1)
    if spec.subagents:
        if "background_agents" not in params:  # pragma: no cover - MAF >= 1.19 has it
            raise SpecError(f"{spec.path.name}: this agent_framework has no background_agents support")
        kwargs["background_agents"] = _build_subagents(spec, client, subagent_bindings)
        if spec.subagent_instructions:
            kwargs["background_agents_instructions"] = spec.subagent_instructions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MAF harness APIs emit ExperimentalWarning
        policy = "harness" if spec.harness_root is not None else "meta_agents"
        return governed_harness_agent(client, name=spec.name, description=spec.description, policy=policy,
                                      agent_instructions=spec.instructions,
                                      tools=[_described(bindings[t], t, spec.tool_descriptions.get(t))
                                             for t in spec.tools],
                                      middleware=[*middleware, *loop_limit_middleware(spec)],
                                      governance={"agent_name": spec.name, "model": spec.model}, **kwargs)


def _model_override(path: Path, manifest: Mapping[str, Any]) -> str | None:
    """``CI_META_MODEL`` (see :mod:`ci_lab.maf.models`), checked against the allowlist."""
    from ci_lab.maf.models import META_MODEL_ENV, ModelEnvError, meta_model_override

    try:
        override = meta_model_override()
    except ModelEnvError as exc:
        raise SpecError(f"{path.name}: {exc}") from exc
    if override is not None and override not in (allowed := manifest_allowed_models(manifest)):
        raise SpecError(f"{path.name}: {META_MODEL_ENV}={override!r} is not in the allowlist {list(allowed)}; "
                        f"extend it with CI_ALLOWED_MODELS")
    return override


def manifest_allowed_models(manifest: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Model allowlist for the meta agents: manifest ``allowed_models`` (else its ``model``),
    plus the operator's ``CI_ALLOWED_MODELS`` extras (:mod:`ci_lab.maf.models`)."""
    from ci_lab.maf.models import ModelEnvError, with_extra_allowed

    manifest = manifest if manifest is not None else load_manifest()
    models = manifest.get("allowed_models") or ([manifest["model"]] if manifest.get("model") else [])
    try:
        return with_extra_allowed(str(m) for m in models)
    except ModelEnvError as exc:
        raise SpecError(str(exc)) from exc


def loader_builder(build_agent: Callable[..., Any], *,
                   allowed_models: Sequence[str] | None = None) -> AgentBuilder:
    """Adapt ``ci_lab.maf.loader.build_agent(spec_path, client=, bindings=, runtime="harness",
    allowed_models=, ...)``.

    Only keyword arguments the loader accepts are passed (it must honor ``x-ci``).
    ``allowed_models`` defaults to :func:`manifest_allowed_models` from the manifest.
    """
    params = inspect.signature(build_agent).parameters
    open_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    models = tuple(allowed_models) if allowed_models is not None else None

    def build(spec: MetaAgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
              middleware: Sequence[Any] = (), loop_should_continue: Callable[..., Any] | None = None,
              loop_next_message: Callable[..., Any] | None = None,
              subagent_bindings: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None) -> Any:
        if spec.subagents and not (open_kwargs or "subagent_bindings" in params):
            raise SpecError(f"{spec.path.name}: the loader cannot build subagents (no subagent_bindings)")
        # The loader re-reads the file, so it cannot apply CI_META_MODEL; refuse rather than record
        # a model other than the one the run selected.
        file_model = ((yaml.safe_load(spec.path.read_text(encoding="utf-8")) or {}).get("model") or {}).get("id")
        if file_model and str(file_model) != spec.model:
            raise SpecError(f"{spec.path.name}: the loader would build model {file_model!r}, not the selected "
                            f"{spec.model!r} (CI_META_MODEL); use validated_harness_builder")
        file_doc = yaml.safe_load(spec.path.read_text(encoding="utf-8")) or {}
        file_tools = [t.get("name") for t in file_doc.get("tools") or []]
        if spec.tool_descriptions or list(spec.tools) != file_tools:
            raise SpecError(f"{spec.path.name}: the loader cannot apply tools/tools.yaml; use validated_harness_builder")
        extra = {"middleware": [*middleware, *loop_limit_middleware(spec)], "loop_should_continue": loop_should_continue,
                 "loop_next_message": loop_next_message, "subagent_bindings": subagent_bindings,
                 "allowed_models": models if models is not None else manifest_allowed_models(),
                 "allowed_providers": (spec.provider,) if spec.provider else None}
        kwargs = {k: v for k, v in extra.items() if v is not None and (open_kwargs or k in params)}
        return build_agent(spec.path, client=client, bindings={t: bindings[t] for t in spec.tools},
                           runtime=spec.runtime, **kwargs)

    return build


def validated_harness_builder(*, allowed_models: Sequence[str] | None = None) -> AgentBuilder:
    """Validate the spec with ``ci_lab.maf.specs`` (model allowlist, providers, bindings,
    expressions), then build with :func:`harness_builder`.

    ``ci_lab.maf.loader.build_agent`` cannot build meta specs directly: its ``x-ci`` schema is
    strict (no ``terminal_tool``/``purpose``/``documents``) and its harness path has no
    terminal-submit nudge loop. A disallowed spec raises :class:`SpecError` (no fallback).
    """
    from ci_lab.maf.specs import SpecError as MafSpecError
    from ci_lab.maf.specs import parse_agent_spec

    def validate(spec: MetaAgentSpec, models: tuple[str, ...]) -> Any:
        doc = yaml.safe_load(spec.path.read_text(encoding="utf-8")) or {}
        doc.pop("x-ci", None)
        doc["instructions"] = spec.instructions
        # Validate and hash the tools that will actually be exposed (tools/tools.yaml overlay).
        doc["tools"] = [{**t, "description": spec.tool_descriptions.get(t.get("name"), t.get("description"))}
                        for t in doc.get("tools") or [] if t.get("name") in spec.tools]
        # Validate and hash the model that will actually run (spec default, or CI_META_MODEL).
        doc["model"] = {**(doc.get("model") or {}), "id": spec.model}
        try:
            return parse_agent_spec(doc, base_dir=spec.path.parent, allowed_models=models,
                                    allowed_providers=(spec.provider,) if spec.provider else (),
                                    allowed_bindings=spec.tools)
        except MafSpecError as exc:
            raise SpecError(f"{spec.path.name}: {exc}") from exc

    def build(spec: MetaAgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
              middleware: Sequence[Any] = (), loop_should_continue: Callable[..., Any] | None = None,
              loop_next_message: Callable[..., Any] | None = None,
              subagent_bindings: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None) -> Any:
        models = tuple(allowed_models) if allowed_models is not None else manifest_allowed_models()
        loaded = validate(spec, models)
        subs = {s.key: validate(s, models).spec.digest() for s in subagent_specs(spec)}
        agent = harness_builder(spec, client=client, bindings=bindings, middleware=middleware,
                                loop_should_continue=loop_should_continue, loop_next_message=loop_next_message,
                                subagent_bindings=subagent_bindings)
        props = getattr(agent, "additional_properties", None)
        if isinstance(props, dict):
            props["ci_lab"] = {"spec_digest": loaded.spec.digest(), "runtime": spec.runtime,
                               "model": loaded.spec.model.id, "provider": loaded.spec.model.provider}
            if subs:
                props["ci_lab"]["subagents"] = subs
        return agent

    return build


def default_builder(*, allowed_models: Sequence[str] | None = None) -> AgentBuilder:
    """:func:`validated_harness_builder` when ``ci_lab.maf`` is importable, else :func:`harness_builder`.

    ``allowed_models`` defaults to the manifest allowlist (:func:`manifest_allowed_models`).
    """
    try:
        import ci_lab.maf.specs  # noqa: F401
    except ImportError:  # pragma: no cover - maf is a hard dependency today
        return harness_builder
    return validated_harness_builder(allowed_models=allowed_models)
