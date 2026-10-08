"""Build MAF agents from validated declarative specs.

Every ci_lab agent goes through :func:`build_agent`:

* ``runtime="prompt"`` -> ``agent_framework_declarative.AgentFactory(client=..., bindings=...,
  safe_mode=True).create_agent_from_dict(...)``. The pre-processed spec (``x-ci`` stripped,
  instructions composed) is passed as a dict, and model ``id``/``provider`` are dropped from
  it so the factory uses the injected ``client`` rather than constructing its own.
* ``runtime="harness"`` -> ``agent_framework.create_harness_agent(client, ...)`` with the
  same instructions, tools (built from the YAML tool specs + bindings) and options;
  file memory disabled unless ``memory_dir`` is given; no web search; skills (data only,
  no scripts) via ``skills_paths``.

MAF ``ExperimentalWarning`` s are suppressed and recorded; :func:`provenance` reports them
together with the installed MAF package versions.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
import threading
import warnings
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_framework import Agent, FileSystemAgentFileStore, SkillsProvider, create_harness_agent
from agent_framework import FunctionTool as AFFunctionTool

from ci_lab.contracts import ChatClientFactory, Profile, RolloutKey
from ci_lab.maf.specs import (
    DEFAULT_ALLOWED_OPTIONS,
    AgentSpec,
    LoadedSpec,
    Runtime,
    SpecError,
    load_agent_spec,
    load_manifest,
    resolve_contained,
)

try:  # private module; fall back to the public base class if it moves
    from agent_framework._feature_stage import ExperimentalWarning
except ImportError:  # pragma: no cover
    ExperimentalWarning = FutureWarning  # type: ignore[misc,assignment]

MAF_DISTRIBUTIONS = ("agent-framework-core", "agent-framework-declarative", "agent-framework-github-copilot",
                     "agent-framework-openai", "github-copilot-sdk")

_lock = threading.Lock()
_experimental: dict[str, str] = {}  # feature tag -> first warning message seen (or "")

# Option name in declarative YAML -> MAF chat option (mirrors AgentFactory._parse_chat_options).
_OPTION_MAP = {
    "frequencyPenalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "maxOutputTokens": "max_tokens",
    "temperature": "temperature",
    "topP": "top_p",
    "seed": "seed",
    "stopSequences": "stop",
    "allowMultipleToolCalls": "allow_multiple_tool_calls",
    "chatToolMode": "tool_choice",
}


@contextmanager
def experimental_features(tag: str) -> Iterator[None]:
    """Suppress MAF ``ExperimentalWarning`` inside the block and record ``tag`` (+ messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ExperimentalWarning)
        yield
    with _lock:
        _experimental.setdefault(tag, "")
        for w in caught:
            if issubclass(w.category, ExperimentalWarning):
                msg = str(w.message)
                key = msg.split("]", 1)[0].lstrip("[") if msg.startswith("[") else tag
                _experimental.setdefault(key, msg)
            else:  # re-emit anything we did not mean to swallow
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def provenance() -> dict[str, Any]:
    """MAF package versions, experimental features used so far, and the no-.NET invariant."""
    versions: dict[str, str | None] = {}
    for dist in MAF_DISTRIBUTIONS:
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    with _lock:
        experimental = dict(sorted(_experimental.items()))
    return {"versions": versions, "experimental_features": experimental,
            "powerfx_installed": importlib.util.find_spec("powerfx") is not None}


# ---------------------------------------------------------------- agents

def chat_options(spec: AgentSpec) -> dict[str, Any]:
    """MAF ``default_options`` for a spec's ``model.options`` (+ ``outputSchema``)."""
    options: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in spec.model.options.items():
        if value is None:
            continue
        if key in _OPTION_MAP:
            options[_OPTION_MAP[key]] = value
        else:
            extra[key] = value
    if extra:
        options["additional_chat_options"] = extra
    if spec.outputSchema:
        from agent_framework_declarative._models import PropertySchema

        options["response_format"] = PropertySchema.from_dict(dict(spec.outputSchema)).to_json_schema()
    return options


def build_tools(spec: AgentSpec, bindings: Mapping[str, Callable[..., Any]]) -> list[AFFunctionTool]:
    """MAF function tools for the spec's tools, bound to ``bindings`` (first listed match)."""
    tools = []
    for tool in spec.tools:
        func = next((bindings[b.name] for b in tool.bindings if b.name in bindings), None)
        if func is None:
            raise SpecError(f"tool {tool.name!r}: none of its bindings {[b.name for b in tool.bindings]} provided")
        tools.append(AFFunctionTool(name=tool.name, description=tool.description,
                                    input_model=tool.json_schema(), func=func))
    return tools


def _skills_provider(paths: Sequence[Path]) -> SkillsProvider:
    # Skills are data-only: no scripts; loading/reading needs no approval round-trip.
    return SkillsProvider.from_paths(list(paths), script_extensions=(), disable_load_skill_approval=True,
                                     disable_read_skill_resource_approval=True)


def build_agent(spec_path: str | Path, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
                runtime: Runtime = "prompt", allowed_models: Collection[str],
                allowed_providers: Collection[str] = (), frozen_tool_schemas: Mapping[str, Any] | None = None,
                allowed_options: Collection[str] = DEFAULT_ALLOWED_OPTIONS,
                skills_paths: Sequence[str | Path] = (), extra_instructions: str = "",
                middleware: Sequence[Any] = (), memory_dir: str | Path | None = None) -> Agent:
    """Validate ``spec_path`` and build a MAF ``Agent`` on ``client``.

    ``bindings`` is both the implementation map and the allowed binding set: every tool
    binding in the spec must name a key of ``bindings``. ``skills_paths`` must be
    existing directories (absolute, or relative to the spec's directory and contained)."""
    loaded = load_agent_spec(spec_path, allowed_models=allowed_models, allowed_providers=allowed_providers,
                             allowed_bindings=set(bindings), frozen_tool_schemas=frozen_tool_schemas,
                             allowed_options=allowed_options, extra_instructions=extra_instructions)
    skills = [_skill_dir(Path(spec_path).parent, p) for p in skills_paths]
    return build_agent_from_spec(loaded, client=client, bindings=bindings, runtime=runtime, skills_paths=skills,
                                 middleware=middleware, memory_dir=memory_dir)


def build_agent_from_spec(loaded: LoadedSpec | AgentSpec, *, client: Any,
                          bindings: Mapping[str, Callable[..., Any]], runtime: Runtime = "prompt",
                          skills_paths: Sequence[Path] = (), middleware: Sequence[Any] = (),
                          memory_dir: str | Path | None = None) -> Agent:
    """Build from an already validated spec (see :func:`ci_lab.maf.specs.load_agent_spec`)."""
    spec = loaded.spec if isinstance(loaded, LoadedSpec) else loaded
    for tool in spec.tools:
        if not any(b.name in bindings for b in tool.bindings):
            raise SpecError(f"tool {tool.name!r}: none of its bindings {[b.name for b in tool.bindings]} provided")
    if runtime == "prompt":
        agent = _build_prompt_agent(spec, client=client, bindings=bindings)
        if skills_paths:
            agent.context_providers.append(_skills_provider(skills_paths))
        if middleware:
            agent.middleware = [*(agent.middleware or []), *middleware]
    elif runtime == "harness":
        agent = _build_harness_agent(spec, client=client, bindings=bindings, skills_paths=skills_paths,
                                     middleware=middleware, memory_dir=memory_dir)
    else:
        raise SpecError(f"unknown runtime {runtime!r}")
    agent.additional_properties["ci_lab"] = {"spec_digest": spec.digest(), "runtime": runtime,
                                             "model": spec.model.id, "provider": spec.model.provider}
    return agent


def _build_prompt_agent(spec: AgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]]) -> Agent:
    with experimental_features("DECLARATIVE_AGENTS"):
        from agent_framework_declarative import AgentFactory

        # env_file_path=os.devnull: never let python-dotenv pull a stray .env into os.environ.
        factory = AgentFactory(client=client, bindings=dict(bindings), safe_mode=True, env_file_path=os.devnull)
        agent = factory.create_agent_from_dict(spec.declarative_dict())
    if agent.client is not client:  # pragma: no cover - guards a MAF behaviour change
        raise SpecError("AgentFactory did not use the injected client")
    return agent


def _build_harness_agent(spec: AgentSpec, *, client: Any, bindings: Mapping[str, Callable[..., Any]],
                         skills_paths: Sequence[Path], middleware: Sequence[Any],
                         memory_dir: str | Path | None) -> Agent:
    memory_store = None
    with experimental_features("HARNESS"):
        if memory_dir is not None:
            Path(memory_dir).mkdir(parents=True, exist_ok=True)
            memory_store = FileSystemAgentFileStore(Path(memory_dir))
        return create_harness_agent(
            client,
            name=spec.name,
            description=spec.description,
            agent_instructions=spec.instructions,
            tools=build_tools(spec, bindings) or None,
            disable_file_memory=memory_store is None,
            file_memory_store=memory_store,
            skills_provider=_skills_provider(skills_paths) if skills_paths else None,
            disable_web_search=True,
            # Our tools never require approval; the approval middleware would also demand
            # an AgentSession on every run (breaks plain runs and workflow invocations).
            disable_tool_auto_approval=True,
            middleware=list(middleware) or None,
            default_options=chat_options(spec),
        )


def _skill_dir(base: Path, path: str | Path) -> Path:
    p = Path(path)
    resolved = p.resolve() if p.is_absolute() else resolve_contained(base, p)
    if p.is_absolute() and (p.is_symlink() or not resolved.exists()):
        raise SpecError(f"skills path must be an existing, non-symlink directory: {p}")
    if not resolved.is_dir():
        raise SpecError(f"skills path is not a directory: {p}")
    return resolved


def build_agents_from_manifest(manifest_path: str | Path, *, client_factory: ChatClientFactory, profile: Profile,
                               bindings: Mapping[str, Callable[..., Any]], allowed_models: Collection[str],
                               allowed_providers: Collection[str] = (),
                               frozen_tool_schemas: Mapping[str, Mapping[str, Any]] | None = None,
                               allowed_options: Collection[str] = DEFAULT_ALLOWED_OPTIONS,
                               middleware: Sequence[Any] = (), rollout: RolloutKey | None = None,
                               memory_root: str | Path | None = None,
                               only: Collection[str] | None = None) -> dict[str, Agent]:
    """Build every manifest agent (or ``only`` those). Each agent sees just the bindings
    its manifest entry lists; its client comes from ``client_factory(profile=..., model=
    spec.model.id, purpose=entry.purpose, rollout=...)``. ``frozen_tool_schemas`` is keyed
    by agent name. With ``memory_root``, harness agents get ``memory_root/<name>``."""
    manifest = load_manifest(manifest_path)
    agents: dict[str, Agent] = {}
    for name, resolved in manifest.entries.items():
        if only is not None and name not in only:
            continue
        entry = resolved.entry
        if missing := set(entry.bindings) - set(bindings):
            raise SpecError(f"agent {name}: manifest bindings not provided: {sorted(missing)}")
        scoped = {b: bindings[b] for b in entry.bindings}
        loaded = load_agent_spec(resolved.spec_path, allowed_models=allowed_models,
                                 allowed_providers=allowed_providers, allowed_bindings=set(scoped),
                                 frozen_tool_schemas=(frozen_tool_schemas or {}).get(name),
                                 allowed_options=allowed_options)
        if loaded.spec.name != name:
            raise SpecError(f"manifest key {name!r} does not match spec name {loaded.spec.name!r}")
        client = client_factory(profile=profile, model=loaded.spec.model.id, purpose=entry.purpose, rollout=rollout)
        memory_dir = Path(memory_root) / name if (memory_root is not None and entry.runtime == "harness") else None
        agents[name] = build_agent_from_spec(loaded, client=client, bindings=scoped, runtime=entry.runtime,
                                             skills_paths=resolved.skills_paths, middleware=middleware,
                                             memory_dir=memory_dir)
    if only is not None and (unknown := set(only) - set(agents)):
        raise SpecError(f"agents not in manifest: {sorted(unknown)}")
    return agents
