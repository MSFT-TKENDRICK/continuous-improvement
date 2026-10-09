from __future__ import annotations

import asyncio
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from agent_framework import AgentMiddleware

from ci_lab.maf.loader import (
    ExperimentalWarning,
    build_agent,
    build_agents_from_manifest,
    chat_options,
    provenance,
)
from ci_lab.maf.specs import SpecError, load_agent_spec
from ci_lab.testing import Call, FakeChatClient

MODELS = ["gpt-5-mini"]
SKILL = """\
---
name: file-reading
description: How to inspect files safely.
---
# File reading
Always inspect before editing.
"""


def _instructions(client: FakeChatClient, i: int = 0) -> str:
    return str(client.requests[i][1].get("instructions"))


def _tool_names(client: FakeChatClient, i: int = 0) -> list[str]:
    return [t.name for t in client.requests[i][1].get("tools", [])]


def _tool_loop_client() -> FakeChatClient:
    return FakeChatClient([[Call("read_file", {"resource_id": "42"})], "Resource 42 is ready."])


@pytest.mark.parametrize("runtime", ["prompt", "harness"])
def test_runtime_runs_tool_loop(runtime: str, agent_yaml: Path, read_file, read_calls) -> None:
    client = _tool_loop_client()
    agent = build_agent(agent_yaml, client=client, bindings={"read_file": read_file}, runtime=runtime,
                        allowed_models=MODELS)
    result = asyncio.run(agent.run("Read resource 42."))
    assert result.text == "Resource 42 is ready."
    assert read_calls == ["42"]
    assert len(client.requests) == 2
    assert "read_file" in _tool_names(client)
    assert "You are the harness student agent." in _instructions(client)
    opts = client.requests[0][1]
    assert opts["max_tokens"] == 256
    assert opts["additional_chat_options"] == {"reasoningEffort": "low"}
    tool_msgs = [m for m in client.requests[1][0] if m.role == "tool"]
    assert any("resource 42: ready" in str(c.result) for m in tool_msgs for c in m.contents)
    meta = agent.additional_properties["ci_lab"]
    assert meta["runtime"] == runtime and meta["model"] == "gpt-5-mini" and meta["provider"] == "GitHubCopilot"
    assert meta["spec_digest"] == load_agent_spec(agent_yaml, allowed_models=MODELS).spec.digest()


def test_prompt_runtime_uses_injected_client(agent_yaml: Path, read_file) -> None:
    client = FakeChatClient()
    agent = build_agent(agent_yaml, client=client, bindings={"read_file": read_file}, allowed_models=MODELS)
    assert agent.client is client


def test_harness_has_no_file_memory_or_web_search(agent_yaml: Path, read_file) -> None:
    client = FakeChatClient(["hi"])
    agent = build_agent(agent_yaml, client=client, bindings={"read_file": read_file}, runtime="harness",
                        allowed_models=MODELS)
    asyncio.run(agent.run("hi"))
    names = _tool_names(client)
    assert not any("memory" in n or "web" in n for n in names), names


def test_harness_memory_dir(agent_yaml: Path, read_file, tmp_path: Path) -> None:
    client = FakeChatClient(["hi"])
    mem = tmp_path / "mem"
    agent = build_agent(agent_yaml, client=client, bindings={"read_file": read_file}, runtime="harness",
                        allowed_models=MODELS, memory_dir=mem)
    asyncio.run(agent.run("hi"))
    assert mem.is_dir()


@pytest.mark.parametrize("runtime", ["prompt", "harness"])
def test_extra_instructions_and_skills(runtime: str, agent_yaml: Path, write, read_file) -> None:
    write("agents/skills/file-reading/SKILL.md", SKILL)
    client = FakeChatClient(["hi"])
    agent = build_agent(agent_yaml, client=client, bindings={"read_file": read_file}, runtime=runtime,
                        allowed_models=MODELS, skills_paths=["skills"], extra_instructions="Rollout 7.")
    asyncio.run(agent.run("hi"))
    instructions = _instructions(client)
    assert "Rollout 7." in instructions
    assert "file-reading" in instructions
    assert "load_skill" in _tool_names(client)


@pytest.mark.parametrize("path", ["../outside", "missing", "harness_agent.yaml"])
def test_bad_skills_paths(path: str, agent_yaml: Path, write, read_file) -> None:
    write("outside/SKILL.md", SKILL)
    with pytest.raises(SpecError):
        build_agent(agent_yaml, client=FakeChatClient(), bindings={"read_file": read_file},
                    allowed_models=MODELS, skills_paths=[path])


@pytest.mark.parametrize("runtime", ["prompt", "harness"])
def test_middleware_applied(runtime: str, agent_yaml: Path, read_file) -> None:
    seen: list[str] = []

    class Spy(AgentMiddleware):
        async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
            seen.append(context.agent.name)
            await call_next()

    agent = build_agent(agent_yaml, client=FakeChatClient(["hi"]), bindings={"read_file": read_file},
                        runtime=runtime, allowed_models=MODELS, middleware=[Spy()])
    asyncio.run(agent.run("hi"))
    assert seen == ["CiStudent"]


def test_bindings_are_the_allowlist(agent_yaml: Path, read_file) -> None:
    with pytest.raises(SpecError, match="bindings not allowed"):
        build_agent(agent_yaml, client=FakeChatClient(), bindings={"other": read_file}, allowed_models=MODELS)


def test_unknown_runtime(agent_yaml: Path, read_file) -> None:
    with pytest.raises(SpecError, match="runtime"):
        build_agent(agent_yaml, client=FakeChatClient(), bindings={"read_file": read_file},
                    runtime="shell", allowed_models=MODELS)  # type: ignore[arg-type]


def test_no_experimental_warning_leaks(agent_yaml: Path, read_file) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for runtime in ("prompt", "harness"):
            build_agent(agent_yaml, client=FakeChatClient(), bindings={"read_file": read_file},
                        runtime=runtime, allowed_models=MODELS)
    assert not [w for w in caught if issubclass(w.category, ExperimentalWarning)]


def test_provenance(agent_yaml: Path, read_file) -> None:
    for runtime in ("prompt", "harness"):
        build_agent(agent_yaml, client=FakeChatClient(), bindings={"read_file": read_file},
                    runtime=runtime, allowed_models=MODELS)
    p = provenance()
    assert p["versions"]["agent-framework-core"]
    assert p["versions"]["agent-framework-declarative"]
    assert {"DECLARATIVE_AGENTS", "HARNESS"} <= set(p["experimental_features"])
    assert p["powerfx_installed"] is False


def test_chat_options_output_schema(tmp_path: Path, agent_text: str, write) -> None:
    text = agent_text + "outputSchema:\n  properties:\n    verdict:\n      kind: string\n      required: true\n"
    spec = load_agent_spec(write("agents/s.yaml", text), allowed_models=MODELS).spec
    opts = chat_options(spec)
    assert opts["max_tokens"] == 256
    assert opts["response_format"]["properties"]["verdict"]["type"] == "string"


# ---------------------------------------------------------------- manifest

JUDGE_YAML = """\
kind: Prompt
name: Judge
description: Grades answers.
instructions: Grade it.
model: {id: gpt-5, provider: GitHubCopilot}
"""


@pytest.fixture
def manifest(write, agent_yaml: Path) -> Path:
    write("agents/judge.yaml", JUDGE_YAML)
    write("agents/skills/file-reading/SKILL.md", SKILL)
    return write("agents/manifest.yaml", """\
        agents:
          CiStudent:
            spec: harness_agent.yaml
            runtime: harness
            purpose: target
            bindings: [read_file]
            skills_paths: [skills]
          Judge:
            spec: judge.yaml
            purpose: judge
        """)


class RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.clients: dict[str, FakeChatClient] = {}

    def __call__(self, *, profile: Any, model: str, purpose: str, rollout: Any = None) -> FakeChatClient:
        self.calls.append({"profile": profile, "model": model, "purpose": purpose, "rollout": rollout})
        client = FakeChatClient([[Call("read_file", {"resource_id": "7"})], "done"] if purpose == "target" else [],
                                model=model)
        self.clients[purpose] = client
        return client


def test_build_agents_from_manifest(manifest: Path, read_file, read_calls) -> None:
    from ci_lab.contracts import Profile

    factory = RecordingFactory()
    profile = next(iter(Profile))
    agents = build_agents_from_manifest(manifest, client_factory=factory, profile=profile,
                                        bindings={"read_file": read_file, "unused": print},
                                        allowed_models=["gpt-5-mini", "gpt-5"])
    assert set(agents) == {"CiStudent", "Judge"}
    assert sorted((c["purpose"], c["model"]) for c in factory.calls) == [("judge", "gpt-5"),
                                                                         ("target", "gpt-5-mini")]
    assert all(c["profile"] is profile for c in factory.calls)
    assert agents["CiStudent"].additional_properties["ci_lab"]["runtime"] == "harness"
    assert agents["Judge"].additional_properties["ci_lab"]["runtime"] == "prompt"
    assert asyncio.run(agents["CiStudent"].run("where is 7")).text == "done"
    assert read_calls == ["7"]
    assert "file-reading" in _instructions(factory.clients["target"])

    only = build_agents_from_manifest(manifest, client_factory=RecordingFactory(), profile=profile,
                                      bindings={"read_file": read_file},
                                      allowed_models=["gpt-5-mini", "gpt-5"], only={"Judge"})
    assert set(only) == {"Judge"}


def test_manifest_bindings_are_scoped(write, manifest: Path, read_file) -> None:
    from ci_lab.contracts import Profile

    # Judge's manifest entry grants no bindings, so a tool in its spec must be rejected.
    write("agents/judge.yaml", JUDGE_YAML + "tools:\n  - {kind: function, name: read_file, "
          "description: d, bindings: [{name: read_file}]}\n")
    with pytest.raises(SpecError, match="bindings not allowed"):
        build_agents_from_manifest(manifest, client_factory=RecordingFactory(), profile=next(iter(Profile)),
                                   bindings={"read_file": read_file}, allowed_models=["gpt-5-mini", "gpt-5"])


@pytest.mark.parametrize("kw, match", [
    ({"bindings": {}}, "not provided"),
    ({"allowed_models": ["gpt-5-mini"]}, "allowlist"),
    ({"only": {"Nobody"}}, "not in manifest"),
])
def test_manifest_build_errors(manifest: Path, read_file, kw: dict, match: str) -> None:
    from ci_lab.contracts import Profile

    args: dict[str, Any] = {"bindings": {"read_file": read_file}, "allowed_models": ["gpt-5-mini", "gpt-5"]}
    args.update(kw)
    with pytest.raises(SpecError, match=match):
        build_agents_from_manifest(manifest, client_factory=RecordingFactory(), profile=next(iter(Profile)), **args)


def test_manifest_name_must_match_spec(write, manifest: Path, read_file) -> None:
    from ci_lab.contracts import Profile

    write("agents/judge.yaml", JUDGE_YAML.replace("name: Judge", "name: Grader"))
    with pytest.raises(SpecError, match="does not match"):
        build_agents_from_manifest(manifest, client_factory=RecordingFactory(), profile=next(iter(Profile)),
                                   bindings={"read_file": read_file}, allowed_models=["gpt-5-mini", "gpt-5"])


def test_manifest_frozen_schemas_per_agent(manifest: Path, read_file, agent_yaml: Path) -> None:
    from ci_lab.contracts import Profile

    schemas = load_agent_spec(agent_yaml, allowed_models=MODELS).spec.tool_schemas()
    schemas["read_file"] = {**schemas["read_file"], "required": []}
    with pytest.raises(SpecError, match="frozen"):
        build_agents_from_manifest(manifest, client_factory=RecordingFactory(), profile=next(iter(Profile)),
                                   bindings={"read_file": read_file}, allowed_models=["gpt-5-mini", "gpt-5"],
                                   frozen_tool_schemas={"CiStudent": schemas})
