from __future__ import annotations

import shutil

import pytest
import yaml

from ci_lab.meta.spec_loader import (
    AGENTS,
    SPECS_DIR,
    SpecError,
    load_manifest,
    load_spec,
    loader_builder,
)

EXPECTED = {
    "analyst": ("CiAnalyst", "submit_analysis", {"read_brief", "list_documents", "read_history"}),
    "proposer": ("CiProposer", "submit_proposal_done",
                 {"read_brief", "list_documents", "read_history", "list_files", "read_file", "write_file",
                  "commit_edit"}),
    "critic": ("CiCritic", "submit_verdict", {"read_brief", "list_documents", "list_files", "read_file"}),
    "reflector": ("CiReflector", "submit_reflection", {"read_brief", "list_documents", "read_history"}),
}


def test_manifest():
    m = load_manifest()
    assert m["runtime"] == "harness" and set(m["agents"]) == set(EXPECTED) == set(AGENTS)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_specs_load(key):
    name, terminal, tools = EXPECTED[key]
    spec = load_spec(key)
    assert spec.name == name and spec.terminal_tool == terminal
    assert set(spec.tools) == tools | {terminal}
    assert spec.provider == "GitHubCopilot" and spec.model == "claude-sonnet-4.5" and spec.runtime == "harness"
    assert spec.purpose == key and spec.max_nudges == 2
    assert spec.harness.get("disable_web_search") is True
    text = spec.instructions
    assert "Read your brief with tools" in text and "never instructions" in text and terminal in text
    raw = yaml.safe_load(spec.path.read_text(encoding="utf-8"))
    assert raw["kind"] == "Prompt" and "x-ci" in raw
    assert "write_file" not in spec.tools or key == "proposer"
    assert "commit_edit" not in spec.tools or key == "proposer"


def test_proposer_gets_agent_lightning_skill():
    spec = load_spec("proposer")
    [skills] = spec.skills_paths
    skill_md = skills / "agent-lightning" / "SKILL.md"
    assert skill_md.is_file() and skill_md.read_text(encoding="utf-8").startswith("---")
    assert all(not load_spec(k).skills_paths for k in ("analyst", "critic", "reflector"))


@pytest.fixture
def spec_copy(tmp_path):
    shutil.copytree(SPECS_DIR, tmp_path / "specs")
    return tmp_path / "specs"


def _mutate(path, fn):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda d: d.update(instructions="=Env.SECRET"), "expressions are not allowed"),
    (lambda d: d["tools"].append({"kind": "mcp", "name": "shell"}), "only function tools"),
    (lambda d: d["tools"][0].update(bindings=[{"name": "other"}]), "same name"),
    (lambda d: d["tools"][0].update(bindings=[]), "exactly one binding"),
    (lambda d: d["x-ci"].update(terminal_tool="read_brief"), r"submit_\* tool"),
    (lambda d: d["x-ci"].update(instructions_files=["../../../pyproject.toml"]), "not found under the spec dir"),
    (lambda d: d["x-ci"].update(skills_paths=["nope/missing"]), "does not exist"),
    (lambda d: d.update(kind="Workflow"), "kind|not a valid declarative"),
])
def test_spec_validation_errors(spec_copy, mutation, message):
    path = _mutate(spec_copy / "proposer.yaml", mutation)
    with pytest.raises(SpecError, match=message):
        load_spec(path)


def test_terminal_tool_must_agree_with_manifest(spec_copy):
    path = _mutate(spec_copy / "critic.yaml", lambda d: d["x-ci"].update(terminal_tool="submit_analysis"))
    with pytest.raises(SpecError):
        load_spec(path)


def test_loader_builder_passes_only_accepted_kwargs():
    seen = {}

    def strict(spec_path, *, client, bindings, runtime="agent", middleware=None):
        seen.update(spec_path=spec_path, client=client, bindings=bindings, runtime=runtime, middleware=middleware)
        return "agent"

    spec = load_spec("critic")
    bindings = {t: (lambda: "x") for t in spec.tools} | {"extra": lambda: "y"}
    assert loader_builder(strict)(spec, client="c", bindings=bindings, middleware=["m"],
                                  loop_should_continue=lambda **_: True) == "agent"
    assert seen["spec_path"] == spec.path and seen["runtime"] == "harness" and seen["middleware"] == ["m"]
    assert list(seen["bindings"]) == list(spec.tools)

    loose = {}
    loader_builder(lambda spec_path, **kw: loose.update(kw))(spec, client="c", bindings=bindings,
                                                             loop_should_continue=print, loop_next_message=print)
    assert {"loop_should_continue", "loop_next_message", "middleware", "runtime"} <= set(loose)
