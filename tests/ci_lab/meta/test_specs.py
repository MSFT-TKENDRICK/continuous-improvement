from __future__ import annotations

import shutil
import warnings

import pytest
import yaml

from ci_lab.harness_tree import repo_harness_dir
from ci_lab.meta.spec_loader import (
    AGENTS,
    SPECS_DIR,
    SpecError,
    default_builder,
    load_manifest,
    load_spec,
    loader_builder,
    manifest_allowed_models,
    subagent_specs,
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
    assert spec.provider == "GitHubCopilot" and spec.model == "claude-sonnet-5" and spec.runtime == "harness"
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
    skills, own = spec.skills_paths
    skill_md = skills / "agent-lightning" / "SKILL.md"
    assert skill_md.is_file() and skill_md.read_text(encoding="utf-8").startswith("---")
    assert own == repo_harness_dir().resolve() / "skills" and (own / "harness-editing" / "SKILL.md").is_file()
    assert all(not load_spec(k).skills_paths for k in ("analyst", "critic", "reflector"))


@pytest.fixture
def spec_copy(tmp_path):
    shutil.copytree(repo_harness_dir(), tmp_path / "harness")
    return tmp_path / "harness" / "agents"


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
    (lambda d: d["x-ci"].update(instructions_files=["../../../pyproject.toml"]), "not found under the harness dir"),
    (lambda d: d["x-ci"].update(skills_paths=["nope/missing"]), "does not exist"),
    (lambda d: d.update(kind="Workflow"), "kind|not a valid declarative"),
])
def test_spec_validation_errors(spec_copy, mutation, message):
    path = _mutate(spec_copy / "proposer.yaml", mutation)
    with pytest.raises(SpecError, match=message):
        load_spec(path)


def test_terminal_tool_must_agree_with_manifest(tmp_path):
    shutil.copytree(SPECS_DIR, tmp_path / "specs")
    path = _mutate(tmp_path / "specs" / "critic.yaml", lambda d: d["x-ci"].update(terminal_tool="submit_analysis"))
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


def test_loader_builder_passes_allowed_models_when_accepted():
    seen = {}

    def build_agent(spec_path, *, client, bindings, runtime="prompt", allowed_models, allowed_providers=()):
        seen.update(allowed_models=allowed_models, allowed_providers=allowed_providers)
        return "agent"

    spec = load_spec("critic")
    bindings = {t: (lambda: "x") for t in spec.tools}
    loader_builder(build_agent)(spec, client="c", bindings=bindings)
    assert seen == {"allowed_models": manifest_allowed_models(), "allowed_providers": ("GitHubCopilot",)}
    loader_builder(build_agent, allowed_models=["m1"])(spec, client="c", bindings=bindings)
    assert seen["allowed_models"] == ("m1",)


def test_manifest_allowlist_covers_every_spec():
    allowed = manifest_allowed_models()
    assert allowed and all(load_spec(k).model in allowed for k in AGENTS)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_default_builder_validates_via_maf_without_warnings(key):
    from ci_lab.testing import FakeChatClient

    spec = load_spec(key)
    bindings = {t: (lambda: "x") for t in spec.tools}
    subs = {s.key: {t: (lambda: "x") for t in s.tools} for s in subagent_specs(spec)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        agent = default_builder()(spec, client=FakeChatClient(), bindings=bindings,
                                  loop_should_continue=lambda **_: False, loop_next_message=lambda **_: "x",
                                  **({"subagent_bindings": subs} if subs else {}))
    assert agent.additional_properties["ci_lab"]["model"] == spec.model
    assert len(agent.additional_properties["ci_lab"]["spec_digest"]) == 64
    assert set(agent.additional_properties["ci_lab"].get("subagents", {})) == set(subs)


def test_default_builder_rejects_models_outside_the_allowlist():
    from ci_lab.testing import FakeChatClient

    spec = load_spec("critic")
    with pytest.raises(SpecError, match="allowlist"):
        default_builder(allowed_models=["gpt-5-mini"])(spec, client=FakeChatClient(),
                                                       bindings={t: (lambda: "x") for t in spec.tools})


def test_evolvable_specs_load_from_the_given_harness_dir(spec_copy):
    root = spec_copy.parent
    prompt = root / "prompts" / "analyst.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\nCANDIDATE-MARKER\n", encoding="utf-8")
    spec = load_spec("analyst", harness_dir=root)
    assert spec.path == (spec_copy / "analyst.yaml").resolve() and spec.harness_root == root.resolve()
    assert "CANDIDATE-MARKER" in spec.instructions
    assert "CANDIDATE-MARKER" not in load_spec("analyst").instructions
    assert load_spec("analyst").harness_root == repo_harness_dir().resolve()
    assert load_spec("student", harness_dir=root).name == "CiStudent"
    assert load_spec("proposer", harness_dir=root).subagents[0].path.parent == spec_copy.resolve()
    critic = load_spec("critic", harness_dir=root)
    assert critic.path == (SPECS_DIR / "critic.yaml").resolve() and critic.harness_root is None
    assert critic.instructions == load_spec("critic").instructions


def test_missing_evolvable_spec_fails_closed(spec_copy):
    (spec_copy / "reflector.yaml").unlink()
    with pytest.raises(SpecError, match="not found"):
        load_spec("reflector", harness_dir=spec_copy.parent)
