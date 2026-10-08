"""The declarative harness: agent.yaml + instruction files -> MAF agent with the frozen tools."""

from __future__ import annotations

import shutil

import pytest
import yaml

from ci_lab.testing import FakeChatClient
from order_support import agent, data, tools


def test_spec_strips_x_ci_and_composes_instructions():
    spec = agent._load_spec()
    assert agent.X_CI_KEY not in spec
    assert spec["kind"] == "Prompt" and spec["name"] == "OrderSupport"
    assert spec["model"] == {"id": agent.spec_model_alias(spec), "provider": "GitHubCopilot"}
    text = spec["instructions"]
    assert text.startswith(data.load_policy())
    assert text.count(data.load_policy()) == 1
    assert "## Learned preferences" in text
    assert "name: order-support" not in text  # SKILL.md front matter dropped
    assert text.endswith("\n\n" + agent.DATE_LINE)
    assert agent.instructions() == text


def test_system_prompt_file_is_the_frozen_policy():
    system = (agent.HARNESS_DIR / "prompts" / "system.md").read_text(encoding="utf-8")
    assert system.strip() == data.load_policy()
    assert agent.SYSTEM_PROMPT == data.load_policy() + "\n" + agent.DATE_LINE


def test_skill_has_learned_preferences_section():
    skill = (agent.HARNESS_DIR / "skills" / "order-support" / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\nname: order-support\n")
    assert "\n## Learned preferences" in skill


def test_agent_tools_match_frozen_tool_schemas():
    built = agent.build_agent(FakeChatClient())
    by_name = {t.name: t for t in built.default_options["tools"]}
    assert list(by_name) == [s["function"]["name"] for s in tools.TOOL_SCHEMAS]
    for schema in tools.TOOL_SCHEMAS:
        fn = schema["function"]
        assert by_name[fn["name"]].description == fn["description"]
        assert by_name[fn["name"]].parameters() == fn["parameters"]
    assert built.default_options["instructions"] == agent.instructions()
    assert built.name == "OrderSupport"


def test_verify_identity_is_declared_and_instructed():
    raw = yaml.safe_load((agent.HARNESS_DIR / "agent.yaml").read_text(encoding="utf-8"))
    decl = next(t for t in raw["tools"] if t["name"] == "verify_identity")
    assert decl["bindings"] == [{"name": "verify_identity"}]
    assert raw["x-ci"]["instructions_files"] == ["prompts/system.md", "prompts/identity.md",
                                                 "skills/order-support/SKILL.md"]
    text = agent.instructions()
    assert "call verify_identity" in text
    assert text.index("verify_identity") > text.index(data.load_policy()) + len(data.load_policy()) - 1


def test_model_routing_stays_out_of_the_factory():
    spec = agent._load_spec()
    factory_spec = agent._factory_spec(spec)
    assert "model" not in factory_spec
    assert spec["model"]["provider"] == "GitHubCopilot"  # caller's spec untouched
    client = FakeChatClient()
    assert agent.build_agent(client, spec).client is client


def _copy_harness(tmp_path):
    root = tmp_path / "harness"
    shutil.copytree(agent.HARNESS_DIR, root)
    return root


def test_harness_dir_env_override(tmp_path, monkeypatch, use_client):
    root = _copy_harness(tmp_path)
    skill = root / "skills" / "order-support" / "SKILL.md"
    skill.write_text(skill.read_text(encoding="utf-8") + "- Always greet the customer by name.\n",
                     encoding="utf-8")
    monkeypatch.setenv(agent.HARNESS_ENV, str(root))
    assert agent.harness_dir() == root.resolve()
    assert "- Always greet the customer by name.\n\n" + agent.DATE_LINE in agent.instructions()
    client = use_client("hello")
    assert agent.chat("hi") == "hello"
    assert "Always greet the customer by name." in client.requests[0][1]["instructions"]


@pytest.mark.parametrize("bad", ["../outside.md", "prompts/../../outside.md"])
def test_instruction_files_must_stay_inside_the_harness(tmp_path, bad):
    root = _copy_harness(tmp_path)
    (tmp_path / "outside.md").write_text("leaked", encoding="utf-8")
    spec = yaml.safe_load((root / "agent.yaml").read_text(encoding="utf-8"))
    spec["x-ci"]["instructions_files"] = [bad]
    (root / "agent.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        agent._load_spec(root)


def test_absolute_instruction_file_is_rejected(tmp_path):
    root = _copy_harness(tmp_path)
    spec = yaml.safe_load((root / "agent.yaml").read_text(encoding="utf-8"))
    spec["x-ci"]["instructions_files"] = [str((root / "prompts" / "system.md").resolve())]
    (root / "agent.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        agent._load_spec(root)


def test_front_matter_is_only_stripped_at_the_top():
    assert agent._strip_front_matter("---\na: 1\n---\nbody\n") == "body\n"
    assert agent._strip_front_matter("body\n---\nmore\n") == "body\n---\nmore\n"
    assert agent._strip_front_matter("---\nunterminated\n") == "---\nunterminated\n"
