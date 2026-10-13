"""Declarative MAF subagents for the meta agents (``x-ci.subagents`` -> ``background_agents``)."""
from __future__ import annotations

import asyncio
import shutil

import pytest
import yaml

from ci_lab.contracts import ArmContext, ArmDirective, FailureRecord, Profile
from ci_lab.harness_tree import repo_harness_dir
from ci_lab.meta.run import ArmSurface, MetaAgentError, run_meta_agent, run_proposer
from ci_lab.meta.spec_loader import (
    SPECS_DIR,
    SUBAGENT_TOOLS,
    SpecError,
    default_builder,
    harness_builder,
    load_manifest,
    load_spec,
    loader_builder,
    subagent_specs,
)
from ci_lab.testing import Call, FakeChatClient

H = "harness"
PROMPT = f"{H}/prompts/system.md"
FAILURE = FailureRecord(case_id="change-abc", suite="harness_proposal", category="write_without_read",
                        rule_ids=("judge.policy_violation",), rubric_scores={"policy_violation": 0.0},
                        excerpt="The edit was written before the resource was read.")
SUB_MARK = "You are the failure analyst subagent"
ANSWER = "ROOT CAUSE: system.md never says to read the resource before editing (change-abc)."


def _named(name):
    def fn(**_):
        return f"{name}-result"
    fn.__name__ = name
    return fn


def _bindings(spec):
    return {t: _named(t) for t in spec.tools}


@pytest.fixture
def spec_copy(tmp_path):
    shutil.copytree(repo_harness_dir(), tmp_path / "harness")
    shutil.copy(SPECS_DIR / "critic.yaml", tmp_path / "harness" / "agents")
    return tmp_path / "harness" / "agents"


def _mutate(path, fn):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- parsing


def test_manifest_declares_failure_analyst_and_proposer_uses_it():
    assert load_manifest()["subagents"] == {"failure_analyst": {"spec": "failure_analyst.yaml"}}
    spec = load_spec("proposer")
    [sub] = spec.subagents
    assert sub.key == "failure_analyst" and sub.role == "subagent" and sub.name == "CiFailureAnalyst"
    assert sub.terminal_tool == "" and set(sub.tools) <= SUBAGENT_TOOLS and "read_file" in sub.tools
    assert "failures" in sub.documents and SUB_MARK in sub.instructions
    assert sub.model in load_manifest()["allowed_models"]
    assert "{background_agents}" in spec.subagent_instructions
    assert all(not load_spec(k).subagents for k in ("analyst", "critic", "reflector"))
    assert load_spec("failure_analyst") == sub


def test_subagent_by_contained_path(spec_copy):
    path = _mutate(spec_copy / "proposer.yaml", lambda d: d["x-ci"].update(subagents=["failure_analyst.yaml"]))
    [sub] = load_spec(path).subagents
    assert sub.path == (spec_copy / "failure_analyst.yaml").resolve() and sub.role == "subagent"


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda d: d["x-ci"].update(subagents=["nope"]), "unknown subagent 'nope'"),
    (lambda d: d["x-ci"].update(subagents=["missing.yaml"]), "subagent 'missing.yaml'"),
    (lambda d: d["x-ci"].update(subagents=["../../pyproject.yaml"]), "escapes"),
    (lambda d: d["x-ci"].update(subagents="failure_analyst"), "must be a list"),
    (lambda d: d["x-ci"].update(subagents=["critic"]), "unknown subagent"),
    (lambda d: d["x-ci"].update(subagents=["critic.yaml"]), "only role: subagent"),
    (lambda d: d["x-ci"].update(subagents=["failure_analyst", "failure_analyst.yaml"]), "unique"),
    (lambda d: d["x-ci"].pop("subagents"), "needs subagents"),
    (lambda d: d["x-ci"].update(subagent_instructions=["x"]), "must be a string"),
    (lambda d: d["x-ci"].update(role="boss"), "role 'boss'"),
])
def test_parent_validation_errors(spec_copy, mutation, message):
    path = _mutate(spec_copy / "proposer.yaml", mutation)
    with pytest.raises(SpecError, match=message):
        load_spec(path)


def _write_tool(d):
    d["tools"].append({"kind": "function", "name": "write_file", "description": "w",
                       "bindings": [{"name": "write_file"}]})


@pytest.mark.parametrize(("mutation", "message"), [
    (_write_tool, "write_file are not read-only"),
    (lambda d: d["tools"].append({"kind": "function", "name": "submit_analysis", "description": "s",
                                  "bindings": [{"name": "submit_analysis"}]}), "not read-only"),
    (lambda d: d["x-ci"].update(terminal_tool="submit_analysis"), "no terminal_tool"),
    (lambda d: d["model"].update(id="gpt-5"), "not in the allowlist"),
    (lambda d: d["x-ci"].update(role="agent"), "only role: subagent"),
    (lambda d: d["x-ci"].update(subagents=["proposer.yaml"]), "cycle proposer -> failure_analyst -> proposer"),
    (lambda d: d["x-ci"].update(subagents=["failure_analyst.yaml"]), "cycle failure_analyst -> failure_analyst"),
    (lambda d: d["tools"].append({"kind": "mcp", "name": "shell"}), "only function tools"),
])
def test_subagent_validation_errors(spec_copy, mutation, message):
    _mutate(spec_copy / "failure_analyst.yaml", mutation)
    with pytest.raises(SpecError, match=message):
        load_spec(spec_copy / "proposer.yaml")


def test_subagent_model_must_be_in_manifest_allowlist(spec_copy):
    manifest = {**load_manifest(), "allowed_models": ["gpt-5-mini"], "model": "gpt-5-mini"}
    with pytest.raises(SpecError, match="allowlist"):
        load_spec(spec_copy / "proposer.yaml", manifest=manifest)


# ---------------------------------------------------------------- building


def test_built_proposer_has_maf_background_agent_with_readonly_tools():
    from agent_framework import Agent, BackgroundAgentsProvider

    spec = load_spec("proposer")
    subs = {s.key: _bindings(s) for s in subagent_specs(spec)}
    agent = default_builder()(spec, client=FakeChatClient(), bindings=_bindings(spec), subagent_bindings=subs)
    [provider] = [p for p in agent.context_providers if isinstance(p, BackgroundAgentsProvider)]
    assert provider.source_id == "background_agents"
    [sub] = provider._agents.values()
    assert isinstance(sub, Agent) and sub.name == "CiFailureAnalyst"
    sub_tools = {t.name for t in sub.default_options["tools"]}
    assert sub_tools == set(spec.subagents[0].tools) and sub_tools <= SUBAGENT_TOOLS
    assert not [p for p in sub.context_providers if isinstance(p, BackgroundAgentsProvider)]
    parent_tools = {t.name for t in agent.default_options["tools"]}
    assert {"write_file", "commit_edit", "submit_proposal_done"} <= parent_tools
    assert sub.client is agent.client  # same chat client (profile/provider) as the parent
    assert agent.additional_properties["ci_lab"]["subagents"].keys() == {"failure_analyst"}


def test_builder_fails_closed_without_subagent_bindings(tmp_path):
    spec = load_spec("proposer")
    with pytest.raises(SpecError, match="no bindings for subagent 'failure_analyst'"):
        harness_builder(spec, client=FakeChatClient(), bindings=_bindings(spec))
    with pytest.raises(SpecError, match="loader cannot build subagents"):
        loader_builder(lambda spec_path, *, client, bindings: "agent")(spec, client="c", bindings=_bindings(spec))
    with pytest.raises(MetaAgentError, match="no bindings for subagent"):
        asyncio.run(run_meta_agent("proposer", tmp_path, FakeChatClient(),
                                   _bindings(spec), spec=spec, reuse=False))


def test_builder_never_binds_more_than_the_subagent_spec_tools():
    from agent_framework import BackgroundAgentsProvider

    spec = load_spec("proposer")
    sub = spec.subagents[0]
    extra = {**_bindings(sub), "write_file": _named("write_file"), "commit_edit": _named("commit_edit")}
    agent = harness_builder(spec, client=FakeChatClient(), bindings=_bindings(spec),
                            subagent_bindings={sub.key: extra})
    [provider] = [p for p in agent.context_providers if isinstance(p, BackgroundAgentsProvider)]
    [built] = provider._agents.values()
    assert {t.name for t in built.default_options["tools"]} == set(sub.tools)


# ---------------------------------------------------------------- delegation (end to end)


class RoutingClient(FakeChatClient):
    """One shared chat client; requests whose instructions are the subagent's use ``sub_script``."""

    def __init__(self, parent_script, sub_script):
        super().__init__(default="UNEXPECTED")
        self.parent_script, self.sub_script = list(parent_script), list(sub_script)
        self.sub_requests: list = []
        self.parent_requests: list = []

    async def _respond(self, messages, options):
        text = str(options.get("instructions") or "") + " ".join(m.text or "" for m in messages)
        tools = {tool.name for tool in options.get("tools", [])}
        sub = SUB_MARK in text or ("read_brief" in tools and "submit_proposal_done" not in tools)
        (self.sub_requests if sub else self.parent_requests).append((list(messages), dict(options)))
        self.script = self.sub_script if sub else self.parent_script
        return await super()._respond(messages, options)


def _texts(requests):
    out = []
    for msgs, _ in requests:
        parts = []
        for m in msgs:
            for c in m.contents:
                value = getattr(c, "text", None) or getattr(c, "result", None)
                parts.append(value if isinstance(value, str) else str(value or ""))
        out.append(" ".join(parts))
    return out


def test_proposer_delegates_to_failure_analyst_and_gets_its_answer(repo, tmp_path, layout):
    wt, base = repo
    ctx = ArmContext(experiment_id="exp-1", directive=ArmDirective(arm="arm-a", component_focus=("prompt",),
                                                                   edit_budget=1),
                     worktree=wt, base_commit=base, failures=[FAILURE], profile=Profile.FAKE,
                     run_dir=tmp_path / "runs" / "arm-a")
    surface = ArmSurface(layout.surface, layout.component_globs, layout.frozen)
    new = "You are a careful harness agent.\nRead the resource before editing.\nValidate every change.\n"
    parent = [
        [Call("background_agents_start_task", {"agent_name": "CiFailureAnalyst", "input": "Why do changes fail?",
                                               "description": "change drill-down"})],
        [Call("background_agents_wait_for_first_completion", {"task_ids": [1]})],
        [Call("background_agents_get_task_results", {"task_id": 1})],
        [Call("write_file", {"path": PROMPT, "content": new})],
        [Call("commit_edit", {"component": "prompt", "hypothesis": "Read resources before changes."})],
        [Call("submit_proposal_done", {"summary": "read first", "predicted_fixes": ["change-abc"]})],
    ]
    sub = [
        [Call("read_brief", {"name": "failures"})],
        [Call("read_file", {"path": PROMPT})],
        [Call("write_file", {"path": PROMPT, "content": "hacked"})],  # not bound: refused by MAF
        ANSWER,
    ]
    client = RoutingClient(parent, sub)
    result = asyncio.run(run_proposer(ctx, client, surface=surface))

    assert result.submission.predicted_fixes == ["change-abc"] and len(result.edits) == 1
    assert len(client.sub_requests) == 4 and not client.sub_script and not client.parent_script
    sub_tools = {t.name for t in client.sub_requests[0][1]["tools"]}
    assert sub_tools == set(load_spec("failure_analyst").tools)
    parent_tools = {t.name for t in client.parent_requests[0][1]["tools"]}
    assert {"background_agents_start_task", "write_file", "commit_edit"} <= parent_tools
    sub_texts = _texts(client.sub_requests)
    assert "change-abc" in sub_texts[1]  # failures document (not readable by the proposer) reached the subagent
    assert "Read resources before edits." in sub_texts[2]
    assert (wt / PROMPT).read_text(encoding="utf-8") == new  # subagent write refused; parent's edit landed
    parent_texts = _texts(client.parent_requests)
    # subagent answer flowed back via get_task_results, sanitized for the student (contract v2 §6)
    assert any("never says to read the resource before editing" in t for t in parent_texts)
    assert not any("change-abc" in t for t in parent_texts)
    assert "CiFailureAnalyst" in str(client.parent_requests[0][1].get("instructions"))
