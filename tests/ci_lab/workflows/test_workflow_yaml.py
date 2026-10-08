from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from ci_lab.testing import Call, FakeChatClient
from ci_lab.workflows import (
    ARM_YAML,
    FORBIDDEN_ACTIONS,
    WORKFLOW_FILES,
    agent_names,
    assert_expression_free,
    function_names,
)
from ci_lab.workflows.runtime import (
    STATUS_FILE,
    GatedAgent,
    StepAborted,
    StepFailed,
    build_workflow,
    run_or_resume,
)

EXPECTED_ORDER = {
    "round": ["begin_round", "analyst", "run_arms", "select", "record", "publish"],
    "arm": ["provision_slot", "proposer", "critique_1", "repair_1", "critique_2", "repair_2",
            "critique_final", "evaluate", "finalize_arm"],
    "calibrate": ["aa_runs", "delta", "record"],
    "confirm": ["reserve_look", "evaluate_heldout", "decide", "record"],
}


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _assert_expression_free(path: Path) -> None:
    """Independent check of the no-PowerFx contract (prefers M1's checker when present)."""
    try:
        from ci_lab.maf.workflows import assert_expression_free as m1_check  # type: ignore[import-not-found]
    except ImportError:
        m1_check = None
    if m1_check is not None:
        m1_check(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.split("#", 1)[0]
        if ":" in stripped:
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            assert not value.startswith("="), f"expression in {path.name}: {line!r}"
    for node in _walk(doc):
        assert node.get("kind") not in FORBIDDEN_ACTIONS | {"Goto", "SetVariable", "SetValue"}
        for value in node.values():
            if isinstance(value, str):
                assert not value.lstrip().startswith("=")
    for action in doc["trigger"]["actions"]:
        assert action["kind"] in {"InvokeFunctionTool", "InvokeAzureAgent"}
        for value in (action.get("arguments") or {}).values():
            assert isinstance(value, (str, int, float, bool)), "arguments must be literals"


@pytest.mark.parametrize("name", sorted(WORKFLOW_FILES))
def test_yaml_is_expression_free(name: str) -> None:
    path = WORKFLOW_FILES[name]
    _assert_expression_free(path)
    doc = assert_expression_free(path)
    assert [a["id"] for a in doc["trigger"]["actions"]] == EXPECTED_ORDER[name]


def test_round_yaml_analyst_message_is_literal() -> None:
    doc = assert_expression_free(WORKFLOW_FILES["round"])
    analyst = next(a for a in doc["trigger"]["actions"] if a["kind"] == "InvokeAzureAgent")
    assert analyst["agent"]["name"] == "Analyst"
    assert analyst["input"]["messages"] == "Read your brief with your tools, then call submit_analysis."


@pytest.mark.parametrize("bad", [
    "kind: Workflow\ntrigger: {kind: OnConversationStart, id: x, actions: "
    "[{kind: InvokeFunctionTool, id: a, functionName: f, arguments: {x: '=Local.y'}}]}",
    "kind: Workflow\ntrigger: {kind: OnConversationStart, id: x, actions: "
    "[{kind: If, id: a, condition: '=true', actions: []}]}",
    "kind: Workflow\ntrigger: {kind: OnConversationStart, id: x, actions: "
    "[{kind: Foreach, id: a, items: [1], actions: []}]}",
    "kind: Workflow\ntrigger: {kind: OnConversationStart, id: x, actions: "
    "[{kind: InvokeFunctionTool, id: a, functionName: f, arguments: {x: {nested: 1}}}]}",
    "kind: Workflow\ntrigger: {kind: OnConversationStart, id: x, actions: "
    "[{kind: InvokeFunctionTool, id: a, functionName: f}, {kind: InvokeFunctionTool, id: a, functionName: g}]}",
    "kind: Workflow\ntrigger: {kind: OnConversationStart, id: x, actions: "
    "[{kind: InvokeAzureAgent, id: a, agent: {name: A}, input: {messages: '=System.LastMessage'}}]}",
])
def test_checker_rejects_expressions_and_control_flow(bad: str) -> None:
    with pytest.raises(ValueError):
        assert_expression_free(bad)


def _agent(name: str, tool_name: str, log: list[str]) -> Any:
    from agent_framework import Agent

    def tool(note: str = "") -> str:
        log.append(f"{name}:{tool_name}")
        return "ok"

    tool.__name__ = tool_name
    tool.__doc__ = f"{tool_name} (stub)"
    client = FakeChatClient([[Call(tool_name, {"note": "x"})], "done"])
    return Agent(client=client, name=name, instructions="stub", tools=[tool])


@pytest.mark.parametrize("name", sorted(WORKFLOW_FILES))
def test_yaml_loads_and_runs_with_workflow_factory(name: str, tmp_path: Path) -> None:
    path = WORKFLOW_FILES[name]
    log: list[str] = []

    def stub(fn: str):
        async def call(**kwargs: Any) -> dict[str, Any]:
            log.append(f"{fn}:{sorted(kwargs.items())}")
            return {"ok": True}
        return call

    tools = {fn: stub(fn) for fn in function_names(path)}
    agents = {a: _agent(a, f"submit_{a.lower()}", log) for a in agent_names(path)}
    workflow = build_workflow(path, agents, tools, tmp_path / "ckpt")
    result = asyncio.run(run_or_resume(workflow, tmp_path / "ckpt"))
    assert result["status"] == "completed"
    called = [entry.split(":")[0] for entry in log]
    for fn in function_names(path):
        assert fn in called
    for agent in agent_names(path):
        assert agent in called
    assert len([e for e in log if e.startswith("critique")]) == (3 if name == "arm" else 0)


def test_runtime_crash_then_resume_reruns_only_from_failed_step(tmp_path: Path) -> None:
    calls: list[str] = []
    state = {"crash": True}

    def make_tools() -> dict[str, Any]:
        def gen(step: str):
            async def fn(**kwargs: Any) -> str:
                calls.append(step)
                if step == "evaluate" and state["crash"]:
                    raise StepAborted("evaluate", RuntimeError("boom"))
                return step
            return fn
        return {n: gen(n) for n in function_names(ARM_YAML)}

    proposals: list[str] = []
    done = tmp_path / "proposal.json"

    def make_agents() -> dict[str, Any]:
        from agent_framework import Agent

        def submit_proposal(note: str = "") -> str:
            """stub"""
            proposals.append(note)
            done.write_text("{}")
            return "ok"

        agent = Agent(client=FakeChatClient([[Call("submit_proposal", {"note": "n"})], "done"]),
                      name="Proposer", instructions="x", tools=[submit_proposal])
        return {"Proposer": GatedAgent(agent, done)}

    ckpt = tmp_path / "ckpt"
    with pytest.raises(StepFailed, match="evaluate"):
        asyncio.run(run_or_resume(build_workflow(ARM_YAML, make_agents(), make_tools(), ckpt), ckpt))
    assert '"running"' in (ckpt / STATUS_FILE).read_text()
    first = list(calls)
    assert first[-1] == "evaluate" and "finalize_arm" not in first
    state["crash"] = False
    calls.clear()
    result = asyncio.run(run_or_resume(build_workflow(ARM_YAML, make_agents(), make_tools(), ckpt), ckpt))
    assert result["resumed"] is True
    assert calls == ["evaluate", "finalize_arm"]
    assert proposals == ["n"]
    again = asyncio.run(run_or_resume(build_workflow(ARM_YAML, make_agents(), make_tools(), ckpt), ckpt))
    assert again["skipped"] is True


def test_gated_agent_skips_when_output_exists(tmp_path: Path) -> None:
    class Inner:
        name = "inner"
        ran = 0

        async def run(self, messages: Any = None, **kw: Any) -> str:
            self.ran += 1
            return "ran"

    inner = Inner()
    gated = GatedAgent(inner, tmp_path / "out.json")
    assert asyncio.run(gated.run("x")) == "ran"
    (tmp_path / "out.json").write_text("{}")
    assert asyncio.run(gated.run("x")).startswith("already submitted")
    assert inner.ran == 1 and gated.name == "inner"
