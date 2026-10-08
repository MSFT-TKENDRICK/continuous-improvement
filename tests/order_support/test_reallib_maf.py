"""The real order-support declarative agent inside a checkpointed MAF declarative workflow.

``test_runtime.py`` covers one ``agent.chat`` turn over an ``httpx.MockTransport``, and
``tests/ci_lab/maf/test_workflows.py`` covers checkpoint/resume with a plain ``Agent`` on a
``FakeChatClient``. This test joins the production pieces. ``build_agent`` (MAF
``AgentFactory`` over ``harness/agent.yaml`` + the frozen tool bindings) runs on the offline
profile's real ``OpenAIChatCompletionClient``, which speaks HTTP to a deterministic
:class:`~ci_lab.testing.LoopbackLLM`. The agent is driven by ``WorkflowFactory`` through
``ci_lab.maf.workflows.run_or_resume`` with ``FileCheckpointStorage``. The MAF tool loop
executes the real ``lookup_order`` tool between the two model calls.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ci_lab.maf.workflows import latest_checkpoint, run_or_resume
from ci_lab.testing import LoopbackLLM, tool_call
from order_support import agent, tools

WORKFLOW = """\
kind: Workflow
trigger:
  kind: OnConversationStart
  id: order_support_turn
  actions:
    - kind: InvokeAzureAgent
      id: support
      agent: {name: OrderSupport}
      input: {messages: "Where is my order NW-10001?"}
    - kind: InvokeFunctionTool
      id: close
      functionName: close_ticket
      arguments: {}
"""
FINAL = "Your order NW-10001 is on its way."


def respond(body):
    msgs = body["messages"]
    if not any(m.get("role") == "tool" for m in msgs):
        return tool_call("lookup_order", {"order_id": "NW-10001"})
    return FINAL


@pytest.fixture
def llm(monkeypatch):
    srv = LoopbackLLM(respond).start()
    monkeypatch.setenv(agent.PROFILE_ENV, "offline")
    monkeypatch.setenv("OPENAI_API_BASE", f"{srv.url}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "loopback")
    monkeypatch.setenv("ORDER_AGENT_MODEL", "openai/reallib-agent")
    for name in (agent.TIMEOUT_ENV, agent.EVALS_TIMEOUT_ENV, agent.HARNESS_ENV, "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    agent.set_client_override(None)
    yield srv
    srv.stop()


def test_declarative_order_support_agent_runs_in_checkpointed_workflow(llm, tmp_path):
    yaml_path = tmp_path / "turn.yaml"
    yaml_path.write_text(WORKFLOW, encoding="utf-8")
    ckpt = tmp_path / "ckpt"
    closed: list[str] = []

    def close_ticket() -> str:
        closed.append("x")
        return "closed"

    def kwargs():
        client = agent._offline_client()
        agent._configure_client(client)
        return {"agents": {"OrderSupport": agent.build_agent(client)}, "tools": {"close_ticket": close_ticket},
                "checkpoint_dir": ckpt}

    outputs = asyncio.run(run_or_resume(yaml_path, "start", **kwargs()))

    assert outputs == [FINAL, "closed"] and closed == ["x"]
    first, second = llm.of_kind("tools")
    assert first["model"] == second["model"] == "reallib-agent"
    assert first["messages"][0] == {"role": "system", "content": agent.instructions()}
    assert [t["function"] for t in first["tools"]] == [s["function"] for s in tools.TOOL_SCHEMAS]
    tool_msg = next(m for m in second["messages"] if m.get("role") == "tool")
    assert json.loads(tool_msg["content"]) == tools.lookup_order("NW-10001")  # frozen tool really ran

    async def checkpoints():
        from agent_framework import FileCheckpointStorage

        from ci_lab.maf.workflows import declarative_allowlist

        storage = FileCheckpointStorage(ckpt, allowed_checkpoint_types=declarative_allowlist())
        cps = await storage.list_checkpoints(workflow_name="order_support_turn")
        return cps, await latest_checkpoint(storage, "order_support_turn")

    cps, latest = asyncio.run(checkpoints())
    assert len(cps) >= 3 and sorted(c.iteration_count for c in cps) == list(range(len(cps)))
    assert latest.iteration_count == len(cps) - 1

    # resuming a finished run replays nothing: no model call, no tool, no new outputs
    assert asyncio.run(run_or_resume(yaml_path, "again", **kwargs())) == []
    assert len(llm.of_kind("tools")) == 2 and closed == ["x"]
