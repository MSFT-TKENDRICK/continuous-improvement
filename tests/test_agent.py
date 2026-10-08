"""order_support.agent.chat on the MAF declarative agent (FakeChatClient; no model server)."""

import json

import pytest

from ci_lab.testing import Call, FakeChatClient
from order_support import agent


@pytest.fixture
def use_client(monkeypatch):
    for name in (agent.TIMEOUT_ENV, agent.EVALS_TIMEOUT_ENV, "ORDER_AGENT_TEMPERATURE", agent.HARNESS_ENV):
        monkeypatch.delenv(name, raising=False)

    def use(*script, client=None):
        client = client if client is not None else FakeChatClient(script=list(script))
        agent.set_client_override(client)
        return client

    yield use
    agent.set_client_override(None)


def _role(message):
    return str(getattr(message.role, "value", message.role))


def test_tool_loop_executes_tools_and_returns_final_text(use_client):
    client = use_client([Call("lookup_order", {"order_id": "NW-10007"}, "t1")],
                        "Your order is delayed until 2026-09-26.")
    assert agent.chat("Where is NW-10007? ivy.chen@example.com") == "Your order is delayed until 2026-09-26."
    assert len(client.requests) == 2
    messages, options = client.requests[1]
    assert options["instructions"] == agent.instructions()
    assert options["instructions"].startswith(agent.data.load_policy())
    result = messages[-1].contents[0]
    assert _role(messages[-1]) == "tool" and result.call_id == "t1"
    assert json.loads(result.result)["new_estimated_delivery"] == "2026-09-26"


def test_history_is_replayed_without_duplicating_current_turn(use_client):
    client = use_client("ok")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
               {"role": "user", "content": "refund NW-10001"}]
    agent.chat("refund NW-10001", history=history)
    messages = client.requests[0][0]
    assert [_role(m) for m in messages] == ["user", "assistant", "user"]
    assert messages[-1].text == "refund NW-10001"


def test_tool_loop_is_bounded(use_client):
    client = use_client(client=FakeChatClient(
        script=[[Call("search_kb", {"query": "returns"}, f"t{i}")] for i in range(20)]))
    assert agent.chat("loop forever") == agent.LOOP_EXCEEDED_TEXT == "[agent: tool loop exceeded]"
    assert len(client.requests) == agent.MAX_TOOL_LOOP_ITERATIONS


def test_loop_bound_allows_a_final_answer_on_the_last_call(use_client):
    script = [[Call("search_kb", {"query": "returns"})]] * (agent.MAX_TOOL_LOOP_ITERATIONS - 1) + ["done"]
    client = use_client(*script)
    assert agent.chat("several lookups") == "done"
    assert len(client.requests) == agent.MAX_TOOL_LOOP_ITERATIONS


@pytest.mark.parametrize(("agent_env", "evals_env", "expected"), [
    (None, None, agent.DEFAULT_TIMEOUT_S),
    (None, "1800", 1800.0),
    ("90", "1800", 90.0),
    ("  ", "45.5", 45.5),
])
def test_every_model_call_gets_an_explicit_timeout(monkeypatch, use_client, agent_env, evals_env, expected):
    use_client([Call("search_kb", {"query": "returns"})], "done")
    for name, value in ((agent.TIMEOUT_ENV, agent_env), (agent.EVALS_TIMEOUT_ENV, evals_env)):
        if value is not None:
            monkeypatch.setenv(name, value)
    timeouts = []
    real_guard = agent._TurnGuard

    class RecordingGuard(real_guard):
        async def process(self, context, call_next):
            timeouts.append(self.timeout_s)
            await super().process(context, call_next)

    monkeypatch.setattr(agent, "_TurnGuard", RecordingGuard)
    agent.chat("returns?")
    assert timeouts == [expected, expected]


@pytest.mark.parametrize("bad", ["0", "-5", "soon"])
def test_invalid_agent_timeout_is_rejected(monkeypatch, bad):
    monkeypatch.setenv(agent.TIMEOUT_ENV, bad)
    with pytest.raises(ValueError):
        agent.agent_timeout()
