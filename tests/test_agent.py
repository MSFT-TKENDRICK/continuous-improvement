import json
from types import SimpleNamespace

from order_support import agent


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _call(cid, name, args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def test_tool_loop_executes_tools_and_returns_final_text(monkeypatch):
    seen = []
    replies = iter([
        _response(tool_calls=[_call("t1", "lookup_order", {"order_id": "NW-10007"})]),
        _response(content="Your order is delayed until 2026-09-26."),
    ])

    def fake_completion(**kwargs):
        seen.append(kwargs["messages"])
        return next(replies)

    monkeypatch.setattr(agent.litellm, "completion", fake_completion)
    assert agent.chat("Where is NW-10007? ivy.chen@example.com") == "Your order is delayed until 2026-09-26."
    second = seen[1]
    assert second[0] == {"role": "system", "content": agent.SYSTEM_PROMPT}
    tool_msg = second[-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "t1"
    assert json.loads(tool_msg["content"])["new_estimated_delivery"] == "2026-09-26"


def test_history_is_replayed_without_duplicating_current_turn(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _response(content="ok")

    monkeypatch.setattr(agent.litellm, "completion", fake_completion)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
               {"role": "user", "content": "refund NW-10001"}]
    agent.chat("refund NW-10001", history=history)
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured["messages"][-1]["content"] == "refund NW-10001"


def test_tool_loop_is_bounded(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(1)
        return _response(tool_calls=[_call(f"t{len(calls)}", "search_kb", {"query": "returns"})])

    monkeypatch.setattr(agent.litellm, "completion", fake_completion)
    assert agent.chat("loop forever") == "[agent: tool loop exceeded]"
    assert len(calls) == agent.MAX_TOOL_LOOP_ITERATIONS
