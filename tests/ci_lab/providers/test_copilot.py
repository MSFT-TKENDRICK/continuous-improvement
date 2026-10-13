import asyncio
import time

import pytest
from agent_framework import Agent, Content, Message, tool
from pydantic import BaseModel

from ci_lab.contracts import PROVIDER_MAPPING, PROVIDER_NAME
from ci_lab.providers.copilot import (CopilotChatClient, CopilotSessionError, CopilotTimeoutError, copilot_scope,
                                      session_scope)
from ci_lab.providers.fake_sdk import Fail, FakeCall, FakeCopilotClient, Hang


def lookup(order_id: str) -> str:
    """Look up an order."""
    return f"{order_id}: shipped"


def echo_results(prefix: str = "done"):
    return lambda s: prefix + ": " + "; ".join(r.text_result_for_llm for _, _, r in s.tool_results)


def make(script, **kw):
    sdk = FakeCopilotClient(script)
    client = CopilotChatClient(model="gpt-5-mini", sdk_client=sdk, **kw)
    return sdk, client


def user(text):
    return [Message(role="user", contents=[text])]


def test_tool_loop_with_two_parallel_calls_through_agent():
    records = []
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"}), FakeCall("lookup", {"order_id": "B"})],
                        echo_results()], on_model_request=records.append)
    agent = Agent(client=client, instructions="Be terse.", tools=[lookup])
    result = asyncio.run(agent.run("where are A and B?"))

    assert result.text == "done: A: shipped; B: shipped"
    assert len(sdk.sessions) == 1
    s = sdk.sessions[0]
    assert s.prompts == ["where are A and B?"]
    assert s.kwargs["model"] == "gpt-5-mini"
    assert s.kwargs["system_message"] == {"mode": "replace", "content": "Be terse."}
    assert [t.name for t in s.kwargs["tools"]] == ["lookup"]
    assert all(t.skip_permission for t in s.kwargs["tools"])
    assert "reasoning_effort" not in s.kwargs
    assert [r.result_type for _, _, r in s.tool_results] == ["success", "success"]
    # the second model call continued the same session through the suspended bridge
    assert [r["request"]["path"] for r in records] == ["replay", "bridge"]
    assert client.live_sessions == 1  # kept idle for follow-ups


def test_builtin_tools_are_never_available():
    from copilot import ToolSet

    sdk, client = make(["hi"])
    asyncio.run(client.get_response(user("hello")))
    avail = sdk.sessions[0].kwargs["available_tools"]
    assert isinstance(avail, ToolSet)
    assert avail.to_list() == ["custom:*"]
    assert sdk.sessions[0].kwargs["tools"] == []
    assert sdk.sessions[0].disconnected  # tool-less sessions are one-shot


def test_tool_exception_is_reported_to_copilot_as_failure():
    def broken(order_id: str) -> str:
        """Always fails."""
        raise RuntimeError("database down")

    sdk, client = make([[FakeCall("broken", {"order_id": "A"})], echo_results("sorry")])
    result = asyncio.run(Agent(client=client, tools=[broken]).run("check A"))
    (_, name, tr), = sdk.sessions[0].tool_results
    assert name == "broken"
    assert tr.result_type == "failure"
    assert tr.error
    assert "Error" in tr.text_result_for_llm
    assert result.text == "sorry: " + tr.text_result_for_llm


def test_fallback_replay_for_history_without_live_session():
    sdk, client = make(["It shipped."])
    history = [
        Message(role="user", contents=["where is A?"]),
        Message(role="assistant", contents=[Content.from_function_call(call_id="call_9", name="lookup",
                                                                       arguments={"order_id": "A"})]),
        Message(role="tool", contents=[Content.from_function_result(call_id="call_9", result="A: shipped")]),
    ]
    resp = asyncio.run(client.get_response(history, options={"tools": [tool(lookup)], "instructions": "sys"}))
    assert resp.text == "It shipped."
    assert resp.additional_properties["copilot_path"] == "replay"
    prompt = sdk.sessions[0].prompts[0]
    assert "[user]\nwhere is A?" in prompt
    assert '[assistant tool call call_9] lookup({"order_id": "A"})' in prompt
    assert "[tool result call_9]\nA: shipped" in prompt
    assert sdk.sessions[0].kwargs["system_message"]["content"] == "sys"


def test_continuity_mismatch_aborts_and_replays():
    scripts = iter([[[FakeCall("lookup", {"order_id": "A"})], "first"], ["replayed"]])
    sdk, client = make(lambda kw: next(scripts))

    async def main():
        msgs = user("where is A?")
        opts = {"tools": [tool(lookup)]}
        r1 = await client._inner_get_response(messages=msgs, stream=False, options=opts)
        call = r1.messages[0].contents[0]
        assert call.type == "function_call" and call.call_id == "call_1"
        edited = [*msgs, Message(role="assistant", contents=[Content.from_text("edited"), call]),
                  Message(role="tool", contents=[Content.from_function_result(call_id="call_1", result="x")])]
        r2 = await client._inner_get_response(messages=edited, stream=False, options=opts)
        await asyncio.sleep(0.01)
        return r2

    r2 = asyncio.run(main())
    assert r2.text == "replayed"
    assert r2.additional_properties["copilot_path"] == "replay"
    first = sdk.sessions[0]
    assert first.aborted and first.disconnected
    assert [r.result_type for _, _, r in first.tool_results] == ["failure"]


def test_session_isolation_with_identical_prompts_and_call_ids():
    async def scoped_lookup(order_id: str) -> str:
        """Look up an order."""
        await asyncio.sleep(0.01)
        return f"{order_id}@{session_scope.get()}"

    sdk, client = make([[FakeCall("lookup", {"order_id": "A"}, call_id="call_same")], echo_results("r")])
    agent = Agent(client=client, instructions="sys", tools=[tool(scoped_lookup, name="lookup")])

    async def run(scope):
        with copilot_scope(scope):
            return (await agent.run("where is A?")).text

    async def main():
        return await asyncio.gather(*(run(f"rollout-{i}") for i in range(4)))

    texts = asyncio.run(main())
    assert texts == [f"r: A@rollout-{i}" for i in range(4)]
    assert len(sdk.sessions) == 4
    assert sorted(s.tool_results[0][2].text_result_for_llm for s in sdk.sessions) == [
        f"A@rollout-{i}" for i in range(4)]
    assert sorted(lv.key[0] for lv in client._lives) == [f"rollout-{i}" for i in range(4)]


def test_session_scope_callable_overrides_contextvar():
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], echo_results()], session_scope=lambda: "x")
    with copilot_scope("ignored"):
        asyncio.run(Agent(client=client, tools=[lookup]).run("q"))
    assert [lv.key[0] for lv in client._lives] == ["x"]


def test_followup_turn_reuses_idle_session():
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], echo_results(), "second answer"])
    agent = Agent(client=client, instructions="sys", tools=[lookup])

    async def main():
        session = agent.create_session()
        r1 = await agent.run("where is A?", session=session)
        r2 = await agent.run("thanks, anything else?", session=session)
        return r1, r2

    r1, r2 = asyncio.run(main())
    assert r1.text == "done: A: shipped"
    assert r2.text == "second answer"
    assert len(sdk.sessions) == 1
    assert sdk.sessions[0].prompts == ["where is A?", "thanks, anything else?"]


def test_ttl_reap_aborts_orphaned_session_and_resolves_futures():
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], "never"], session_ttl_s=60)

    async def main():
        r = await client._inner_get_response(messages=user("q"), stream=False, options={"tools": [tool(lookup)]})
        assert r.finish_reason == "tool_calls"
        assert await client.reap() == 0
        assert await client.reap(now=time.monotonic() + 61) == 1
        await asyncio.sleep(0.01)

    asyncio.run(main())
    s = sdk.sessions[0]
    assert s.aborted and s.disconnected
    assert client.live_sessions == 0
    assert [r.result_type for _, _, r in s.tool_results] == ["failure"]


def test_timeout_aborts_session_and_raises():
    sdk, client = make([Hang()], timeout_s=0.05)
    with pytest.raises(CopilotTimeoutError):
        asyncio.run(client.get_response(user("q")))
    s = sdk.sessions[0]
    assert s.aborted and s.disconnected
    assert client.live_sessions == 0


def test_timeout_after_tool_round_aborts():
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], Hang()], timeout_s=0.1)
    with pytest.raises(CopilotTimeoutError):
        asyncio.run(Agent(client=client, tools=[lookup]).run("q"))
    assert sdk.sessions[0].aborted
    assert client.live_sessions == 0


def test_session_error_raises_and_destroys():
    sdk, client = make([Fail("quota exceeded")])
    with pytest.raises(CopilotSessionError, match="quota exceeded"):
        asyncio.run(client.get_response(user("q")))
    assert sdk.sessions[0].aborted and client.live_sessions == 0


def test_usage_and_served_model_mapping():
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], echo_results()])
    result = asyncio.run(Agent(client=client, tools=[lookup]).run("q"))
    u = result.usage_details
    assert (u["input_token_count"], u["output_token_count"], u["total_token_count"]) == (20, 10, 30)
    assert u["cache_read_input_token_count"] == 4
    assert u["copilot_total_nano_aiu"] == 2000
    assert client.last_served_model == "gpt-fake-served"
    resp = asyncio.run(CopilotChatClient(model="gpt-5-mini", sdk_client=FakeCopilotClient(["x"]))
                       .get_response(user("q")))
    assert resp.model == "gpt-fake-served"
    assert resp.additional_properties["requested_model"] == "gpt-5-mini"
    assert resp.usage_details["input_token_count"] == 10
    assert resp.finish_reason == "stop"


def test_on_model_request_callback_shape():
    records = []
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], "fine"], on_model_request=records.append)
    with copilot_scope("rollout-7"):
        asyncio.run(Agent(client=client, tools=[lookup]).run("q"))
    assert len(records) == 2
    first, second = records
    for r in records:
        assert r["model"] == "gpt-fake-served" and r["requested_model"] == "gpt-5-mini"
        assert r["status"] == "ok" and r["scope"] == "rollout-7" and r["latency_ms"] == 7.0
        assert set(r["usage"]) == {"input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                                   "reasoning_tokens", "total_nano_aiu", "cost"}
        assert r["usage"]["input_tokens"] == 10 and r["usage"]["total_nano_aiu"] == 1000
        assert r["session_id"] == sdk.sessions[0].session_id
    assert first["finish_reason"] == "tool_calls"
    assert first["response"]["tool_calls"] == [{"name": "lookup", "call_id": "call_1"}]
    assert second["finish_reason"] == "stop" and second["response"]["text"] == "fine"
    assert second["request"]["tool_results"] == 1


def test_callback_failure_does_not_break_the_call():
    def bad(_):
        raise RuntimeError("nope")

    _, client = make(["ok"], on_model_request=bad)
    assert asyncio.run(client.get_response(user("q"))).text == "ok"


def test_ignored_options_are_recorded():
    sdk, client = make(["ok"])
    resp = asyncio.run(client.get_response(user("q"), options={"temperature": 0.2, "seed": 7, "top_p": 0.9}))
    assert resp.additional_properties["ignored_options"] == {"temperature": 0.2, "seed": 7, "top_p": 0.9}
    assert "temperature" not in sdk.sessions[0].kwargs


def test_reasoning_effort_is_forwarded():
    sdk, client = make(["ok"], reasoning_effort="high")
    asyncio.run(client.get_response(user("q")))
    assert sdk.sessions[0].kwargs["reasoning_effort"] == "high"


class Verdict(BaseModel):
    score: int
    reason: str


def test_structured_output_with_pydantic_and_repair():
    sdk, client = make(["not json at all", '```json\n{"score": 4, "reason": "good"}\n```'])
    resp = asyncio.run(client.get_response(user("grade it"), options={"response_format": Verdict}))
    assert resp.value == Verdict(score=4, reason="good")
    s = sdk.sessions[0]
    assert '"score"' in s.kwargs["system_message"]["content"]
    assert len(s.prompts) == 2 and "not valid" in s.prompts[1]


def test_structured_output_json_schema_dict_and_failure():
    from agent_framework.exceptions import ChatClientInvalidResponseException

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}
    _, client = make(['{"ok": true}'])
    resp = asyncio.run(client.get_response(user("q"), options={"response_format": fmt}))
    assert resp.value == {"ok": True}
    _, client = make(['{"ok": 1}', '{"nope": 1}'])
    with pytest.raises(ChatClientInvalidResponseException):
        asyncio.run(client.get_response(user("q"), options={"response_format": fmt}))


def test_tool_choice_none_fails_new_tool_requests():
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], "no tools then"])
    resp = asyncio.run(client._inner_get_response(messages=user("q"), stream=False,
                                                  options={"tools": [tool(lookup)], "tool_choice": "none"}))
    assert resp.text == "no tools then"
    assert sdk.sessions[0].tool_results[0][2].result_type == "failure"


def test_close_aborts_sessions_and_leaves_injected_client_running():
    sdk, client = make([[FakeCall("lookup", {"order_id": "A"})], "x"])

    async def main():
        async with client:
            await client._inner_get_response(messages=user("q"), stream=False, options={"tools": [tool(lookup)]})
            assert client.live_sessions == 1
        assert client.live_sessions == 0

    asyncio.run(main())
    assert sdk.sessions[0].aborted
    assert sdk.stopped == 0  # injected client is not owned


def test_client_survives_successive_event_loops():
    sdk, client = make(["one"])
    assert asyncio.run(client.get_response(user("a"))).text == "one"
    sdk.script = ["two"]
    assert asyncio.run(client.get_response(user("b"))).text == "two"


def test_declarative_agent_factory_constructs_copilot_client():
    from agent_framework_declarative import AgentFactory

    sdk = FakeCopilotClient(["declared"])
    factory = AgentFactory(client_kwargs={"sdk_client": sdk}, additional_mappings={PROVIDER_NAME: PROVIDER_MAPPING})
    agent = factory.create_agent_from_yaml("""
kind: Prompt
name: judge
instructions: You are a judge.
model:
  id: gpt-5-mini
  provider: GitHubCopilot
  options:
    temperature: 0.1
""")
    assert isinstance(agent.client, CopilotChatClient)
    assert agent.client.model == "gpt-5-mini"
    result = asyncio.run(agent.run("rate this"))
    assert result.text == "declared"
    assert sdk.sessions[0].kwargs["system_message"]["content"] == "You are a judge."


def test_declarative_construction_without_injection():
    from agent_framework_declarative import AgentFactory

    agent = AgentFactory(additional_mappings={PROVIDER_NAME: PROVIDER_MAPPING}).create_agent_from_yaml(
        "kind: Prompt\nname: a\ninstructions: x\nmodel:\n  id: gpt-5-mini\n  provider: GitHubCopilot\n")
    assert isinstance(agent.client, CopilotChatClient)
    assert agent.client._sdk is None  # nothing is started until the first call
