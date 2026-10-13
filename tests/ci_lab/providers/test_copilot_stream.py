"""Buffered streaming path of ``CopilotChatClient`` (used by AG-UI, which always streams)."""

import asyncio

from agent_framework import Agent, ChatResponse, Content, Message, tool

from ci_lab.providers.copilot import CopilotChatClient
from ci_lab.providers.fake_sdk import FakeCall, FakeCopilotClient
from ci_lab.providers.streaming import response_updates


def lookup(path: str) -> str:
    """Read a path."""
    return f"{path}: found"


def make(script, **kw):
    sdk = FakeCopilotClient(script)
    return sdk, CopilotChatClient(model="gpt-5-mini", sdk_client=sdk, **kw)


def collect(stream):
    async def main():
        updates = [u async for u in stream]
        return updates, await stream.get_final_response()
    return asyncio.run(main())


def test_stream_without_tools_yields_text_and_final_response():
    sdk, client = make(["hello there"])
    updates, final = collect(client.get_response([Message(role="user", contents=["hi"])], stream=True))
    assert "".join(u.text for u in updates) == "hello there"
    assert all(u.role == "assistant" for u in updates)
    assert final.text == "hello there"
    assert final.additional_properties["copilot_path"] == "single"
    assert sdk.sessions[0].prompts == ["hi"]


def test_stream_runs_tool_loop_through_agent():
    sdk, client = make([[FakeCall("lookup", {"path": "A"})],
                        lambda s: "done: " + s.tool_results[0][2].text_result_for_llm])
    agent = Agent(client=client, tools=[lookup])
    updates, final = collect(agent.run("where is A?", stream=True))
    types = [c.type for u in updates for c in u.contents]
    assert "function_call" in types and "function_result" in types
    assert final.text == "done: A: found"
    assert len(sdk.sessions) == 1  # the bridge continued the same session


def test_stream_surfaces_approval_request_and_resume_bridges_into_suspended_session():
    ran, records = [], []

    @tool(approval_mode="always_require")
    def launch(cid: str) -> str:
        """Launch a campaign."""
        ran.append(cid)
        return f"launched {cid}"

    sdk, client = make([[FakeCall("launch", {"cid": "abc"})],
                        lambda s: "ok: " + s.tool_results[0][2].text_result_for_llm],
                       on_model_request=records.append)
    agent = Agent(client=client, tools=[launch])

    async def main():
        session = agent.create_session()
        stream = agent.run("launch abc", session=session, stream=True)
        updates = [u async for u in stream]
        await stream.get_final_response()
        requests = [c for u in updates for c in u.contents if c.type == "function_approval_request"]
        assert len(requests) == 1 and requests[0].function_call.name == "launch"
        assert ran == []  # gated: nothing runs before the human answers
        approved = Message(role="user", contents=[requests[0].to_function_approval_response(approved=True)])
        stream = agent.run(approved, session=session, stream=True)
        [u async for u in stream]
        return await stream.get_final_response()

    final2 = asyncio.run(main())
    assert ran == ["abc"]
    assert final2.text == "ok: launched abc"
    assert [r["request"]["path"] for r in records] == ["replay", "bridge"]
    assert len(sdk.sessions) == 1

def test_response_updates_round_trip_preserves_calls_usage_and_finish_reason():
    resp = ChatResponse(
        messages=[Message(role="assistant", contents=[Content.from_text("thinking"),
                                                      Content.from_function_call(call_id="c1", name="lookup",
                                                                                 arguments={"path": "A"})])],
        response_id="r1", model="m", finish_reason="tool_calls",
        usage_details={"input_token_count": 3, "output_token_count": 5, "total_token_count": 8})
    ups = response_updates(resp)
    assert len(ups) == 1 and ups[0].response_id == "r1" and ups[0].finish_reason == "tool_calls"
    back = ChatResponse.from_updates(ups)
    assert back.text == "thinking"
    assert [c.type for c in back.messages[0].contents] == ["text", "function_call"]
    assert back.usage_details["total_token_count"] == 8
    assert back.finish_reason == "tool_calls" and back.model == "m"


def test_response_updates_of_empty_response_is_one_empty_update():
    ups = response_updates(ChatResponse(messages=[]))
    assert len(ups) == 1 and ups[0].contents == []
