"""Client profiles, sync->async execution, per-call timeout and the tool-loop bound."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types

import httpx
import openai
import pytest

from ci_lab.testing import Call, FakeChatClient
from order_support import agent, tools


def _tool_parented(spans):
    [root] = [s for s in spans if s.attributes.get("openinference.span.kind") == "AGENT"]
    children = [s for s in spans if s.attributes.get("openinference.span.kind") in ("TOOL", "LLM")]
    return bool(children) and all(s.parent_span_id == root.span_id for s in children)


def _lookup_then(text):
    return [[Call("lookup_order", {"order_id": "NW-10001"})], text]


def test_chat_inside_a_running_event_loop(use_client, captured):
    use_client(*_lookup_then("done"))

    async def main():
        return agent.chat("Where is NW-10001?")

    assert asyncio.run(main()) == "done"
    assert _tool_parented(captured())


def test_chat_from_a_worker_thread_like_assert(use_client, captured):
    use_client(*_lookup_then("done"))

    async def main():  # assert_ai invoke_callable runs sync targets via asyncio.to_thread
        return await asyncio.to_thread(agent.chat, "Where is NW-10001?")

    assert asyncio.run(main()) == "done"
    assert _tool_parented(captured())


def test_concurrent_chats_on_threads(use_client):
    use_client(client=FakeChatClient(default="same"))
    results = []
    threads = [threading.Thread(target=lambda: results.append(agent.chat("hi"))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["same"] * 4


def test_errors_propagate_from_the_worker_thread(use_client):
    def boom(messages, options):
        raise RuntimeError("model exploded")

    use_client(boom)

    async def main():
        return agent.chat("hi")

    with pytest.raises(RuntimeError, match="model exploded"):
        asyncio.run(main())


class SlowClient(FakeChatClient):
    async def _respond(self, messages, options):
        await asyncio.sleep(5)
        return await super()._respond(messages, options)


def test_model_call_timeout_is_enforced(use_client, monkeypatch):
    monkeypatch.setenv(agent.TIMEOUT_ENV, "0.05")
    use_client(client=SlowClient())
    with pytest.raises(TimeoutError):
        agent.chat("hi")


def test_temperature_env_is_forwarded(use_client, monkeypatch):
    monkeypatch.setenv("ORDER_AGENT_TEMPERATURE", "0.2")
    client = use_client("ok")
    agent.chat("hi")
    assert client.requests[0][1]["temperature"] == 0.2


def test_tool_calls_run_sequentially_in_model_order(use_client, captured):
    calls = [Call("search_kb", {"query": "returns"}), Call("lookup_order", {"order_id": "NW-10001"}),
             Call("search_kb", {"query": "warranty"})]
    use_client(calls, "done")
    agent.chat("questions")
    tool_spans = [s for s in captured() if s.attributes.get("openinference.span.kind") == "TOOL"]
    assert [json.loads(s.attributes["input.value"]) for s in tool_spans] == [c.arguments for c in calls]
    assert all(a.end_time_ns <= b.start_time_ns for a, b in zip(tool_spans, tool_spans[1:]))


# ------------------------------------------------------------------ profiles

def test_profile_defaults_to_offline(monkeypatch):
    monkeypatch.delenv(agent.PROFILE_ENV, raising=False)
    assert agent.agent_profile() == "offline"
    monkeypatch.setenv(agent.PROFILE_ENV, " Copilot ")
    assert agent.agent_profile() == "copilot"
    monkeypatch.setenv(agent.PROFILE_ENV, "litellm")
    with pytest.raises(ValueError, match="ORDER_AGENT_PROFILE"):
        agent.agent_profile()


def test_fake_profile(monkeypatch, captured):
    agent.set_client_override(None)
    monkeypatch.setenv(agent.PROFILE_ENV, "fake")
    assert agent.chat("hi") == "ok"
    [root] = [s for s in captured() if s.name == "agent.chat"]
    assert root.attributes["llm.model_name"] == "fake-model"


def test_offline_profile_talks_openai_chat_completions(monkeypatch, captured):
    agent.set_client_override(None)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        message = {"role": "assistant", "content": "hello from llama"}
        if len(seen) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "lookup_order", "arguments": '{"order_id": "NW-10007"}'}}]}
        return httpx.Response(200, json={
            "id": f"c{len(seen)}", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if len(seen) == 1 else "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}})

    real = openai.AsyncOpenAI
    built = []

    def async_openai(**kwargs):
        client = real(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", async_openai)
    monkeypatch.setenv(agent.PROFILE_ENV, "offline")
    monkeypatch.setenv("ORDER_AGENT_MODEL", "openai/qwen3")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:8081/v1")
    monkeypatch.setenv(agent.TIMEOUT_ENV, "42")
    assert agent.chat("Where is NW-10007?") == "hello from llama"

    assert str(built[0].base_url) == "http://127.0.0.1:8081/v1/"
    assert built[0].timeout == 42.0
    assert [b["model"] for b in seen] == ["qwen3", "qwen3"]
    assert seen[0]["messages"][0] == {"role": "system", "content": agent.instructions()}
    assert seen[0]["messages"][1] == {"role": "user", "content": "Where is NW-10007?"}
    assert [t["function"] for t in seen[0]["tools"]] == [s["function"] for s in tools.TOOL_SCHEMAS]
    tool_msg = seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "t1"
    assert json.loads(tool_msg["content"]) == tools.lookup_order("NW-10007")

    spans = captured()
    [root] = [s for s in spans if s.name == "agent.chat"]
    assert root.attributes["llm.model_name"] == "openai/qwen3"
    llm = [s for s in spans if s.attributes.get("openinference.span.kind") == "LLM"]
    assert [s.attributes["llm.token_count.prompt"] for s in llm] == [50, 50]
    assert all(s.attributes["llm.provider"] == "openai" for s in llm)


def _providers_package(monkeypatch):
    try:
        import ci_lab.providers  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "ci_lab.providers", types.ModuleType("ci_lab.providers"))


def test_copilot_profile_uses_copilot_chat_client(monkeypatch, captured):
    agent.set_client_override(None)
    _providers_package(monkeypatch)
    made = []

    def copilot_chat_client(model=None):
        made.append(model)
        return FakeChatClient(script=["from copilot"], model=model)

    module = types.ModuleType("ci_lab.providers.copilot")
    module.CopilotChatClient = copilot_chat_client
    monkeypatch.setitem(sys.modules, "ci_lab.providers.copilot", module)
    monkeypatch.setenv(agent.PROFILE_ENV, "copilot")
    assert agent.chat("hi") == "from copilot"
    alias = agent.spec_model_alias(agent._load_spec())
    assert made == [alias]
    [root] = [s for s in captured() if s.name == "agent.chat"]
    assert root.attributes["llm.model_name"] == alias


def test_copilot_profile_without_provider_fails_clearly(monkeypatch):
    agent.set_client_override(None)
    monkeypatch.setitem(sys.modules, "ci_lab.providers.copilot", None)
    monkeypatch.setenv(agent.PROFILE_ENV, "copilot")
    with pytest.raises(RuntimeError, match="CopilotChatClient"):
        agent.chat("hi")


def test_override_beats_profile(monkeypatch, use_client):
    monkeypatch.setenv(agent.PROFILE_ENV, "copilot")
    monkeypatch.setitem(sys.modules, "ci_lab.providers.copilot", None)
    use_client("pinned")
    assert agent.chat("hi") == "pinned"
