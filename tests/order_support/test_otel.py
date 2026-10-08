"""OpenInference LLM spans from the MAF middleware: ASSERT reads the same transcript as before M3."""

from __future__ import annotations

import asyncio
import json

from agent_framework import observability as maf_obs
from assert_ai.core.otel import OTelSpan, _spans_to_events, validate_spans

from ci_lab.testing import Call, FakeChatClient
from order_support import agent, otel, tools

USER = "Where is NW-10007? My email is ivy.chen@example.com"
FINAL = "Your order is delayed until 2026-09-26 because of a carrier weather delay."
MODEL = "fake-model"


class UsageClient(FakeChatClient):
    """FakeChatClient that reports token usage like a real endpoint."""

    async def _respond(self, messages, options):
        response = await super()._respond(messages, options)
        response.usage_details = {"input_token_count": 100 + len(self.requests),
                                  "output_token_count": 10, "total_token_count": 110 + len(self.requests)}
        return response


def _legacy_spans() -> list[OTelSpan]:
    """The spans the pre-M3 LiteLLM loop emitted for the same scenario (end order)."""
    args = {"order_id": "NW-10007"}
    first_output = json.dumps({"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "lookup_order", "arguments": json.dumps(args)}}]}}]})

    def span(sid, parent, name, attrs):
        return OTelSpan(trace_id="t", span_id=sid, parent_span_id=parent, name=name,
                        kind=attrs["openinference.span.kind"], start_time_ns=0, end_time_ns=0, attributes=attrs)

    llm = {"openinference.span.kind": "LLM", "llm.model_name": MODEL}
    return [
        span("l1", "a", "completion", {**llm, "output.value": first_output}),
        span("t1", "a", "tool.lookup_order", {"openinference.span.kind": "TOOL", "tool.name": "lookup_order",
                                              "input.value": tools._json(args),
                                              "output.value": tools._json(tools.lookup_order("NW-10007"))}),
        span("l2", "a", "completion", {**llm, "output.value": FINAL}),
        span("a", None, "agent.chat", {"openinference.span.kind": "AGENT", "input.value": USER,
                                       "output.value": FINAL, "llm.model_name": MODEL}),
    ]


def _run_scenario(use_client, captured, client_cls=FakeChatClient):
    use_client(client=client_cls(script=[[Call("lookup_order", {"order_id": "NW-10007"})], FINAL]))
    assert agent.chat(USER) == FINAL
    return captured()


def _by_kind(spans, kind):
    return [s for s in spans if s.attributes.get("openinference.span.kind") == kind]


def test_transcript_parity_with_litellm_path(use_client, captured):
    spans = _run_scenario(use_client, captured)
    events, aggregate = _spans_to_events(spans)
    old_events, old_aggregate = _spans_to_events(_legacy_spans())

    def tool_calls(evts):
        return [e["edit"] for e in evts if e["edit"]["type"] == "tool_call"]

    def messages(evts):
        return [e["edit"]["message"] for e in evts if e["edit"]["type"] == "add_message"]

    assert tool_calls(events) == tool_calls(old_events)
    assert [e["actor"] for e in events] == [e["actor"] for e in old_events]
    new_msgs, old_msgs = messages(events), messages(old_events)
    assert len(new_msgs) == len(old_msgs) == 3
    assert new_msgs[1:] == old_msgs[1:] == [{"role": "assistant", "content": FINAL}] * 2
    assert "lookup_order" in new_msgs[0]["content"]  # tool-call turn is recorded, as before
    for key in ("llm_call_count", "tools_called", "total_tokens"):
        assert aggregate[key] == old_aggregate[key]


def test_spans_nest_under_the_agent_root(use_client, captured):
    spans = _run_scenario(use_client, captured)
    [root] = _by_kind(spans, "AGENT")
    assert root.name == "agent.chat" and root.parent_span_id is None
    assert root.attributes["input.value"] == USER
    assert root.attributes["output.value"] == FINAL
    assert root.attributes["llm.model_name"] == MODEL
    children = _by_kind(spans, "LLM") + _by_kind(spans, "TOOL")
    assert len(children) == 3
    assert all(s.parent_span_id == root.span_id and s.trace_id == root.trace_id for s in children)
    assert len(spans) == 4  # nothing else, in particular no MAF gen_ai.* spans
    assert not [s for s in spans if s.convention == "gen_ai" or any(k.startswith("gen_ai.") for k in s.attributes)]


def test_llm_span_attribute_contract(use_client, captured):
    first, second = _by_kind(_run_scenario(use_client, captured), "LLM")
    a = first.attributes
    assert first.name == otel.LLM_SPAN_NAME
    assert a["llm.model_name"] == MODEL
    assert a["llm.input_messages.0.message.role"] == "system"
    assert a["llm.input_messages.0.message.content"] == agent.instructions()
    assert a["llm.input_messages.1.message.role"] == "user"
    assert a["llm.input_messages.1.message.content"] == USER
    assert a["llm.output_messages.0.message.role"] == "assistant"
    prefix = "llm.output_messages.0.message.tool_calls.0.tool_call"
    assert a[f"{prefix}.function.name"] == "lookup_order"
    assert json.loads(a[f"{prefix}.function.arguments"]) == {"order_id": "NW-10007"}
    call_id = a[f"{prefix}.id"]
    assert json.loads(a["input.value"])["messages"][1] == {"role": "user", "content": USER}
    assert a["input.mime_type"] == "application/json"
    assert ({json.loads(a[f"llm.tools.{i}.tool.json_schema"])["function"]["name"] for i in range(len(tools.TOOLS))}
            == set(tools.TOOLS))
    b = second.attributes
    assert b["llm.input_messages.2.message.tool_calls.0.tool_call.id"] == call_id
    assert b["llm.input_messages.3.message.role"] == "tool"
    assert b["llm.input_messages.3.message.tool_call_id"] == call_id
    assert b["llm.input_messages.3.message.name"] == "lookup_order"
    assert json.loads(b["llm.input_messages.3.message.content"]) == tools.lookup_order("NW-10007")
    assert b["llm.output_messages.0.message.content"] == FINAL
    assert b["output.value"] == FINAL


def test_token_counts_reach_assert(use_client, captured):
    spans = _run_scenario(use_client, captured, client_cls=UsageClient)
    llm = _by_kind(spans, "LLM")
    assert [s.attributes["llm.token_count.prompt"] for s in llm] == [101, 102]
    assert all(s.attributes["llm.token_count.completion"] == 10 for s in llm)
    _, aggregate = _spans_to_events(spans)
    assert aggregate["total_tokens"] == {"input": 203, "output": 20}
    assert validate_spans(spans).valid


def test_maf_telemetry_suppression_is_scoped():
    settings = maf_obs.OBSERVABILITY_SETTINGS
    before = settings.ENABLED
    with otel.suppress_maf_telemetry():
        assert settings.ENABLED is False and settings.SENSITIVE_DATA_ENABLED is False
        assert otel.maf_telemetry_suppressed()

        async def child():
            return settings.ENABLED

        assert asyncio.run(child()) is False  # tasks inherit the context
    assert settings.ENABLED == before
    assert not otel.maf_telemetry_suppressed()


def test_maf_spans_are_not_emitted_even_when_maf_telemetry_is_on(use_client, captured, monkeypatch):
    otel._install_scoped_settings()
    base = type(maf_obs.OBSERVABILITY_SETTINGS).__mro__[1]
    monkeypatch.setattr(base, "ENABLED", property(lambda self: True))
    assert maf_obs.OBSERVABILITY_SETTINGS.ENABLED is True
    spans = _run_scenario(use_client, captured)
    assert not [s for s in spans if any(k.startswith("gen_ai.") for k in s.attributes)]
    assert len(spans) == 4
