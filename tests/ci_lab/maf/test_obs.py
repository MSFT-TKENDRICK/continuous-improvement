from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent_framework import Agent
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab.contracts import ATTR_ROLLOUT, ATTR_VARIANT, SPAN_CASE, RolloutKey
from ci_lab.maf import build_workflow, record_rollout, run_or_resume
from ci_lab.testing import FakeChatClient

WORKFLOW = """\
kind: Workflow
trigger:
  kind: OnConversationStart
  id: traced_wf
  actions:
    - {kind: InvokeFunctionTool, id: prep, functionName: prep, arguments: {}}
    - {kind: InvokeAzureAgent, id: think, agent: {name: Thinker}, input: {messages: "go"}}
"""


def _trace_id() -> int:
    return trace.get_current_span().get_span_context().trace_id


def test_run_or_resume_propagates_context_and_tags_span(tmp_path: Path) -> None:
    # A local provider (never the global one, which ci_lab.telemetry owns).
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    seen: dict[str, int] = {}

    def prep() -> str:
        seen["tool"] = _trace_id()
        return "prepped"

    def respond(messages: Any, options: Any) -> str:
        seen["agent"] = _trace_id()
        return "thought"

    yaml_path = tmp_path / "wf.yaml"
    yaml_path.write_text(WORKFLOW, encoding="utf-8")
    agent = Agent(client=FakeChatClient([respond]), name="Thinker", instructions="Think.")
    rollout = RolloutKey(experiment_id="exp", variant="B", case_id="c1", trial=2)

    async def go() -> list[Any]:
        with tracer.start_as_current_span(SPAN_CASE) as case:
            outputs = await run_or_resume(yaml_path, "start", agents={"Thinker": agent}, tools={"prep": prep},
                                          checkpoint_dir=tmp_path / "ckpt", rollout=rollout)
            seen["case"] = case.get_span_context().trace_id
        return outputs

    assert asyncio.run(go()) == ["prepped", "thought"]
    assert seen["tool"] == seen["agent"] == seen["case"] != 0
    [case_span] = [s for s in exporter.get_finished_spans() if s.name == SPAN_CASE]
    assert case_span.attributes[ATTR_ROLLOUT] == rollout.rollout_id
    assert case_span.attributes[ATTR_VARIANT] == "B"


def test_record_rollout_is_noop_without_recording_span() -> None:
    record_rollout(RolloutKey(experiment_id="e", variant="A", case_id="c"), extra=None)


def test_resume_continues_stored_trace_context(tmp_path: Path) -> None:
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    seen: list[int] = []

    def prep() -> str:
        seen.append(_trace_id())
        return "prepped"

    yaml_path = tmp_path / "wf.yaml"
    yaml_path.write_text(WORKFLOW.replace("functionName: prep, arguments: {}}",
                                          "functionName: prep, arguments: {}}\n"
                                          "    - {kind: InvokeFunctionTool, id: again, functionName: prep, "
                                          "arguments: {}}"), encoding="utf-8")
    ckpt = tmp_path / "ckpt"

    def kwargs() -> dict[str, Any]:
        agent = Agent(client=FakeChatClient(["thought"]), name="Thinker", instructions="Think.")
        return {"agents": {"Thinker": agent}, "tools": {"prep": prep}, "checkpoint_dir": ckpt}

    async def crash_after_first_tool() -> None:
        storage = build_workflow(yaml_path, **kwargs())[1]
        for cp in await storage.list_checkpoints(workflow_name="traced_wf"):
            if cp.iteration_count > 2:
                await storage.delete(cp.checkpoint_id)

    async def go() -> None:
        with tracer.start_as_current_span("first") as first:
            await run_or_resume(yaml_path, "start", **kwargs())
        original = first.get_span_context().trace_id
        assert seen == [original, original]
        assert list(ckpt.glob("otel-carrier-*.w3c"))

        # Resume with no active span: continue the stored trace.
        await crash_after_first_tool()
        seen.clear()
        await run_or_resume(yaml_path, "ignored", **kwargs())
        assert seen == [original]

        # Resume under the caller's own span: the caller's context wins.
        await crash_after_first_tool()
        seen.clear()
        with tracer.start_as_current_span("resumed") as resumed:
            await run_or_resume(yaml_path, "ignored", **kwargs())
        assert seen == [resumed.get_span_context().trace_id] != [original]

    asyncio.run(go())


def test_carrier_file_is_sanitised(tmp_path: Path) -> None:
    from ci_lab.maf.workflows import _read_carrier

    p = tmp_path / "c.w3c"
    p.write_text('{"traceparent": "00-ab-cd-01", "authorization": "secret"}', encoding="utf-8")
    assert _read_carrier(p) == {"traceparent": "00-ab-cd-01"}
    p.write_text('["x"]', encoding="utf-8")
    assert _read_carrier(p) is None
    assert _read_carrier(tmp_path / "missing") is None
