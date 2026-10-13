from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanKind, Status, StatusCode

FIXTURES = Path(__file__).with_name("fixtures")


def _spans():
    """Finished SDK spans from a *local* provider (never touches the global one)."""
    mem = InMemorySpanExporter()
    tp = TracerProvider(resource=Resource.create({"service.name": "ci-lab.test", "ci.profile": "fake"}))
    tp.add_span_processor(SimpleSpanProcessor(mem))
    t = tp.get_tracer("ci_lab", "9.9")
    with t.start_as_current_span("ci.round", attributes={"ci.round": 2, "ci.ok": True}) as root:
        root.add_event("note", {"k": "v"})
        with t.start_as_current_span(
                "chat gpt-5", kind=SpanKind.CLIENT,
                links=[Link(root.get_span_context(), {"ci.link": "x"})],
                attributes={"gen_ai.request.model": "gpt-5",
                            "gen_ai.input.messages": "[{\"role\":\"user\",\"content\":\"secret\"}]",
                            "gen_ai.output.messages": "secret-out",
                            "gen_ai.system_instructions": "sys",
                            "gen_ai.prompt.0.content": "old-style",
                            "gen_ai.usage.input_tokens": 7}) as child:
            child.add_event("gen_ai.content.prompt", {"gen_ai.prompt": "secret"})
            child.add_event("retry", {"attempt": 1})
            child.set_status(Status(StatusCode.ERROR, "boom"))
    tp.shutdown()
    return list(mem.get_finished_spans())


@pytest.fixture
def sdk_spans():
    return _spans()
