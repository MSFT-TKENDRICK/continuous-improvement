"""Builders annotate the caller's current span with the decision (design §12.3)."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab.contracts import (
    ATTR_CAMPAIGN,
    ATTR_DECISION,
    ATTR_EXPERIMENT,
    ATTR_NIGHT,
    ATTR_ROUND,
    ATTR_VARIANT,
    SPAN_CAMPAIGN_ROUND,
)


@pytest.fixture
def tracer():
    # Local provider: never touch the global one (owned by ci_lab.telemetry.setup, C28).
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


@pytest.mark.parametrize(("kind", "expected"), [
    ("round", {ATTR_EXPERIMENT: "tone-a1-r01", ATTR_DECISION: "ship", ATTR_CAMPAIGN: "tone-a1", ATTR_ROUND: 1,
               ATTR_VARIANT: "v1"}),
    ("calibration", {ATTR_EXPERIMENT: "tone-a1-cal", ATTR_DECISION: "do_not_ship", ATTR_ROUND: 0}),
    ("confirm", {ATTR_EXPERIMENT: "tone-a1-confirm", ATTR_DECISION: "ship", ATTR_VARIANT: "final"}),
    ("sleep", {ATTR_EXPERIMENT: "sleep-20261008", ATTR_DECISION: "ship", ATTR_NIGHT: "2026-10-08",
               ATTR_VARIANT: "candidate"}),
])
def test_decision_attributes_on_current_span(fx, tracer, kind, expected):
    t, exporter = tracer
    with t.start_as_current_span(SPAN_CAMPAIGN_ROUND):
        doc = fx.BUILDERS[kind]()
    (span,) = exporter.get_finished_spans()
    assert {k: span.attributes[k] for k in expected} == expected
    assert doc == fx.BUILDERS[kind]()  # annotation does not change the envelope


def test_no_variant_attribute_without_ship(fx, tracer):
    t, exporter = tracer
    with t.start_as_current_span(SPAN_CAMPAIGN_ROUND):
        fx.build_round(winner=None)
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs[ATTR_DECISION] == "do_not_ship" and ATTR_VARIANT not in attrs


def test_no_span_is_a_noop(fx):
    assert fx.build_round()["decision"]["outcome"] == "ship"
