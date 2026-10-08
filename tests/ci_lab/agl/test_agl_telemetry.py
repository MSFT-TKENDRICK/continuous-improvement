"""Telemetry: one ``ci.case`` span per rollout; C28 seam. Uses a local SDK provider (never global)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.agl.journal import FileRolloutJournal
from ci_lab.agl.scope import RolloutScope
from ci_lab.agl.tracing import attach_telemetry
from ci_lab.contracts import (
    ATTR_ATTEMPT,
    ATTR_CASE,
    ATTR_ROLLOUT,
    ATTR_SCORE,
    ATTR_SPLIT,
    ATTR_TRIAL,
    ATTR_VARIANT,
    SPAN_CASE,
    SPAN_STEP,
    RolloutKey,
)

KEY = RolloutKey("camp-r00", "base", "case-1", trial=2, attempt=1)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(obs, "tracer", lambda: provider.get_tracer(obs.TRACER_NAME))
    return exporter


def test_rollout_is_one_case_span(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    j = FileRolloutJournal(tmp_path)
    with obs.span(SPAN_STEP) as parent, RolloutScope(j, KEY, {"split": "evolve"}) as scope:
        assert trace.get_current_span() is scope.span
        scope.reward(0.5)
    (case, step) = spans.get_finished_spans()
    assert case.name == SPAN_CASE and case.parent is not None
    assert case.parent.span_id == parent.get_span_context().span_id
    attrs = dict(case.attributes or {})
    assert attrs[ATTR_ROLLOUT] == KEY.rollout_id and attrs[ATTR_ATTEMPT] == "1"
    assert attrs[ATTR_CASE] == "case-1" and attrs[ATTR_TRIAL] == 2 and attrs[ATTR_VARIANT] == "base"
    assert attrs[ATTR_SPLIT] == "evolve" and attrs[ATTR_SCORE] == 0.5
    assert case.status.status_code == trace.StatusCode.UNSET
    assert step.name == SPAN_STEP


def test_failure_marks_span_error(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    j = FileRolloutJournal(tmp_path)
    with pytest.raises(ValueError), RolloutScope(j, KEY):
        raise ValueError("x")
    with RolloutScope(j, RolloutKey("e", "v", "c2")) as s:
        s.fail()
    errs = [sp.status.status_code for sp in spans.get_finished_spans()]
    assert errs == [trace.StatusCode.ERROR, trace.StatusCode.ERROR]


def test_reuses_callers_case_span(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    with obs.span(SPAN_CASE, {ATTR_CASE: "case-1"}) as outer, RolloutScope(FileRolloutJournal(tmp_path), KEY) as s:
        assert s.span is outer
    (only,) = spans.get_finished_spans()
    assert dict(only.attributes or {})[ATTR_ROLLOUT] == KEY.rollout_id


def test_async_scope_span_and_context_propagation(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    j = FileRolloutJournal(tmp_path)

    async def child() -> str:
        return trace.get_current_span().get_span_context().span_id.__format__("016x")

    async def main() -> tuple[str, str]:
        async with RolloutScope(j, KEY) as scope:
            inner = await asyncio.create_task(child())  # tasks inherit the OTel context
            return inner, format(scope.span.get_span_context().span_id, "016x")

    inner, mine = asyncio.run(main())
    assert inner == mine and [s.name for s in spans.get_finished_spans()] == [SPAN_CASE]


def test_noop_without_provider(tmp_path: Path) -> None:
    with RolloutScope(FileRolloutJournal(tmp_path), KEY) as s:
        s.reward(1.0)
    assert not s.span.is_recording()


def test_attach_telemetry_seam() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    assert attach_telemetry(provider, [SimpleSpanProcessor(exporter)]) == 1
    assert attach_telemetry(provider, [SimpleSpanProcessor(exporter)]) == 0  # idempotent
    with provider.get_tracer("agl").start_as_current_span("x"):
        pass
    assert len(exporter.get_finished_spans()) == 1
    assert attach_telemetry(None) == 0
    assert attach_telemetry(TracerProvider(), lambda: []) == 0


def test_attach_telemetry_defaults_to_telemetry_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    exporter = InMemorySpanExporter()
    fake = types.ModuleType("ci_lab.telemetry")
    fake.span_processors = lambda: [SimpleSpanProcessor(exporter)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ci_lab.telemetry", fake)
    provider = TracerProvider()
    assert attach_telemetry(provider) == 1
    with provider.get_tracer("agl").start_as_current_span("y"):
        pass
    assert [s.name for s in exporter.get_finished_spans()] == ["y"]
