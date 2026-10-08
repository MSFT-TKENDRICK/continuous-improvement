import json
import subprocess
import sys

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.contracts import ATTR_EXPERIMENT, SPAN_ARM, SPAN_CAMPAIGN_ROUND


@pytest.fixture
def exporter(monkeypatch):
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    return exp


def test_noop_without_provider(tmp_path):
    with obs.span(SPAN_CAMPAIGN_ROUND, {ATTR_EXPERIMENT: "e1"}):
        pass  # global provider is the no-op proxy in tests


def test_span_nesting_and_attr_cleaning(exporter):
    with obs.span(SPAN_CAMPAIGN_ROUND, {ATTR_EXPERIMENT: "e1", "x": None, "p": ("a",)}):
        with obs.span(SPAN_ARM, {"ci.arm": "a1"}):
            ids = obs.current_ids()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans[SPAN_ARM].parent.span_id == spans[SPAN_CAMPAIGN_ROUND].context.span_id
    assert "x" not in spans[SPAN_CAMPAIGN_ROUND].attributes
    assert spans[SPAN_CAMPAIGN_ROUND].attributes["p"] == "('a',)"
    assert ids[0] == format(spans[SPAN_ARM].context.trace_id, "032x")


def test_exception_recorded(exporter):
    with pytest.raises(ValueError):
        with obs.span("ci.step"):
            raise ValueError("boom")
    (s,) = exporter.get_finished_spans()
    assert s.status.status_code == trace.StatusCode.ERROR


def test_link_to(exporter):
    with obs.span("ci.round", links=[obs.link_to("ab" * 16, "cd" * 8)]):
        pass
    (s,) = exporter.get_finished_spans()
    assert format(s.links[0].context.trace_id, "032x") == "ab" * 16


def test_child_env_propagates_traceparent(exporter, tmp_path):
    tp = TracerProvider()
    with tp.get_tracer("t").start_as_current_span("parent") as p:
        env = obs.child_env({})
        tid = format(p.get_span_context().trace_id, "032x")
    assert env[obs.TRACEPARENT_ENV].split("-")[1] == tid
    code = ("from ci_lab import obs; from opentelemetry import trace; obs.attach_from_env();"
            "print(format(trace.get_current_span().get_span_context().trace_id,'032x'))")
    out = subprocess.run([sys.executable, "-c", code], env={**_base_env(), **env},
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == tid


def _base_env():
    import os
    return {k: v for k, v in os.environ.items() if k != obs.TRACEPARENT_ENV}


def test_write_status_merges(tmp_path):
    p = obs.write_status(tmp_path, "exp1", phase="propose", arms={"a1": {"strategy": "gepa"}})
    obs.write_status(tmp_path, "exp1", phase="evaluate", arms={"a1": {"state": "running"},
                                                               "a2": {"strategy": "agent"}})
    d = json.loads(p.read_text())
    assert d["phase"] == "evaluate" and d["experiment_id"] == "exp1"
    assert d["arms"]["a1"] == {"strategy": "gepa", "state": "running", "updated": d["arms"]["a1"]["updated"]}
    assert set(d["arms"]) == {"a1", "a2"}
    assert not list(p.parent.glob(".status-*"))
