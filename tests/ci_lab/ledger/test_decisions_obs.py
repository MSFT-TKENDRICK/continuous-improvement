from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.contracts import ATTR_DECISION, ATTR_EXPERIMENT, ATTR_PHASE, ATTR_ROUND, SPAN_STEP
from ci_lab.ledger import Frontier, Layout, cas_frontier, read_decisions, record_decisions


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(obs, "tracer", lambda: provider.get_tracer("test"))  # no global provider
    return exporter


def test_record_decisions_writes_and_traces(tmp_path, spans):
    p = record_decisions(tmp_path, "cmp-a", "cmp-a-r001", {"decision": "ship", "winner": "b"})
    assert p == Layout(tmp_path).decisions_json("cmp-a", "cmp-a-r001")
    assert read_decisions(tmp_path, "cmp-a", "cmp-a-r001") == {"decision": "ship", "winner": "b"}
    (s,) = spans.get_finished_spans()
    assert s.name == SPAN_STEP
    assert s.attributes[ATTR_PHASE] == "record"
    assert s.attributes[ATTR_EXPERIMENT] == "cmp-a-r001"
    assert s.attributes[ATTR_DECISION] == "ship"


def test_record_decisions_validates(tmp_path):
    with pytest.raises(ValueError):
        record_decisions(tmp_path, "cmp-a", "cmp-a-r001", {"decision": "maybe"})
    assert read_decisions(tmp_path, "cmp-a", "cmp-a-r001") is None


def test_record_decisions_noop_without_provider(tmp_path):
    record_decisions(tmp_path, "cmp-a", "cmp-a-r002", {"decision": "rerun"})
    assert obs.current_ids() is None


def test_cas_frontier_traces_experiment(tmp_path, spans):
    cas_frontier(tmp_path / "frontier.json", None, Frontier("c" * 40, "t" * 40, 0.5, 1),
                 experiment_id="cmp-a-r001")
    (s,) = spans.get_finished_spans()
    assert s.attributes[ATTR_EXPERIMENT] == "cmp-a-r001" and s.attributes[ATTR_ROUND] == 1
