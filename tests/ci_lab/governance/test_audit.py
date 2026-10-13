from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab.governance import audit as ga

DID = "did:mesh:0123456789abcdef0123456789abcdef"
H1, H2 = "a" * 64, "b" * 64
TS = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def rec(decision="allow", **kw):
    base = {"agent_did": DID, "intervention_point": "pre_tool_call", "decision": decision,
            "reason": "ok", "input_identity": H1, "enforced_identity": H2, "run": "run-1",
            "eid": "e1", "ts": TS}
    return {**base, **kw}


def test_append_verify_head_and_oes_extension(tmp_path):
    trail = ga.AuditTrail(tmp_path / "a.jsonl")
    assert trail.head() is None and trail.verify().ok
    e1 = trail.append(rec())
    e2 = trail.append(ga.DecisionRecord(**rec("deny", reason="host_error:approval_unresolved")))
    trail.append(rec("transform", intervention_point="output", mode="evaluate_only"))
    assert e2.previous_hash == e1.entry_hash
    res = ga.verify(trail.path)
    assert res.ok and res.entries == 3 and res.head == trail.head() and res.merkle_root
    assert trail.oes_extension() == {"x-ci-governance": {"audit_head": res.head, "decisions": 3,
                                                          "denies": 1}}
    line = json.loads(trail.path.read_text(encoding="utf-8").splitlines()[1])
    assert line["action"] == "pre_tool_call" and line["outcome"] == "denied"
    assert set(line["data"]) == {"reason", "input_identity", "enforced_identity", "mode", "run",
                                 "eid"}


@pytest.mark.parametrize("field", ["prompt", "content", "arguments", "snapshot", "messages"])
def test_content_fields_rejected(tmp_path, field):
    with pytest.raises(ga.AuditError, match="content-free"):
        ga.AuditTrail(tmp_path / "a.jsonl").append({**rec(), field: "secret customer text"})
    assert not (tmp_path / "a.jsonl").exists()


@pytest.mark.parametrize("bad", [
    {"reason": "line one\nline two"}, {"reason": "ok\n"}, {"reason": "x" * 500},
    {"input_identity": "not-a-hash"}, {"decision": "maybe"}, {"intervention_point": "anywhere"},
    {"mode": "permissive"}, {"agent_did": "alice"}, {"run": "has space"},
    {"ts": datetime(2026, 1, 1)},  # noqa: DTZ001 - naive on purpose
])
def test_invalid_fields_rejected(bad):
    with pytest.raises(ga.AuditError):
        ga.DecisionRecord(**rec(**bad))


def test_tamper_detected(tmp_path):
    trail = ga.AuditTrail(tmp_path / "a.jsonl")
    for d in ("allow", "deny", "allow"):
        trail.append(rec(d))
    lines = trail.path.read_text(encoding="utf-8").splitlines()
    doc = json.loads(lines[1])
    doc["policy_decision"], doc["outcome"], doc["data"]["reason"] = "allow", "success", "ok2"
    trail.path.write_text("\n".join([lines[0], json.dumps(doc), lines[2]]) + "\n", "utf-8")
    res = trail.verify()
    assert not res.ok and "entry 1 hash mismatch" in res.error
    with pytest.raises(ga.AuditError):
        trail.oes_extension()
    trail.path.write_text("\n".join([lines[0], lines[2]]) + "\n", "utf-8")
    assert "chain broken" in ga.verify(trail.path).error
    trail.path.write_text("{not json\n", "utf-8")
    assert not ga.verify(trail.path).ok


def test_span_event_has_no_content(tmp_path):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("t").start_as_current_span("s"):
        entry = ga.AuditTrail(tmp_path / "a.jsonl").append(rec("deny", eid=None))
    (event,) = exporter.get_finished_spans()[0].events
    assert event.name == ga.EVENT
    assert event.attributes["ci.governance.decision"] == "deny"
    assert event.attributes["ci.governance.audit_hash"] == entry.entry_hash
    assert "ci.governance.eid" not in event.attributes
    assert all(k.startswith("ci.governance.") for k in event.attributes)
