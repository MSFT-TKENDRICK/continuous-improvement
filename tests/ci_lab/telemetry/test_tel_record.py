"""D7: versioned span record, adapters, golden fixtures."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from ci_lab.contracts import SPAN_SCHEMA_VERSION
from ci_lab.telemetry import record
from ci_lab.telemetry.jsonl import read_jsonl
from ci_lab.telemetry.record import SchemaError

FIXTURES = Path(__file__).with_name("fixtures")


def _by_id(recs):
    return sorted(recs, key=lambda r: r["spanId"])


def _golden():
    recs, bad = read_jsonl([FIXTURES / "span_v1.jsonl"])
    assert bad == 0
    return recs


def test_golden_jsonl_is_v1_and_normalized():
    recs = _golden()
    assert len(recs) == 2 and SPAN_SCHEMA_VERSION == 1
    raw = [json.loads(x) for x in (FIXTURES / "span_v1.jsonl").read_text("utf-8").splitlines()]
    assert recs == raw  # golden file is already in canonical form


def test_aspire_api_fixture_matches_golden():
    payload = json.loads((FIXTURES / "aspire_trace.json").read_text("utf-8"))
    assert _by_id(record.from_otlp_json(payload)) == _by_id(_golden())


def test_otlp_protobuf_roundtrip_preserves_everything():
    golden = _golden()
    req = record.to_otlp_request(golden)
    assert len(req.resource_spans) == 1 and len(req.resource_spans[0].scope_spans) == 2
    wire = type(req).FromString(req.SerializeToString())
    assert _by_id(record.from_otlp_request(wire)) == _by_id(golden)


def test_readable_span_adapter(sdk_spans):
    recs = {s.name: record.validate(record.from_readable_span(s)) for s in sdk_spans}
    root, child = recs["ci.round"], recs["chat gpt-5"]
    assert root["kind"] == 1 and child["kind"] == 3  # SDK INTERNAL/CLIENT -> OTLP 1/3
    assert child["parentSpanId"] == root["spanId"] and child["traceId"] == root["traceId"]
    assert child["status"] == {"code": 2, "message": "boom"}
    assert child["links"][0]["spanId"] == root["spanId"]
    assert root["attributes"] == {"ci.round": 2, "ci.ok": True}
    assert root["resource"]["service.name"] == "ci-lab.test"
    assert root["scope"] == {"name": "ci_lab", "version": "9.9"}
    assert isinstance(root["startTimeUnixNano"], str) and int(root["endTimeUnixNano"]) >= int(root["startTimeUnixNano"])
    assert _by_id(record.from_otlp_request(record.to_otlp_request(recs.values()))) == _by_id(recs.values())


def test_otlp_json_tolerates_enum_names_base64_ids_and_kvlist():
    tid, sid = bytes(range(16)), bytes(range(8))
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": [{
        "traceId": base64.b64encode(tid).decode(), "spanId": base64.b64encode(sid).decode(),
        "name": "x", "kind": "SPAN_KIND_SERVER", "startTimeUnixNano": 1, "endTimeUnixNano": 2,
        "status": {"code": "STATUS_CODE_ERROR"},
        "attributes": [{"key": "m", "value": {"kvlistValue": {"values": [
            {"key": "a", "value": {"intValue": "1"}}]}}}]}]}]}]}
    (r,) = record.from_otlp_json(payload)
    assert r["traceId"] == tid.hex() and r["spanId"] == sid.hex()
    assert r["kind"] == 2 and r["status"]["code"] == 2
    assert json.loads(r["attributes"]["m"]) == {"a": 1}


@pytest.mark.parametrize("ver", [None, 0, 2, "1"])
def test_unknown_schema_version_rejected(ver):
    rec = dict(_golden()[0])
    if ver is None:
        rec.pop("schemaVersion")
    else:
        rec["schemaVersion"] = ver
    with pytest.raises(SchemaError, match="schemaVersion"):
        record.validate(rec)


@pytest.mark.parametrize("field,value", [("traceId", "xyz"), ("spanId", "00" * 9),
                                         ("parentSpanId", "nothex!!nothex!!"),
                                         ("startTimeUnixNano", "soon")])
def test_bad_ids_and_times_rejected(field, value):
    rec = dict(_golden()[0], **{field: value})
    with pytest.raises(SchemaError):
        record.validate(rec)


def test_read_jsonl_skips_torn_lines_but_rejects_wrong_version(tmp_path):
    good = (FIXTURES / "span_v1.jsonl").read_text("utf-8").splitlines()
    p = tmp_path / "spans-1.jsonl"
    p.write_text(good[0] + "\n\n" + good[1][:40] + "\n", encoding="utf-8")
    recs, bad = read_jsonl([p])
    assert len(recs) == 1 and bad == 1
    p.write_text(good[0].replace('"schemaVersion":1', '"schemaVersion":99') + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r"spans-1\.jsonl:1"):
        read_jsonl([p])
