"""Importer: JSONL -> OTLP/HTTP protobuf against a local capture server."""
from __future__ import annotations

import http.server
import shutil
import threading
from pathlib import Path

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from ci_lab.telemetry import importer, record
from ci_lab.telemetry.jsonl import read_jsonl

FIXTURES = Path(__file__).with_name("fixtures")


@pytest.fixture
def otlp_server():
    got: list[tuple[str, dict, bytes]] = []
    status = {"code": 200}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.path, dict(self.headers), body))
            self.send_response(status["code"])
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", got, status
    srv.shutdown()


def test_import_preserves_ids_and_times(tmp_path, otlp_server):
    url, got, _ = otlp_server
    d = tmp_path / "run" / "telemetry"
    d.mkdir(parents=True)
    shutil.copy(FIXTURES / "span_v1.jsonl", d / "spans-1.jsonl")
    (d / "other.jsonl").write_text("ignored", encoding="utf-8")
    res = importer.import_files([tmp_path / "run"], otlp_url=url, otlp_key="k", batch=1)
    assert res == {"spans": 2, "batches": 2, "files": 1, "skipped_lines": 0}
    assert all(p == "/v1/traces" for p, _, _ in got)
    hdrs = {k.lower(): v for k, v in got[0][1].items()}
    assert hdrs["x-otlp-api-key"] == "k" and hdrs["content-type"] == "application/x-protobuf"
    sent = [r for _, _, b in got for r in record.from_otlp_request(ExportTraceServiceRequest.FromString(b))]
    golden, _ = read_jsonl([FIXTURES / "span_v1.jsonl"])
    assert sorted(sent, key=lambda r: r["spanId"]) == sorted(golden, key=lambda r: r["spanId"])


def test_import_redacts_by_default(tmp_path, otlp_server):
    url, got, _ = otlp_server
    golden, _ = read_jsonl([FIXTURES / "span_v1.jsonl"])
    golden[1]["attributes"]["gen_ai.input.messages"] = "secret"
    golden[1]["events"] = [{"name": "gen_ai.choice", "timeUnixNano": "1", "attributes": {}}]
    importer.import_records(golden, otlp_url=url)
    assert b"secret" not in got[0][2] and b"gen_ai.choice" not in got[0][2]
    importer.import_records(golden, otlp_url=url, redact=False)
    assert b"secret" in got[1][2]


def test_import_http_error_raises(otlp_server):
    url, _, status = otlp_server
    status["code"] = 401
    golden, _ = read_jsonl([FIXTURES / "span_v1.jsonl"])
    with pytest.raises(importer.TelemetryImportError, match="401"):
        importer.import_records(golden, otlp_url=url)


def test_import_without_dashboard_raises(monkeypatch):
    monkeypatch.setattr(importer._aspire, "live_state", lambda *a, **k: None)
    with pytest.raises(importer.TelemetryImportError, match="dashboard"):
        importer.import_records([], otlp_url=None)


def test_import_defaults_to_live_dashboard(monkeypatch, otlp_server):
    url, got, _ = otlp_server
    monkeypatch.setattr(importer._aspire, "live_state",
                        lambda *a, **k: {"otlp_url": url, "otlp_key": "live-key"})
    golden, _ = read_jsonl([FIXTURES / "span_v1.jsonl"])
    assert importer.import_records(golden)["spans"] == 2
    assert {k.lower(): v for k, v in got[0][1].items()}["x-otlp-api-key"] == "live-key"


def test_loopback_detection():
    assert importer._is_loopback("http://127.0.0.1:1") and importer._is_loopback("http://localhost:2")
    assert importer._is_loopback("http://[::1]:3")
    assert not importer._is_loopback("https://example.com")
