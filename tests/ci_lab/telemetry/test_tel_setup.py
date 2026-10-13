"""setup(): provider ownership (C28/D4), idempotence, flush, exporters. Each case runs in a
fresh interpreter because the OTel global TracerProvider can only be set once per process."""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

from ci_lab.telemetry import aspire, record
from ci_lab.telemetry.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parents[3]


def run_py(code: str, tmp_path: Path, state: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env["CI_DASHBOARD_STATE"] = str(state or tmp_path / "no-dashboard.json")
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, env=env,
                       check=False, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _spans(run_dir: Path):
    recs, bad = read_jsonl(sorted((run_dir / "telemetry").glob("spans-*.jsonl")))
    assert bad == 0
    return recs


def test_setup_idempotent_flush_and_resource(tmp_path):
    rd = tmp_path / "run"
    out = run_py(f"""
        import json
        from opentelemetry import trace
        from agent_framework.observability import OBSERVABILITY_SETTINGS as S
        from ci_lab import obs, telemetry
        h = telemetry.setup("night", profile="fake", run_dir=r"{rd}", campaign_id="c9")
        h2 = telemetry.setup("other", run_dir=r"{tmp_path / 'ignored'}")
        with obs.span("ci.round", {{"ci.round": 1}}):
            with obs.span("ci.arm"):
                ids = obs.current_ids()
        telemetry.shutdown()
        telemetry.shutdown()
        print(json.dumps({{"same": h is h2, "global": trace.get_tracer_provider() is h.provider,
                           "owns": h.owns_provider, "ids": ids, "aspire": h.aspire,
                           "maf": [S.ENABLED, S.SENSITIVE_DATA_ENABLED],
                           "current": telemetry.current() is None}}))
    """, tmp_path)
    assert out["same"] and out["global"] and out["owns"] and not out["aspire"] and out["current"]
    assert out["maf"] == [True, False]
    assert not (tmp_path / "ignored").exists()
    recs = _spans(rd)
    assert {r["name"] for r in recs} == {"ci.round", "ci.arm"}
    arm = next(r for r in recs if r["name"] == "ci.arm")
    assert [arm["traceId"], arm["spanId"]] == out["ids"]
    res = arm["resource"]
    assert res["service.name"] == "ci-lab.night" and res["ci.profile"] == "fake"
    assert res["ci.campaign_id"] == "c9" and res["ci.telemetry.sensitive"] is False
    assert "vcs.ref" in res and res["process.pid"] > 0


def test_late_run_dir_attaches_jsonl(tmp_path):
    rd = tmp_path / "late"
    out = run_py(f"""
        import json
        from ci_lab import obs, telemetry
        h = telemetry.setup("svc")
        assert h.jsonl_path is None
        with obs.span("before"):
            pass
        h = telemetry.setup("svc", run_dir=r"{rd}")
        with obs.span("after"):
            pass
        telemetry.shutdown()
        print(json.dumps({{"path": str(h.jsonl_path)}}))
    """, tmp_path)
    assert {r["name"] for r in _spans(rd)} == {"after"}
    assert Path(out["path"]).parent == rd / "telemetry"


def test_fails_fast_on_foreign_non_sdk_provider(tmp_path):
    out = run_py("""
        import json
        from opentelemetry import trace
        from ci_lab import telemetry
        trace.set_tracer_provider(trace.NoOpTracerProvider())
        try:
            telemetry.setup("x")
            print(json.dumps({"raised": None}))
        except RuntimeError as exc:
            print(json.dumps({"raised": str(exc), "current": telemetry.current() is None}))
    """, tmp_path)
    assert "non-SDK" in out["raised"] and out["current"]


def test_attaches_to_existing_sdk_provider(tmp_path):
    """C28: e.g. an Agent Lightning tracer installed its provider first."""
    rd = tmp_path / "run"
    out = run_py(f"""
        import json
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from ci_lab import obs, telemetry
        mem = InMemorySpanExporter()
        tp = TracerProvider(); tp.add_span_processor(SimpleSpanProcessor(mem))
        trace.set_tracer_provider(tp)
        h = telemetry.setup("agl", run_dir=r"{rd}")
        with obs.span("ci.rollout"):
            pass
        telemetry.shutdown()
        print(json.dumps({{"owns": h.owns_provider, "same": h.provider is tp,
                           "mem": [s.name for s in mem.get_finished_spans()]}}))
    """, tmp_path)
    assert out == {"owns": False, "same": True, "mem": ["ci.rollout"]}
    assert [r["name"] for r in _spans(rd)] == ["ci.rollout"]


def test_span_processors_for_foreign_provider(tmp_path):
    rd = tmp_path / "run"
    run_py(f"""
        import json
        from opentelemetry.sdk.trace import TracerProvider
        from ci_lab import telemetry
        h = telemetry.setup("agl", run_dir=r"{rd}")
        agl = TracerProvider()  # a provider we do not own and that is not global
        for p in h.span_processors():
            agl.add_span_processor(p)
        with agl.get_tracer("agl").start_as_current_span("agl.rollout"):
            pass
        agl.force_flush()
        agl.shutdown()
        telemetry.shutdown()
        print("{{}}")
    """, tmp_path)
    assert [r["name"] for r in _spans(rd)] == ["agl.rollout"]


def test_aspire_modes_without_dashboard(tmp_path):
    out = run_py("""
        import json
        from ci_lab import telemetry
        try:
            telemetry.setup("x", aspire="on")
            on = None
        except RuntimeError as exc:
            on = str(exc)
        h = telemetry.setup("x", aspire="auto")
        print(json.dumps({"on": on, "auto": h.otlp_url}))
    """, tmp_path)
    assert "dashboard" in out["on"] and out["auto"] is None


def test_redaction_applies_to_all_exporters(tmp_path):
    rd = tmp_path / "run"
    run_py(f"""
        from opentelemetry import trace
        from ci_lab import telemetry
        telemetry.setup("x", run_dir=r"{rd}")
        with trace.get_tracer("t").start_as_current_span("chat") as s:
            s.set_attribute("gen_ai.input.messages", "TOPSECRET")
            s.set_attribute("gen_ai.request.model", "m")
            s.add_event("gen_ai.choice", {{"x": "TOPSECRET"}})
        telemetry.shutdown()
        print("{{}}")
    """, tmp_path)
    text = "".join(p.read_text("utf-8") for p in (rd / "telemetry").glob("*.jsonl"))
    assert "TOPSECRET" not in text and '"gen_ai.request.model":"m"' in text


@pytest.fixture
def fake_dashboard(tmp_path):
    posts: list[tuple[dict, bytes]] = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path == "/api/telemetry/resources" and self.headers.get("x-api-key") == "api"
            self.send_response(200 if ok else 401)
            self.end_headers()
            self.wfile.write(b"[]")

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            posts.append(({k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(200 if self.headers.get("x-otlp-api-key") == "otlp" else 401)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    state = aspire.write_state({"pid": os.getpid(), "api_url": url, "ui_url": url, "otlp_url": url,
                                "api_key": "api", "otlp_key": "otlp", "browser_token": "b"},
                               tmp_path / "dashboard.json")
    yield state, posts
    srv.shutdown()


def test_auto_exports_otlp_to_running_dashboard(tmp_path, fake_dashboard):
    state, posts = fake_dashboard
    out = run_py("""
        import json
        from opentelemetry import trace
        from ci_lab import obs, telemetry
        h = telemetry.setup("night", aspire="auto")
        with obs.span("ci.night"):
            with trace.get_tracer("t").start_as_current_span("chat") as s:
                s.set_attribute("gen_ai.output.messages", "TOPSECRET")
        telemetry.shutdown()
        print(json.dumps({"otlp": h.otlp_url, "aspire": h.aspire}))
    """, tmp_path, state=state)
    assert out["aspire"] and out["otlp"].startswith("http://127.0.0.1:")
    assert posts and all(h["x-otlp-api-key"] == "otlp" for h, _ in posts)
    assert all(h["content-type"] == "application/x-protobuf" for h, _ in posts)
    recs = [r for _, b in posts for r in record.from_otlp_request(ExportTraceServiceRequest.FromString(b))]
    assert {r["name"] for r in recs} == {"ci.night", "chat"}
    assert all(r["resource"]["service.name"] == "ci-lab.night" for r in recs)
    assert not any(b"TOPSECRET" in b for _, b in posts)