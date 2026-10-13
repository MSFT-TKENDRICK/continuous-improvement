"""End-to-end against a real Aspire dashboard (deselected by default; run with ``-m live``).

``dashboard up`` -> ``setup()`` + harness span in a child process -> query the dashboard API
-> replay the JSONL via the importer -> ``dashboard down``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ci_lab.telemetry import aspire, importer, record

ROOT = Path(__file__).resolve().parents[3]


def _wait_trace(trace_id: str, st: dict, timeout: float = 20.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            recs = record.from_otlp_json(aspire.query(f"traces/{trace_id}", st))
        except aspire.DashboardError:
            recs = []
        if recs:
            return recs
        time.sleep(0.5)
    raise AssertionError(f"trace {trace_id} not visible in dashboard")


@pytest.mark.live
def test_dashboard_end_to_end(tmp_path, monkeypatch):
    sp = tmp_path / "dashboard.json"
    monkeypatch.setenv("CI_DASHBOARD_STATE", str(sp))
    up = aspire.up(path=sp, wait=90)
    try:
        assert up["running"] and not set(aspire.SECRET_FIELDS) & set(up)
        st = aspire.live_state(sp)
        assert st is not None
        rd = tmp_path / "run"
        env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
        code = textwrap.dedent(f"""
            import json
            from ci_lab import obs, telemetry
            h = telemetry.setup("live", profile="fake", run_dir=r"{rd}", aspire="on")
            with obs.span("ci.round", {{"ci.round": 1}}):
                with obs.span("ci.arm"):
                    ids = obs.current_ids()
            telemetry.shutdown()
            print(json.dumps(ids))
        """)
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=False, capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        trace_id, span_id = json.loads(r.stdout.strip().splitlines()[-1])

        live = _wait_trace(trace_id, st)
        assert {x["name"] for x in live} == {"ci.round", "ci.arm"}
        arm = next(x for x in live if x["name"] == "ci.arm")
        assert arm["spanId"] == span_id and arm["resource"]["service.name"] == "ci-lab.live"
        names = [x.get("name") for x in aspire.query("resources", st)]
        assert "ci-lab.live" in names

        # Replay the JSONL under a new trace id (same timestamps) and read it back.
        src = next((rd / "telemetry").glob("spans-*.jsonl"))
        lines = src.read_text("utf-8").replace(trace_id, "f" * 32)
        replay = tmp_path / "replay" / "spans-1.jsonl"
        replay.parent.mkdir()
        replay.write_text(lines, encoding="utf-8")
        assert importer.import_files([replay.parent])["spans"] == 2
        back = {x["spanId"]: x for x in _wait_trace("f" * 32, st)}
        orig = {x["spanId"]: x for x in live}
        assert set(back) == set(orig)
        for sid, x in back.items():
            assert x["startTimeUnixNano"] == orig[sid]["startTimeUnixNano"]
            assert x["parentSpanId"] == orig[sid]["parentSpanId"]
    finally:
        res = aspire.down(sp)
    assert res["stopped"] is True
    assert not aspire.pid_alive(up["pid"])