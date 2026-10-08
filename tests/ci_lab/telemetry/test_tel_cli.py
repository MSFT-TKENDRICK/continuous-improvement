"""CLI: ``python -m ci_lab.telemetry`` / ``register()`` — JSON output, no secrets leaked."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from ci_lab.telemetry import aspire, cli, importer, pull

FIXTURES = Path(__file__).with_name("fixtures")
STATE = {"pid": 1, "ui_url": "http://127.0.0.1:5", "api_url": "http://127.0.0.1:5",
         "otlp_url": "http://127.0.0.1:6", "browser_token": "BTOKEN", "otlp_key": "OKEY",
         "api_key": "AKEY"}


def _run(capsys, *argv):
    rc = cli.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_register_adds_both_command_groups():
    p = argparse.ArgumentParser()
    cli.register(p.add_subparsers(dest="command"))
    assert p.parse_args(["dashboard", "url", "--with-token"]).with_token
    assert p.parse_args(["telemetry", "pull", "--run", "1"]).run == "1"
    with pytest.raises(SystemExit):
        p.parse_args(["dashboard", "up", "--rid", "bogus"])


def test_status_and_url_never_leak_secrets(capsys, monkeypatch):
    monkeypatch.setattr(aspire, "status", lambda *a, **k: {**aspire.public(STATE), "running": True})
    monkeypatch.setattr(aspire, "live_state", lambda *a, **k: dict(STATE))
    for argv in (["dashboard", "status"], ["dashboard", "url"]):
        rc, out, _ = _run(capsys, *argv)
        assert rc == 0 and not any(s in out for s in ("BTOKEN", "OKEY", "AKEY")), out
    rc, out, _ = _run(capsys, "dashboard", "url", "--with-token")
    assert out.strip() == "http://127.0.0.1:5/login?t=BTOKEN"


def test_url_and_open_when_not_running(capsys, monkeypatch):
    monkeypatch.setattr(aspire, "live_state", lambda *a, **k: None)
    assert _run(capsys, "dashboard", "url")[0] == 1
    assert _run(capsys, "dashboard", "open")[0] == 1


def test_open_uses_login_url(capsys, monkeypatch):
    monkeypatch.setattr(aspire, "live_state", lambda *a, **k: dict(STATE))
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    rc, out, _ = _run(capsys, "dashboard", "open")
    assert rc == 0 and opened == ["http://127.0.0.1:5/login?t=BTOKEN"] and "BTOKEN" not in out


def test_up_down_error_paths(capsys, monkeypatch):
    def boom(**kw):
        raise aspire.UntrustedPackageError("no pin; --trust-new")

    monkeypatch.setattr(aspire, "up", boom)
    rc, _, err = _run(capsys, "dashboard", "up", "--rid", "linux-x64")
    assert rc == 1 and "--trust-new" in err
    monkeypatch.setattr(aspire, "down", lambda *a: {"stopped": False, "reason": "not running"})
    assert _run(capsys, "dashboard", "down")[0] == 0
    monkeypatch.setattr(aspire, "down", lambda *a: {"stopped": False, "reason": "pid 3 did not exit"})
    assert _run(capsys, "dashboard", "down")[0] == 1


def test_import_command(capsys, monkeypatch, tmp_path):
    shutil.copy(FIXTURES / "span_v1.jsonl", tmp_path / "spans-1.jsonl")
    seen = {}

    def fake(records, **kw):
        seen.update(kw, n=len(list(records)))
        return {"spans": 2, "batches": 1}

    monkeypatch.setattr(importer, "import_records", fake)
    rc, out, _ = _run(capsys, "telemetry", "import", str(tmp_path), "--otlp-url", "http://h:1",
                      "--otlp-key", "k")
    assert rc == 0 and json.loads(out)["files"] == 1 and seen["n"] == 2 and seen["redact"] is True
    bad = tmp_path / "spans-2.jsonl"
    bad.write_text('{"schemaVersion": 5}\n', encoding="utf-8")
    rc, _, err = _run(capsys, "telemetry", "import", str(bad), "--otlp-url", "http://h:1")
    assert rc == 1 and "schemaVersion" in err


def test_pull_command_caches_when_no_dashboard(capsys, monkeypatch):
    monkeypatch.setattr(pull, "pull", lambda *a, **k: {"run_id": "9", "files": [], "cached": False})
    monkeypatch.setattr(aspire, "live_state", lambda *a, **k: None)
    rc, out, _ = _run(capsys, "telemetry", "pull", "--run", "9")
    res = json.loads(out)
    assert rc == 0 and res["imported"] is None and "dashboard up" in res["note"]
    monkeypatch.setattr(pull, "list_imports", lambda: [{"run_id": "9"}])
    assert json.loads(_run(capsys, "telemetry", "imports")[1]) == [{"run_id": "9"}]


def test_pull_command_imports_into_live_dashboard(capsys, monkeypatch):
    monkeypatch.setattr(pull, "pull", lambda *a, **k: {"run_id": "9", "files": ["f.jsonl"]})
    monkeypatch.setattr(aspire, "live_state", lambda *a, **k: dict(STATE))
    monkeypatch.setattr(importer, "import_files", lambda paths, **k: {"spans": 3, "files": len(paths)})
    rc, out, _ = _run(capsys, "telemetry", "pull", "--run", "9")
    assert rc == 0 and json.loads(out)["imported"] == {"spans": 3, "files": 1}
