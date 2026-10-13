"""``python -m ci_lab.cli chat serve`` as a real process: listening line, HTTP, stdin-EOF shutdown."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

TOKEN = "proc-token-0123456789abcdef"


def _start(tmp_path: Path, *extra: str, token: str | None = TOKEN) -> subprocess.Popen[bytes]:
    env = {k: v for k, v in os.environ.items() if k != "CI_CHAT_TOKEN"}
    if token is not None:
        env["CI_CHAT_TOKEN"] = token
    argv = [sys.executable, "-m", "ci_lab.cli", "chat", "serve", "--profile", "fake", "--port", "0",
            "--run-dir", str(tmp_path / "runs"), "--chat-dir", str(tmp_path / "chat"), "--dry-run-launch", *extra]
    with (tmp_path / "stderr.log").open("wb") as err:
        return subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env,
                                cwd=Path.cwd())


def _readline(proc: subprocess.Popen[bytes], timeout: float) -> bytes:
    out: queue.Queue[bytes] = queue.Queue()
    threading.Thread(target=lambda: out.put(proc.stdout.readline()), daemon=True).start()  # type: ignore[union-attr]
    try:
        return out.get(timeout=timeout)
    except queue.Empty:
        proc.kill()
        pytest.fail("no listening line within timeout")


def test_serve_process_lifecycle(tmp_path: Path) -> None:
    proc = _start(tmp_path)
    try:
        line = _readline(proc, 90)
        listening = json.loads(line)
        assert list(listening) == ["event", "host", "port", "path"]
        assert listening["event"] == "listening" and listening["host"] == "127.0.0.1"
        assert listening["path"] == "/agui" and isinstance(listening["port"], int) and listening["port"] > 0
        base = f"http://127.0.0.1:{listening['port']}"
        with urllib.request.urlopen(f"{base}/healthz", timeout=10) as resp:
            assert json.loads(resp.read()) == {"ok": True}
        body = json.dumps({"threadId": "t", "runId": "r", "messages": [{"id": "u", "role": "user", "content": "hi"}],
                           "state": {}, "tools": [], "context": [], "forwardedProps": {}}).encode()
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(urllib.request.Request(f"{base}/agui", data=body, method="POST", headers={
                "content-type": "application/json"}), timeout=10)
        assert denied.value.code == 401
        req = urllib.request.Request(f"{base}/agui", data=body, method="POST", headers={
            "content-type": "application/json", "x-ci-chat-token": TOKEN})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            events = [json.loads(ln[5:]) for ln in resp.read().decode().splitlines() if ln.startswith("data:")]
        assert events[0]["type"] == "RUN_STARTED" and events[-1]["type"] == "RUN_FINISHED"
        assert "TEXT_MESSAGE_CONTENT" in {e["type"] for e in events}

        started = time.monotonic()
        proc.stdin.close()  # type: ignore[union-attr]
        assert proc.wait(timeout=10) == 0
        assert time.monotonic() - started < 10
        assert proc.stdout.read() == b""  # type: ignore[union-attr]  # exactly one stdout line
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    log = (tmp_path / "stderr.log").read_text(encoding="utf-8", errors="replace")
    assert TOKEN not in log


def test_serve_process_without_token_exits_2(tmp_path: Path) -> None:
    proc = _start(tmp_path, token="too-short")
    assert proc.wait(timeout=60) == 2
    assert proc.stdout.read() == b""  # type: ignore[union-attr]
    err = (tmp_path / "stderr.log").read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "CI_CHAT_TOKEN" in json.loads(err)["error"]
