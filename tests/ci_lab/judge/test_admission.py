"""Host-wide s1 judge admission: cross-process slot leases, FIFO tickets, reentrancy, id_slot."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import httpx
import pytest

from ci_lab.judge import admission as A
from ci_lab.judge import backends as B
from ci_lab.judge.s1types import Question

URL = "http://127.0.0.1:59999"


@pytest.fixture(autouse=True)
def _locks(tmp_path, monkeypatch):
    monkeypatch.setenv(A.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.delenv(A.MAX_INFLIGHT_ENV, raising=False)
    monkeypatch.delenv(A.PIN_SLOT_ENV, raising=False)
    monkeypatch.delenv(A.ADMISSION_LOG_ENV, raising=False)
    monkeypatch.setattr(A, "_props_cache", {})
    monkeypatch.setattr(A, "_slot_pref", {})
    return tmp_path / "locks"


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _max_overlap(intervals: list[tuple[float, float]]) -> int:
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


# ------------------------------------------------------------------ configuration

def test_normalize_url_unifies_loopback_spellings():
    assert A.normalize_url("http://localhost:8081/") == "http://127.0.0.1:8081"
    assert A.normalize_url("http://127.0.0.1:8081/v1") == "http://127.0.0.1:8081"
    assert A.normalize_url("127.0.0.1:8081") == "http://127.0.0.1:8081"
    assert A.lock_dir("http://localhost:8081") == A.lock_dir("http://127.0.0.1:8081/")
    assert A.lock_dir("http://127.0.0.1:8081") != A.lock_dir("http://127.0.0.1:8082")


def test_lock_root_defaults_to_per_user_dir(monkeypatch):
    monkeypatch.delenv(A.LOCK_DIR_ENV)
    root = A.lock_root()
    assert root.parts[-2:] == ("ci-lab", "locks")
    if os.name == "nt":
        assert str(root).startswith(os.environ.get("LOCALAPPDATA", str(Path.home())))


def _props_transport(payload: dict | None, calls: list | None = None) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if payload is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(respond)


def test_capacity_from_props_is_capped_and_cached():
    calls: list = []
    t = _props_transport({"total_slots": 4}, calls)
    assert A.resolve_capacity(URL, transport=t) == (min(4, A.DEFAULT_MAX_INFLIGHT), 4)
    assert A.resolve_capacity(URL, transport=t) == (min(4, A.DEFAULT_MAX_INFLIGHT), 4)
    assert calls == ["/props"]


def test_capacity_env_overrides_props(monkeypatch):
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "3")
    assert A.resolve_capacity(URL, transport=_props_transport({"total_slots": 4})) == (3, 4)
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "-1")
    with pytest.raises(ValueError):
        A.resolve_capacity(URL)


def test_capacity_falls_back_to_one_when_props_unreachable():
    assert A.resolve_capacity(URL, transport=_props_transport(None)) == (1, None)
    assert A.resolve_capacity(URL, transport=_props_transport({"model_path": "m.gguf"})) == (1, None)


def test_capacity_known_slots_skip_props():
    assert A.resolve_capacity(URL, total_slots=0, transport=_props_transport(None, calls := [])) == (1, 0)
    assert calls == []


def test_capacity_zero_disables_admission(monkeypatch, _locks):
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "0")
    with A.hold(URL) as a, A.hold("http://127.0.0.1:59998") as b:
        assert a.index == -1 and b.index == -1 and a.id_slot is None
    assert not _locks.exists()


# ------------------------------------------------------------------ leases in one process

def test_hold_is_reentrant_and_releases(monkeypatch):
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "1")
    with A.hold(URL) as outer:
        with A.hold(URL) as inner:
            assert inner is outer and A.current_lease(URL) is outer
        assert outer.held
        w = A._Waiter(A.normalize_url(URL), 1, None)
        try:
            assert w.attempt() is None  # capacity 1 is taken, even within this process
        finally:
            w.close()
    assert A.current_lease() is None and not outer.held
    with A.hold(URL) as again:
        assert again.index == 0


def test_hold_async_shares_lease_with_to_thread(monkeypatch):
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "1")

    def nested() -> A.Lease:
        with A.hold(URL) as lease:
            return lease

    async def main() -> tuple[A.Lease, A.Lease]:
        async with A.hold_async(URL) as outer:
            inner = await asyncio.wait_for(asyncio.to_thread(nested), timeout=5)
        return outer, inner

    outer, inner = asyncio.run(main())
    assert inner is outer and not outer.held


def test_different_servers_get_independent_leases(monkeypatch):
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "1")
    with A.hold(URL) as a, A.hold("http://127.0.0.1:59998") as b:
        assert a is not b and a.index == b.index == 0


def test_tickets_are_served_in_order():
    url = A.normalize_url(URL)
    first = A._Waiter(url, 1, None)
    time.sleep(0.002)
    second = A._Waiter(url, 1, None)
    try:
        assert second.attempt() is None and second.position == 1
        lease = first.attempt()
        assert lease is not None and lease.index == 0
        first.close()
        assert second.attempt() is None  # slot still held by first's lease
        lease.release()
        lease2 = second.attempt()
        assert lease2 is not None
        lease2.release()
    finally:
        first.close()
        second.close()


def test_stale_ticket_of_dead_waiter_is_reaped(monkeypatch):
    url = A.normalize_url(URL)
    qdir = A.lock_dir(url) / "queue"
    qdir.mkdir(parents=True)
    old = time.time_ns() - int(10 * 1e9)
    stale = qdir / f"{old:020d}-1-00000000-abcdef.ticket"
    stale.write_text("")
    w = A._Waiter(url, 1, None)
    try:
        lease = w.attempt()
        assert lease is not None and not stale.exists()
        lease.release()
    finally:
        w.close()


def test_abandoned_lease_stops_nested_work_and_stays_locked(monkeypatch):
    """ASSERT's timeout cancels ``await to_thread(...)``; the thread keeps the slot until it stops."""
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "1")
    started, go_on = threading.Event(), threading.Event()
    outcome: dict = {}

    def nested() -> None:
        with A.hold(URL) as lease:
            started.set()
            go_on.wait(5)
            try:
                lease.check()
            except A.LeaseAbandoned as e:
                outcome["error"] = e

    async def main() -> None:
        async with A.hold_async(URL):
            task = asyncio.ensure_future(asyncio.to_thread(nested))
            await asyncio.to_thread(started.wait, 5)
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await task
        owner_gone.set()

    owner_gone = threading.Event()
    # asyncio.run waits for the default executor on exit, so drive it from a helper thread.
    runner = threading.Thread(target=asyncio.run, args=(main(),))
    runner.start()
    assert owner_gone.wait(10)
    w = A._Waiter(A.normalize_url(URL), 1, None)
    try:
        assert w.attempt() is None  # the orphaned thread still occupies the slot
        go_on.set()
        deadline = time.monotonic() + 5
        while (lease := w.attempt()) is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lease is not None
        lease.release()
    finally:
        go_on.set()
        w.close()
        runner.join(10)
    assert isinstance(outcome.get("error"), A.LeaseAbandoned)
    assert not issubclass(A.LeaseAbandoned, B.BackendError)  # never triggers the provider fallback


def test_admission_log_records_wait_and_hold(monkeypatch, tmp_path):
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "1")
    monkeypatch.setenv(A.ADMISSION_LOG_ENV, str(tmp_path / "adm"))
    with A.hold(URL):
        time.sleep(0.05)
    rows = [json.loads(x) for x in (tmp_path / "adm" / f"admission-{os.getpid()}.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["slot"] == 0 and rows[0]["held_s"] >= 0.04 and rows[0]["capacity"] == 1


def test_s1_local_url(monkeypatch):
    monkeypatch.setenv(B.LLAMA_URL_ENV, "http://127.0.0.1:9000")
    assert A.s1_local_url("s1/llamacpp/qwen3.5-4b") == "http://127.0.0.1:9000"
    assert A.s1_local_url("s1/local/x", api_base="http://h:1") == "http://h:1"
    assert A.s1_local_url("s1/scripted/default") is None
    assert A.s1_local_url("openai/gpt-5-mini") is None


# ------------------------------------------------------------------ across real processes

_HOLDER = textwrap.dedent("""
    import json, sys, time
    from ci_lab.judge import admission
    out, n, hold_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    for _ in range(n):
        with admission.hold("__URL__") as lease:
            t0 = time.time()
            time.sleep(hold_s)
            t1 = time.time()
        with open(out, "a") as f:
            f.write(json.dumps({"start": t0, "end": t1, "slot": lease.index}) + "\\n")
""").replace("__URL__", URL)


@pytest.mark.parametrize("capacity", [1, 2])
def test_capacity_holds_across_processes(tmp_path, capacity):
    env = _env(**{A.MAX_INFLIGHT_ENV: str(capacity)})
    out = tmp_path / "intervals.jsonl"
    procs = [subprocess.Popen([sys.executable, "-c", _HOLDER, str(out), "2", "0.4"], env=env)
             for _ in range(4)]
    for p in procs:
        assert p.wait(timeout=120) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 8
    assert _max_overlap([(r["start"], r["end"]) for r in rows]) <= capacity
    assert {r["slot"] for r in rows} <= set(range(capacity))


def test_lock_released_when_holder_is_killed(tmp_path):
    env = _env(**{A.MAX_INFLIGHT_ENV: "1"})
    script = textwrap.dedent(f"""
        import time
        from ci_lab.judge import admission
        with admission.hold("{URL}"):
            print("held", flush=True)
            time.sleep(120)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        w = A._Waiter(A.normalize_url(URL), 1, None)
        try:
            assert w.attempt() is None
            proc.kill()
            proc.wait(timeout=30)
            deadline = time.monotonic() + 15
            while (lease := w.attempt()) is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert lease is not None, "lock not released after the holder was killed"
            lease.release()
        finally:
            w.close()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()


# ------------------------------------------------------------------ llama.cpp backend integration

class _FakeLlama(BaseHTTPRequestHandler):
    """Tiny llama-server stand-in that measures concurrent /v1/chat/completions requests."""

    server_version = "fake-llama"
    lock = threading.Lock()
    inflight = 0
    max_inflight = 0
    bodies: ClassVar[list] = []
    delay = 0.1

    def log_message(self, *args) -> None:
        pass

    def _send(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send({"total_slots": 4, "model_path": "fake.gguf"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        with cls.lock:
            cls.inflight += 1
            cls.max_inflight = max(cls.max_inflight, cls.inflight)
            cls.bodies.append(body)
        time.sleep(cls.delay)
        with cls.lock:
            cls.inflight -= 1
        top = [{"token": "yes", "logprob": math.log(0.9)}, {"token": "no", "logprob": math.log(0.1)}]
        self._send({"choices": [{"message": {"content": "yes"},
                                 "logprobs": {"content": [{"token": "yes", "logprob": top[0]["logprob"],
                                                           "top_logprobs": top}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1}})


@pytest.fixture
def fake_llama():
    _FakeLlama.inflight = _FakeLlama.max_inflight = 0
    _FakeLlama.bodies = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlama)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


QUESTIONS = {f"q{i}": Question(type="noul", instructions=f"question {i}?") for i in range(3)}


def test_backend_pins_id_slot_to_lease(fake_llama, monkeypatch):
    monkeypatch.setenv(A.MAX_INFLIGHT_ENV, "2")
    monkeypatch.setenv(A.PIN_SLOT_ENV, "1")
    be = B.LlamaCppLogprobBackend(fake_llama)
    d = be.decide("transcript", QUESTIONS)
    assert all(a.ok for a in d.answers.values())
    assert [b.get("id_slot") for b in _FakeLlama.bodies] == [0, 0, 0]
    with A.hold(fake_llama):  # takes slot 0 -> the backend reuses this lease (reentrant)
        be.decide("transcript", QUESTIONS)
    assert [b.get("id_slot") for b in _FakeLlama.bodies[3:]] == [0, 0, 0]


def test_backend_id_slot_is_opt_in(fake_llama, monkeypatch):
    B.LlamaCppLogprobBackend(fake_llama).decide("transcript", QUESTIONS)  # default: off
    monkeypatch.setenv(A.PIN_SLOT_ENV, "0")
    B.LlamaCppLogprobBackend(fake_llama).decide("transcript", QUESTIONS)
    assert all("id_slot" not in b for b in _FakeLlama.bodies)


def test_backend_without_server_slots_does_not_pin(monkeypatch):
    seen: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "m.gguf"})
        seen.append(json.loads(request.content))
        top = [{"token": "no", "logprob": math.log(0.9)}, {"token": "yes", "logprob": math.log(0.1)}]
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [
            {"token": "no", "logprob": top[0]["logprob"], "top_logprobs": top}]}}]})

    be = B.LlamaCppLogprobBackend(URL, transport=httpx.MockTransport(respond))
    be.decide("t", QUESTIONS)
    assert len(seen) == 3 and all("id_slot" not in b for b in seen)


_JUDGE_USER = textwrap.dedent("""
    import sys
    from ci_lab.judge import backends
    from ci_lab.judge.s1types import Question
    be = backends.LlamaCppLogprobBackend(sys.argv[1])
    qs = {f"q{i}": Question(type="noul", instructions=f"q{i}?") for i in range(3)}
    for _ in range(2):
        be.decide("transcript", qs)
""")


@pytest.mark.parametrize("capacity", [1, 2])
def test_concurrent_judge_processes_never_exceed_capacity(fake_llama, capacity):
    """Campaign arms / parallel suites: separate processes judging through one llama-server."""
    env = _env(**{A.MAX_INFLIGHT_ENV: str(capacity), A.PIN_SLOT_ENV: "1"})
    procs = [subprocess.Popen([sys.executable, "-c", _JUDGE_USER, fake_llama], env=env) for _ in range(3)]
    for p in procs:
        assert p.wait(timeout=120) == 0
    assert len(_FakeLlama.bodies) == 3 * 2 * 3
    assert 1 <= _FakeLlama.max_inflight <= capacity
    assert {b["id_slot"] for b in _FakeLlama.bodies} <= set(range(capacity))
