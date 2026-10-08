from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from ci_lab.ledger.atomic import append_jsonl, atomic_write_json, read_json, read_jsonl
from ci_lab.ledger.layout import Layout, night_date, run_dir
from ci_lab.ledger.lock import FileLock, LockTimeout, ledger_lock, lock_dir_for, lock_for


def test_layout_paths(tmp_path):
    lay = Layout(tmp_path)
    assert lay.campaign_json("demo-1") == tmp_path / "experiments/campaigns/demo-1/campaign.json"
    assert lay.frontier_json("demo-1").name == "frontier.json"
    assert lay.history_jsonl("demo-1").name == "history.jsonl"
    exp = lay.experiment_json("demo-1", "demo-1-r03")
    assert exp == tmp_path / "experiments/campaigns/demo-1/rounds/demo-1-r03/experiment.json"
    assert lay.decisions_json("demo-1", "demo-1-r03").parent == exp.parent
    assert lay.eval_json("demo-1", "demo-1-r03", "inc.json") == exp.parent / "eval" / "inc.json"
    assert lay.sleep_state() == tmp_path / "experiments/sleep/state.json"
    assert lay.night_experiment("20261007") == tmp_path / "experiments/sleep/nights/2026-10-07/experiment.json"
    assert lay.holdout_looks() == tmp_path / "experiments/holdout-looks.jsonl"
    assert lay.rel(exp) == "experiments/campaigns/demo-1/rounds/demo-1-r03/experiment.json"


@pytest.mark.parametrize("bad", ["../x", "Demo", "ab", "a/b", ""])
def test_layout_rejects_bad_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        Layout(tmp_path).campaign_dir(bad)


def test_layout_rejects_bad_eval_and_date(tmp_path):
    lay = Layout(tmp_path)
    with pytest.raises(ValueError):
        lay.eval_json("demo-1", "demo-1-r01", "../escape")
    with pytest.raises(ValueError):
        night_date("2026-13-40")


def test_run_dir_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_RUN_DIR", str(tmp_path / "runs"))
    d = run_dir("run-1")
    assert d == tmp_path / "runs" / "run-1" and d.is_dir()
    with pytest.raises(ValueError):
        run_dir("../x")


def test_atomic_json_roundtrip_and_no_temp_left(tmp_path):
    p = tmp_path / "a" / "b.json"
    atomic_write_json(p, {"b": 1, "a": (1, 2)})
    atomic_write_json(p, {"b": 2})
    assert read_json(p) == {"b": 2}
    assert read_json(tmp_path / "missing.json", default=None) is None
    assert [x.name for x in p.parent.iterdir()] == ["b.json"]
    assert p.read_bytes().endswith(b"}\n") and b"\r\n" not in p.read_bytes()


def test_jsonl_survives_torn_line(tmp_path):
    p = tmp_path / "log.jsonl"
    append_jsonl(p, {"n": 1})
    with open(p, "ab") as fh:
        fh.write(b'{"n": 2, "trunc')  # crash mid-append
    append_jsonl(p, [{"n": 3}, {"n": 4}])
    assert [r["n"] for r in read_jsonl(p)] == [1, 3, 4]


def test_lock_reentrant_and_exclusive_across_threads(tmp_path):
    lock = FileLock(tmp_path / "x.lock", timeout=5)
    with lock:
        with lock:  # same thread re-enters
            assert lock.held
        got = []

        def other():
            try:
                FileLock(tmp_path / "x.lock", timeout=0.2).acquire()
            except LockTimeout:
                got.append("timeout")

        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert got == ["timeout"]
    assert not lock.held


def test_lock_async_tasks_exclusive(tmp_path):
    lock_path = tmp_path / "a.lock"
    order: list[str] = []

    async def worker(name):
        async with FileLock(lock_path, timeout=5):
            order.append(f"{name}+")
            await asyncio.sleep(0.05)
            order.append(f"{name}-")

    async def main():
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(main())
    assert order in (["a+", "a-", "b+", "b-"], ["b+", "b-", "a+", "a-"])


def test_lock_cross_process(tmp_path):
    lock_path = tmp_path / "p.lock"
    code = textwrap.dedent(f"""
        import sys, time
        from ci_lab.ledger.lock import FileLock
        with FileLock({str(lock_path)!r}):
            print("locked", flush=True)
            sys.stdin.readline()
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        t0 = time.monotonic()
        with pytest.raises(LockTimeout):
            FileLock(lock_path, timeout=0.3).acquire()
        assert time.monotonic() - t0 >= 0.25
    finally:
        proc.stdin.write("\n")
        proc.stdin.close()
        proc.wait(timeout=30)
    with FileLock(lock_path, timeout=5):
        pass


def test_lock_placement_in_git_common_dir(git_repo):
    target = git_repo / "experiments" / "campaigns" / "c-1" / "frontier.json"
    d = lock_dir_for(target)
    assert d.parent.name == "ci-lab" and d.parent.parent == (git_repo / ".git").resolve()
    lk = lock_for(target)
    with lk:
        assert lk.path.exists()
    assert ledger_lock(git_repo).path == (git_repo / ".git").resolve() / "ci-lab" / "ledger.lock"
    assert not (git_repo / "experiments").exists()


def test_lock_placement_outside_git(tmp_path):
    from ci_lab.gitops import git

    if git.is_repo(tmp_path):
        pytest.skip("tmp dir is inside a git repository")
    assert lock_dir_for(tmp_path / "x" / "f.json") == (tmp_path / "x" / ".ci-lab-locks").absolute()
