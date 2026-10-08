from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading

import pytest

from ci_lab.contracts import Outbox
from ci_lab.ledger.frontier import Frontier, FrontierConflict, cas_frontier, read_frontier
from ci_lab.ledger.looks import LookBudgetExceeded, count_looks, planned_looks, record_look
from ci_lab.ledger.outbox import FileOutbox

# ---------------------------------------------------------------- frontier


def test_frontier_cas(tmp_path):
    path = tmp_path / "frontier.json"
    assert read_frontier(path) is None
    h0 = Frontier("c0", "t0", 0.5, 0)
    assert cas_frontier(path, None, h0) == h0
    with pytest.raises(FrontierConflict):
        cas_frontier(path, None, Frontier("c9", "t9", 0.9, 1))
    h1 = cas_frontier(path, "c0", {"incumbent": "c1", "harness_tree": "t1", "score": 0.6, "round": 1,
                                   "note": "kept"})
    assert read_frontier(path) == h1 and h1.extra == {"note": "kept"}
    # stale writer loses
    with pytest.raises(FrontierConflict) as ei:
        cas_frontier(path, "c0", Frontier("c2", "t2", 0.7, 2))
    assert ei.value.actual.incumbent == "c1"
    # re-applying an already-applied swap (resume) is idempotent
    assert cas_frontier(path, "c0", h1) == h1


def test_frontier_concurrent_cas_single_winner(tmp_path):
    path = tmp_path / "frontier.json"
    cas_frontier(path, None, Frontier("c0", "t0", 0.5, 0))
    wins, losses = [], []

    def attempt(i):
        try:
            cas_frontier(path, "c0", Frontier(f"c{i}", f"t{i}", 0.6, 1))
            wins.append(i)
        except FrontierConflict:
            losses.append(i)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(1, 9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(losses) == 7
    assert read_frontier(path).incumbent == f"c{wins[0]}"


# ---------------------------------------------------------------- outbox


def test_outbox_matches_contract(tmp_path):
    box: Outbox = FileOutbox(tmp_path / "o.jsonl")
    assert callable(box.run_once) and callable(box.arun_once)


def test_outbox_runs_once_and_persists(tmp_path):
    path = tmp_path / "outbox.jsonl"
    calls = []
    box = FileOutbox(path)
    assert box.run_once("op-1", lambda: calls.append(1) or {"pr": 7}) == {"pr": 7}
    assert box.run_once("op-1", lambda: calls.append(2) or {"pr": 8}) == {"pr": 7}
    # a fresh instance (new process) sees the durable result
    assert FileOutbox(path).run_once("op-1", lambda: calls.append(3)) == {"pr": 7}
    assert calls == [1]
    assert FileOutbox(path).status("op-1") == "succeeded"


def test_outbox_failed_op_is_retried(tmp_path):
    box = FileOutbox(tmp_path / "o.jsonl")

    def boom():
        raise RuntimeError("push rejected")

    with pytest.raises(RuntimeError):
        box.run_once("push", boom)
    assert box.status("push") == "failed"
    assert box.run_once("push", lambda: "ok") == "ok"
    assert box.latest("push")["attempt"] == 2


def test_outbox_returns_json_form_consistently(tmp_path):
    box = FileOutbox(tmp_path / "o.jsonl")
    first = box.run_once("t", lambda: (1, 2))
    assert first == [1, 2] == FileOutbox(tmp_path / "o.jsonl").run_once("t", lambda: None)


def test_outbox_unserializable_result_recorded_not_rerun(tmp_path):
    box = FileOutbox(tmp_path / "o.jsonl")
    calls = []
    with pytest.raises(TypeError):
        box.run_once("x", lambda: calls.append(1) or object())
    assert box.run_once("x", lambda: calls.append(2)) is None
    assert calls == [1]


def _crash_mid_effect(path):
    """Run an op in a child process that dies (os._exit) inside fn, leaving 'started'."""
    code = textwrap.dedent(f"""
        import os
        from ci_lab.ledger.outbox import FileOutbox
        FileOutbox({str(path)!r}).run_once("open-pr", lambda: os._exit(3))
    """)
    rc = subprocess.run([sys.executable, "-c", code], timeout=60).returncode
    assert rc == 3


def test_outbox_crash_then_reconcile_finds_remote_state(tmp_path):
    path = tmp_path / "outbox.jsonl"
    _crash_mid_effect(path)
    box = FileOutbox(path)
    assert box.status("open-pr") == "started" and box.pending() == ["open-pr"]
    fn_calls = []
    res = box.run_once("open-pr", lambda: fn_calls.append(1) or {"pr": 99},
                       reconcile=lambda: {"pr": 42})
    assert res == {"pr": 42} and fn_calls == []
    assert box.latest("open-pr")["reconciled"] is True
    assert box.pending() == []


def test_outbox_crash_then_reconcile_none_reruns(tmp_path):
    path = tmp_path / "outbox.jsonl"
    _crash_mid_effect(path)
    seen = []
    res = FileOutbox(path).run_once("open-pr", lambda: seen.append("fn") or {"pr": 5},
                                    reconcile=lambda: seen.append("reconcile") or None)
    assert res == {"pr": 5} and seen == ["reconcile", "fn"]


def test_outbox_concurrent_same_op_runs_once(tmp_path):
    box = FileOutbox(tmp_path / "o.jsonl")
    calls = []
    gate = threading.Event()

    def fn():
        calls.append(1)
        gate.wait(0.2)
        return "done"

    results = []
    threads = [threading.Thread(target=lambda: results.append(box.run_once("same", fn))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1] and results == ["done"] * 5


def test_outbox_async(tmp_path):
    path = tmp_path / "o.jsonl"
    calls = []

    async def fn():
        calls.append(1)
        await asyncio.sleep(0.02)
        return {"sha": "abc"}

    async def reconcile_none():
        return None

    async def main():
        box = FileOutbox(path)
        return await asyncio.gather(*(box.arun_once("commit", fn, reconcile=reconcile_none) for _ in range(4)))

    assert asyncio.run(main()) == [{"sha": "abc"}] * 4
    assert calls == [1]

    async def reconcile_found():
        return {"sha": "remote"}

    async def again():
        return await FileOutbox(path).arun_once("other", fn, reconcile=reconcile_found)

    assert asyncio.run(again()) == {"sha": "remote"} and calls == [1]


def test_outbox_journal_not_in_worktree_locks(git_repo):
    path = git_repo / "experiments" / "outbox.jsonl"
    FileOutbox(path).run_once("a", lambda: 1)
    assert sorted(p.name for p in path.parent.iterdir()) == ["outbox.jsonl"]
    assert any((git_repo / ".git" / "ci-lab" / "locks").iterdir())


# ---------------------------------------------------------------- holdout looks


def test_looks_budget(tmp_path):
    path = tmp_path / "experiments" / "holdout-looks.jsonl"
    h = "sha256-" + "a" * 16
    rec = record_look(path, h, experiment_id="demo-1-confirm", planned=1, campaign_id="demo-1")
    assert rec["look_no"] == 1 and count_looks(path, h) == 1 and planned_looks(path, h) == 1
    # resume of the same confirmation does not consume another look
    assert record_look(path, h, experiment_id="demo-1-confirm", planned=1)["look_id"] == rec["look_id"]
    assert count_looks(path, h) == 1
    with pytest.raises(LookBudgetExceeded):
        record_look(path, h, experiment_id="demo-2-confirm", planned=1)
    with pytest.raises(ValueError):  # cannot raise the plan post hoc
        record_look(path, h, experiment_id="demo-2-confirm", planned=2)
    other = "sha256-" + "b" * 16
    record_look(path, other, experiment_id="demo-2-confirm", planned=2)
    record_look(path, other, experiment_id="demo-3-confirm", planned=2)
    with pytest.raises(LookBudgetExceeded):
        record_look(path, other, experiment_id="demo-4-confirm", planned=2)
    assert count_looks(path, other) == 2 and count_looks(path, h) == 1


def test_looks_rejects_bad_hash(tmp_path):
    with pytest.raises(ValueError):
        record_look(tmp_path / "l.jsonl", "../x", experiment_id="e-1")
    assert not os.path.exists(tmp_path / "l.jsonl")
