from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ci_lab.agl.journal import FileRolloutJournal
from ci_lab.contracts import RolloutKey, op_id

KEY = RolloutKey("camp-r00", "base", "case-1", trial=0)


def _lines(j: FileRolloutJournal, key: RolloutKey = KEY) -> list[str]:
    return j.path(key.rollout_id).read_text(encoding="utf-8").splitlines()


def test_roundtrip_and_iter(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    j.start(KEY, {"intent": "change"})
    j.event(KEY, "reward", {"value": 1.0}, event_id=op_id("a"))
    j.finish(KEY, "succeeded")
    rec = j.load(KEY.rollout_id)
    assert rec is not None
    assert rec.key == KEY and rec.input == {"intent": "change"} and rec.status == "succeeded"
    assert rec.attempts == ["0"] and rec.finished_attempt == "0"
    assert [e["event_type"] for e in j.events(KEY)] == ["reward"]
    assert [r.rollout_id for r in j.iter_rollouts()] == [KEY.rollout_id]
    assert j.load("ro-missing") is None
    for line in _lines(j):
        rec_line = json.loads(line)
        assert rec_line["v"] == 1 and rec_line["rollout_id"] == KEY.rollout_id


def test_dedupe_by_event_id_start_once_and_terminal_once(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path, fsync=False)
    j.start(KEY, {"x": 1})
    j.start(KEY, {"x": 2})  # resume: no second start record
    eid = op_id(KEY.rollout_id, "0", "reward", "reward")
    assert j.append_event(KEY, "reward", {"value": 1.0}, event_id=eid) is True
    assert j.append_event(KEY, "reward", {"value": 0.0}, event_id=eid) is False
    j.finish(KEY, "succeeded")
    assert j.append_finish(KEY, "failed") is False
    rec = j.load(KEY.rollout_id)
    assert rec is not None and rec.status == "succeeded" and rec.input == {"x": 1}
    assert [e["data"]["value"] for e in rec.events] == [1.0]
    assert len(_lines(j)) == 3  # start, event, finish


def test_other_instance_sees_and_dedupes(tmp_path: Path) -> None:
    a, b = FileRolloutJournal(tmp_path, fsync=False), FileRolloutJournal(tmp_path, fsync=False)
    a.start(KEY, {})
    a.event(KEY, "e", {}, event_id="id-1")
    assert b.append_event(KEY, "e", {}, event_id="id-1") is False
    b.event(KEY, "e", {}, event_id="id-2")
    assert [e["event_id"] for e in a.events(KEY)] == ["id-1", "id-2"]
    b.finish(KEY, "failed")
    assert a.append_finish(KEY, "succeeded") is False
    assert a.load(KEY.rollout_id).status == "failed"  # type: ignore[union-attr]


def test_truncated_last_line_tolerated(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    j.start(KEY, {})
    j.event(KEY, "e", {"n": 1}, event_id="id-1")
    with open(j.path(KEY.rollout_id), "ab") as fh:
        fh.write(b'{"v":1,"kind":"event","event_id":"id-2","event_ty')  # crash mid-write
    fresh = FileRolloutJournal(tmp_path)
    rec = fresh.load(KEY.rollout_id)
    assert rec is not None and [e["event_id"] for e in rec.events] == ["id-1"]
    fresh.event(KEY, "e", {"n": 2}, event_id="id-2")
    fresh.finish(KEY, "succeeded")
    again = FileRolloutJournal(tmp_path).load(KEY.rollout_id)
    assert again is not None and again.status == "succeeded"
    assert [e["event_id"] for e in again.events] == ["id-1", "id-2"]
    raw = j.path(KEY.rollout_id).read_bytes()
    assert raw.endswith(b"\n") and b'"event_ty\n' in raw  # garbage isolated on its own line


def test_complete_but_unterminated_line_is_kept(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    j.start(KEY, {})
    path = j.path(KEY.rollout_id)
    path.write_bytes(path.read_bytes() + json.dumps(
        {"v": 1, "kind": "event", "rollout_id": KEY.rollout_id, "attempt_id": "0", "event_id": "id-x",
         "event_type": "e", "data": {}, "ts": 0}).encode())
    fresh = FileRolloutJournal(tmp_path)
    assert [e["event_id"] for e in fresh.events(KEY)] == ["id-x"]
    assert fresh.append_event(KEY, "e", {}, event_id="id-x") is False
    fresh.event(KEY, "e", {}, event_id="id-y")
    assert [e["event_id"] for e in FileRolloutJournal(tmp_path).events(KEY)] == ["id-x", "id-y"]


def test_unsafe_rollout_id_rejected(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    with pytest.raises(ValueError):
        j.load("../escape")
    with pytest.raises(ValueError):
        j.append_event(KEY, "e", {}, event_id="")


def test_multi_attempt_records(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    k1 = RolloutKey("camp-r00", "base", "case-1", attempt=1)
    j.start(KEY, {})
    j.event(KEY, "e", {}, event_id="a0")
    j.start(k1, {})
    j.event(k1, "e", {}, event_id="a1")
    rec = j.load(KEY.rollout_id)
    assert rec is not None and rec.attempts == ["0", "1"] and rec.latest_attempt == "1"
    assert [e["event_id"] for e in rec.events_for(attempt_id="1")] == ["a1"]


def test_cross_process_appends_are_deduped(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[3] / "src"
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(src)!r})
        from ci_lab.agl.journal import FileRolloutJournal
        from ci_lab.contracts import RolloutKey
        key = RolloutKey("camp-r00", "base", "case-1")
        j = FileRolloutJournal({str(tmp_path)!r}, fsync=False)
        j.start(key, {{}})
        worker = sys.argv[1]
        for i in range(40):
            j.event(key, "shared", {{"w": worker}}, event_id=f"shared-{{i}}")
            j.event(key, "own", {{"w": worker}}, event_id=f"own-{{worker}}-{{i}}")
        j.finish(key, "succeeded" if worker == "0" else "failed")
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script, str(w)]) for w in range(4)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    rec = FileRolloutJournal(tmp_path).load(KEY.rollout_id)
    assert rec is not None and rec.terminal
    ids = [e["event_id"] for e in rec.events]
    assert len(ids) == len(set(ids)) == 40 + 4 * 40
    lines = _lines(FileRolloutJournal(tmp_path))
    assert all(json.loads(line) for line in lines)  # no interleaved/torn lines
    assert sum(json.loads(line)["kind"] == "finish" for line in lines) == 1
    assert sum(json.loads(line)["kind"] == "start" for line in lines) == 1
