from __future__ import annotations

import asyncio
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider

from ci_lab.bus.ids import proposal_id
from ci_lab.bus.state import BusInvariantError
from ci_lab.bus.types import (
    GENESIS,
    Author,
    ManifestBody,
    NoteBody,
    ProposalBody,
    VerdictBody,
    VoteBody,
)
from ci_lab.bus.wal import EVENT_APPEND, TORN_TAIL_NOTE, AgentBus, BusCorrupt

T = "r1/t1"
ORCH = Author(role="orchestrator", name="orch")


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def notes(bus: AgentBus, n: int, topic: str = T, author: Author = ORCH) -> list:  # type: ignore[type-arg]
    return [run(bus.append(topic, "note", author, NoteBody(text=f"n{i}"))) for i in range(n)]


def test_round_trip_heads_topics(tmp_path: Path) -> None:
    bus = AgentBus(tmp_path / "bus")
    e0 = run(bus.append(T, "note", ORCH, NoteBody(text="a")))
    e1 = run(bus.append(T, "note", ORCH, {"text": "ü", "data": {"k": [1, 2.5]}}))
    assert (e0.seq, e0.prev, e1.seq, e1.prev) == (0, GENESIS, 1, e0.hash) and e1.hash_ok()
    assert bus.wal_path(T) == tmp_path / "bus" / "r1" / "t1.wal.jsonl"
    assert AgentBus(tmp_path / "bus").read(T) == [e0, e1] == bus.read(T)
    assert bus.head(T) == (1, e1.hash) and bus.head("r1/t2") == (-1, GENESIS)
    notes(bus, 1, "r2/t1")
    assert bus.topics() == ["r1/t1", "r2/t1"] and bus.topics("r2/") == ["r2/t1"]
    assert bus.heads("r1") == {T: e1.hash} and bus.state(T).entries == (e0, e1)


def test_invariant_rejection_leaves_file_unchanged(tmp_path: Path) -> None:
    bus = AgentBus(tmp_path)
    with pytest.raises(BusInvariantError, match="I0"):
        run(bus.append("r1/_run", "note", ORCH, NoteBody(text="x")))
    assert not bus.wal_path("r1/_run").exists()
    notes(bus, 2)
    before = bus.wal_path(T).read_bytes()
    with pytest.raises(BusInvariantError, match="I6"):
        run(bus.append(T, "note", Author(role="student", name="s"), NoteBody(text="x")))
    assert bus.wal_path(T).read_bytes() == before


def test_task_topic_uses_run_manifest_for_quorum(tmp_path: Path) -> None:
    bus = AgentBus(tmp_path)
    stu, judge = Author(role="student", name="s"), Author(role="judge", name="j")
    art = run(bus.put_artifact(T, "out"))
    pid = proposal_id("t1@1", "student", "s")
    p = ProposalBody(proposal=pid, attempt="t1@1", rubric_version="rub@v1", artifact=art, summary="")
    run(bus.append(T, "proposal", stu, p))
    run(bus.append(T, "vote", Author(role="voter", name="v"), VoteBody(
        proposal=pid, rubric_version="rub@v1", voter="v", measure="s1", criterion=None, passed=True, score=None,
        confidence=None), ref=0))
    verdict = VerdictBody(proposal=pid, attempt="t1@1", rubric_version="rub@v1", decision="commit", score=1.0,
                          criteria={}, votes=(1,), correction=None, escalated=False)
    with pytest.raises(BusInvariantError, match="I3"):
        run(bus.append(T, "verdict", judge, verdict, ref=0))
    run(bus.append("r1/_run", "manifest", ORCH, ManifestBody(
        run="r1", created="now", code_rev="c", config_sha256="a" * 64, graph_sha256=None, max_parallel=1,
        voters=("v",), quorum=1)))
    assert run(bus.append(T, "verdict", judge, verdict, ref=0)).seq == 2


def test_tamper_and_mid_file_corruption(tmp_path: Path) -> None:
    bus = AgentBus(tmp_path)
    notes(bus, 3)
    path, good = bus.wal_path(T), bus.wal_path(T).read_bytes()
    path.write_bytes(good.replace(b'"n1"', b'"nX"'))
    with pytest.raises(BusCorrupt):
        AgentBus(tmp_path).read(T)
    path.write_bytes(good.replace(b'"n2"', b'"nX"'))  # same size: the cache re-verifies the last line
    with pytest.raises(BusCorrupt):
        bus.read(T)
    lines = good.splitlines(keepends=True)
    path.write_bytes(lines[0] + b"garbage\n" + b"".join(lines[1:]))
    with pytest.raises(BusCorrupt):
        run(bus.append(T, "note", ORCH, NoteBody(text="x")))
    assert path.read_bytes() == lines[0] + b"garbage\n" + b"".join(lines[1:])


@pytest.mark.parametrize("junk", [b'{"seq":2,"to', b"garbage\n"])
def test_torn_tail_ignored_by_readers_then_repaired(tmp_path: Path, junk: bytes) -> None:
    bus = AgentBus(tmp_path)
    notes(bus, 2)
    path = bus.wal_path(T)
    with path.open("ab") as fh:
        fh.write(junk)
    assert len(bus.read(T)) == 2 and path.read_bytes().endswith(junk)
    e = run(bus.append(T, "note", ORCH, NoteBody(text="after")))
    got = AgentBus(tmp_path).read(T)
    assert [x.body.text for x in got] == ["n0", "n1", TORN_TAIL_NOTE, "after"] and got[-1] == e


def test_concurrent_asyncio_appends_are_dense(tmp_path: Path) -> None:
    bus = AgentBus(tmp_path)

    async def many() -> None:
        await asyncio.gather(*(bus.append(T, "note", ORCH, NoteBody(text=str(i))) for i in range(40)))

    run(many())
    got = AgentBus(tmp_path).read(T)
    assert [e.seq for e in got] == list(range(40)) and {e.body.text for e in got} == {str(i) for i in range(40)}


_CHILD = """
import asyncio, sys
from ci_lab.bus.types import Author, NoteBody
from ci_lab.bus.wal import AgentBus
bus, a = AgentBus(sys.argv[1]), Author(role="orchestrator", name="p" + sys.argv[2])
async def main():
    await asyncio.gather(*(bus.append("r1/t1", "note", a, NoteBody(text=str(i))) for i in range(15)))
asyncio.run(main())
"""


def test_concurrent_process_appends_are_dense(tmp_path: Path) -> None:
    procs = [subprocess.Popen([sys.executable, "-c", _CHILD, str(tmp_path), str(k)]) for k in range(3)]
    notes(AgentBus(tmp_path), 10)
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    got = AgentBus(tmp_path).read(T)
    assert [e.seq for e in got] == list(range(55))
    assert Counter(e.author.name for e in got) == {"orch": 10, "p0": 15, "p1": 15, "p2": 15}


def test_artifacts_idempotent_and_verified(tmp_path: Path) -> None:
    bus = AgentBus(tmp_path)
    ref = run(bus.put_artifact(T, "héllo"))
    assert run(bus.put_artifact(T, "héllo".encode())) == ref and ref.bytes == len("héllo".encode())
    stored = bus.artifact_dir(T) / ref.path
    assert stored == tmp_path / "r1" / "t1" / "artifacts" / ref.sha256[:2] / ref.sha256
    assert bus.read_artifact(T, ref) == "héllo".encode()
    stored.write_bytes(b"evil!!")
    with pytest.raises(BusCorrupt):
        bus.read_artifact(T, ref)
    run(bus.put_artifact(T, "héllo"))  # a damaged copy is rewritten
    assert bus.read_artifact(T, ref) == "héllo".encode()


def test_append_emits_attribute_only_span_event(tmp_path: Path) -> None:
    tracer = TracerProvider().get_tracer("t")
    with tracer.start_as_current_span("s") as span:
        run(AgentBus(tmp_path).append(T, "note", ORCH, NoteBody(text="secret")))
    (ev,) = span.events  # type: ignore[attr-defined]
    assert ev.name == EVENT_APPEND
    assert dict(ev.attributes) == {"ci.bus.topic": T, "ci.bus.seq": 0, "ci.bus.kind": "note",
                                   "ci.bus.role": "orchestrator"}


def test_cancel_mid_append_finishes_write_before_releasing_lock(tmp_path: Path) -> None:
    import threading

    bus = AgentBus(tmp_path)
    notes(bus, 1)
    started, release = threading.Event(), threading.Event()
    orig = bus._append_locked

    def slow(*a, **kw):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(5)
        return orig(*a, **kw)

    async def main() -> None:
        bus._append_locked = slow  # type: ignore[method-assign]
        victim = asyncio.create_task(bus.append(T, "note", ORCH, NoteBody(text="victim")))
        await asyncio.to_thread(started.wait, 5)
        victim.cancel()
        bus._append_locked = orig  # type: ignore[method-assign]
        follower = asyncio.create_task(bus.append(T, "note", ORCH, NoteBody(text="after")))
        await asyncio.sleep(0.05)
        assert not follower.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await victim
        await follower

    run(main())
    entries = AgentBus(tmp_path).read(T)
    assert [e.body.text for e in entries] == ["n0", "victim", "after"]
    assert all(e.hash_ok() for e in entries) and entries[2].prev == entries[1].hash