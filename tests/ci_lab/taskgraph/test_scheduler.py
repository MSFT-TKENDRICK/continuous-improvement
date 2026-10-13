from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ci_lab.adversary.challenger import DeterministicChallenger
from ci_lab.adversary.harden import Hardener, TemplatePatcher
from ci_lab.bus import ids
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.voters.local import Ballot, CallableVoter, DeterministicCheckVoter
from ci_lab.bus.wal import AgentBus
from ci_lab.taskgraph.model import (
    Budget,
    Criterion,
    Deliverable,
    OutputSpec,
    Rubric,
    StudentSpec,
    TaskGraph,
)
from ci_lab.taskgraph.scheduler import GraphResult, run_graph
from ci_lab.taskgraph.validate import validate_rubric
from ci_lab.taskgraph.vault import RubricVault

CANARY = "0123456789abcdef"
QUESTION = "Does the report cite at least two distinct order identifiers taken from the log?"
GOOD = "# Report\nOrders ORD-1 and ORD-2 were escalated.\n"


def rubric(task: str, version: int = 1, extra: tuple[Criterion, ...] = ()) -> Rubric:
    return Rubric(f"{task}-rubric", version, task, (
        Criterion("c-format", "Report starts with a level one heading", "deterministic",
                  {"kind": "regex", "pattern": "^# Report", "independent": True}, 1.0, required=True),
        Criterion("c-cites", "Report cites order identifiers", "s1",
                  {"question": QUESTION, "type": "noul", "options": []}, 0.6), *extra), 0.7, CANARY)


def soft(_p: Any, artifact: bytes, _r: Any, _s: Any) -> Ballot:
    ok = artifact.count(b"ORD-") >= 2
    return Ballot(ok, 1.0 if ok else 0.1, 1.0, () if ok else ("cite more orders",))


def voters(_d: Deliverable) -> list[Any]:
    return [DeterministicCheckVoter(), CallableVoter("soft", soft, measure="s1", criteria=["c-cites"])]


def make_graph(vault: RubricVault, deps: dict[str, tuple[str, ...]], *, attempts: int = 2,
               timeout_s: float = 30.0) -> TaskGraph:
    return TaskGraph("g", "goal", tuple(
        Deliverable(t, f"Task {t}", f"Write the {t} report.", OutputSpec("file", f"out/{t}.md"), (), d,
                    vault.seal(rubric(t)), Budget(max_attempts=attempts, timeout_s=timeout_s))
        for t, d in deps.items()))


class FakeStudent:
    """Scripted artifacts per task and attempt; records every agent it builds and each message."""

    def __init__(self, script: dict[str, list[str]], delay: float = 0.0) -> None:
        self.script, self.delay = script, delay
        self.calls: list[tuple[str, str, Any]] = []
        self.spans: dict[str, tuple[float, float]] = {}

    def __call__(self, spec: StudentSpec, middleware: Any) -> Any:
        async def run(message: str) -> Any:
            self.calls.append((spec.id, message, middleware))
            n = sum(t == spec.id for t, _, _ in self.calls)
            t0 = time.perf_counter()
            await asyncio.sleep(self.delay)
            self.spans[spec.id] = (t0, time.perf_counter())
            texts = self.script[spec.id]
            return SimpleNamespace(text=texts[min(n, len(texts)) - 1])
        return SimpleNamespace(run=run)


def go(tmp_path: Path, graph: TaskGraph, student: FakeStudent, run: str = "r1", **kw: Any) -> GraphResult:
    return asyncio.run(run_graph(graph, bus=AgentBus(tmp_path / "bus"), vault=RubricVault(tmp_path / "vault"),
                                 voters_for=voters, student_factory=student, run_id=run,
                                 pools=ResourcePools({}), **kw))


def test_commit_revise_reject_blocked_and_leak(tmp_path: Path) -> None:
    vault = RubricVault(tmp_path / "vault")
    graph = make_graph(vault, {"a": (), "b": ("a",), "c": (), "d": ("c",), "e": (), "f": ("e",)})
    student = FakeStudent({"a": ["draft only", GOOD], "b": [GOOD], "c": ["nope", "still nope"], "d": [GOOD],
                           "e": [GOOD + CANARY], "f": [GOOD]})
    res = go(tmp_path, graph, student)
    status = {t: (r.status, r.attempts, r.reason) for t, r in res.tasks.items()}
    assert status == {"a": ("committed", 2, None), "b": ("committed", 1, None), "c": ("rejected", 2, "attempts exhausted"),
                      "d": ("blocked", 0, "dependency_blocked"), "e": ("committed", 1, None),
                      "f": ("aborted", 0, "context_leak")}
    assert res.tasks["a"].rubric_versions == ("a-rubric@v1", "a-rubric@v1") and res.tasks["a"].score == 1.0
    assert res.tasks["b"].critical_path[-2:] == ("a", "b") and not res.ok
    json.dumps(res.to_json())
    bus = AgentBus(tmp_path / "bus")
    assert bus.read(ids.run_topic("r1"))[0].kind == "manifest"
    msgs = {(t, i): m for i, (t, m, _) in enumerate(student.calls)}
    a1, a2 = [m for (t, _), m in msgs.items() if t == "a"]
    assert "attempt 1 of 2" in a1 and "## Correction" not in a1
    assert "attempt 2 of 2" in a2 and "## Correction" in a2
    b_msg = next(m for (t, _), m in msgs.items() if t == "b")
    assert "ORD-1 and ORD-2" in b_msg and "### a" in b_msg
    assert {t for t, _, _ in student.calls} == {"a", "b", "c", "e"}
    for _, m, mw in student.calls:  # the student never sees rubric material, always behind the firewall
        assert CANARY not in m and QUESTION not in m and "^# Report" not in m and "c-format" not in m
        assert len(mw) == 2


def test_independent_deliverables_run_in_parallel_by_default(tmp_path: Path) -> None:
    vault = RubricVault(tmp_path / "vault")
    graph = make_graph(vault, {"x": (), "y": ()})
    par = FakeStudent({"x": [GOOD], "y": [GOOD]}, delay=0.4)
    assert go(tmp_path, graph, par, "par").ok
    (x0, x1), (y0, y1) = par.spans["x"], par.spans["y"]
    assert x0 < y1 and y0 < x1  # overlapping student runs
    ser = FakeStudent({"x": [GOOD], "y": [GOOD]}, delay=0.4)
    assert go(tmp_path, graph, ser, "ser", max_parallel=1).ok
    (x0, x1), (y0, y1) = ser.spans["x"], ser.spans["y"]
    assert x1 <= y0 or y1 <= x0


def test_attempt_timeout_aborts(tmp_path: Path) -> None:
    graph = make_graph(RubricVault(tmp_path / "vault"), {"slow": (), "next": ("slow",)}, timeout_s=0.05)
    res = go(tmp_path, graph, FakeStudent({"slow": [GOOD], "next": [GOOD]}, delay=2.0))
    assert (res.tasks["slow"].status, res.tasks["slow"].reason) == ("aborted", "timeout")
    assert res.tasks["next"].status == "blocked"


class Crash(BaseException):
    pass


def test_resume_after_crash_skips_committed_and_reuses_in_flight_attempt(tmp_path: Path) -> None:
    vault = RubricVault(tmp_path / "vault")
    graph = make_graph(vault, {"a": (), "b": ()})
    first = FakeStudent({"a": [GOOD], "b": [GOOD]})
    bus = AgentBus(tmp_path / "bus")
    real = bus.append

    async def crashing(topic: str, kind: Any, *a: Any, **kw: Any) -> Any:
        if topic.endswith("/b") and kind == "vote" or crashing.dead:  # type: ignore[attr-defined]
            crashing.dead = True  # type: ignore[attr-defined]
            raise Crash
        return await real(topic, kind, *a, **kw)

    crashing.dead = False  # type: ignore[attr-defined]
    bus.append = crashing  # type: ignore[method-assign]

    async def crash_run() -> None:
        await run_graph(graph, bus=bus, vault=vault, voters_for=voters, student_factory=first, run_id="r1",
                        pools=ResourcePools({}), max_parallel=1)

    with pytest.raises(BaseExceptionGroup) as info:
        asyncio.run(crash_run())
    assert info.group_contains(Crash)
    st = AgentBus(tmp_path / "bus").state(ids.task_topic("r1", "b"))
    assert st.commit is None and len(st.intents_without_outcome()) == 1 and len(st.proposals) == 1
    second = FakeStudent({"a": ["unused"], "b": ["unused"]})
    res = go(tmp_path, graph, second)
    assert second.calls == []  # a was committed, b's in-flight attempt reused its proposal
    assert {t: (r.status, r.attempts) for t, r in res.tasks.items()} == {"a": ("committed", 1), "b": ("committed", 1)}
    st = AgentBus(tmp_path / "bus").state(ids.task_topic("r1", "b"))
    assert [e.kind for e in st.entries].count("intent") == 1 and not st.intents_without_outcome()


def gullible(_p: Any, artifact: bytes, _r: Any, _s: Any) -> Ballot:
    text = artifact.decode().lower()
    ok = artifact.count(b"ORD-") >= 2 or any(w in text for w in ("grader", "evaluator", "previous instructions",
                                                                  "reviewer"))
    return Ballot(ok, 1.0 if ok else 0.1, 1.0, () if ok else ("cite more orders",))


def test_exploit_hardens_next_attempt_only(tmp_path: Path) -> None:
    vault = RubricVault(tmp_path / "vault")
    assert not validate_rubric(rubric("a"))
    graph = make_graph(vault, {"a": ()})
    student = FakeStudent({"a": ["# Report\nOrder ORD-1.\n", GOOD]})
    hardener = Hardener(vault, tmp_path / "art", patcher=TemplatePatcher(random.Random(0)), seed=0)
    bus = AgentBus(tmp_path / "bus")
    res = asyncio.run(run_graph(
        graph, bus=bus, vault=vault, student_factory=student, run_id="r1", pools=ResourcePools({}),
        voters_for=lambda _d: [DeterministicCheckVoter(), CallableVoter("soft", gullible, measure="s1",
                                                                          criteria=["c-cites"])],
        challenger=DeterministicChallenger(["judge_injection"]), hardener=hardener))
    st = bus.state(ids.task_topic("r1", "a"))
    exploit = next(e for e in st.entries if e.kind == "exploit")
    patch = next(e for e in st.entries if e.kind == "rubric_patch")
    first = st.proposal_by_id("a@1/student:student")
    verdicts = [v for v in st.verdicts.values() if v.ref == first.seq]
    assert exploit.body.attempt == "a@1" and exploit.body.gamer == "judge_injection"
    assert (patch.body.from_version, patch.body.to_version, patch.body.applies_from_attempt) == \
        ("a-rubric@v1", "a-rubric@v2", 2)
    assert len(verdicts) == 1 and verdicts[0].body.rubric_version == "a-rubric@v1" and verdicts[0].seq < patch.seq
    a = res.tasks["a"]
    assert (a.status, a.attempts, a.rubric_versions) == ("committed", 2, ("a-rubric@v1", "a-rubric@v2"))
    assert a.exploits >= 1
    advs = [e for e in st.proposals.values() if e.author.role == "adversary"]
    assert {e.body.attempt for e in advs} == {"a@1", "a@2"}
    assert all("previous instructions" not in m.lower() for _, m, _ in student.calls)
