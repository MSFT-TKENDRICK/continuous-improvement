from __future__ import annotations

import asyncio
import importlib.util
import inspect
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from ci_lab.adversary.challenger import DeterministicChallenger
from ci_lab.adversary.harden import Hardener, TemplatePatcher
from ci_lab.bus import ids
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.voters.local import CallableVoter, DeterministicCheckVoter
from ci_lab.bus.wal import AgentBus
from ci_lab.maf.workflows import CheckpointNotWrittenError
from ci_lab.taskgraph.maf_engine import run_graph_maf
from ci_lab.taskgraph.scheduler import GraphResult, run_graph
from ci_lab.taskgraph.vault import RubricVault


def _fakes() -> Any:  # reuse the scheduler test's fakes (tests are not a package under importlib mode)
    if (mod := sys.modules.get("_taskgraph_scheduler_fakes")) is None:
        spec = importlib.util.spec_from_file_location("_taskgraph_scheduler_fakes",
                                                      Path(__file__).with_name("test_scheduler.py"))
        assert spec is not None and spec.loader is not None
        mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


F = _fakes()
GOOD = F.GOOD
EXPLOIT = "# Report\nResource RES-1.\n"

GRAPHS: dict[str, tuple[dict[str, tuple[str, ...]], dict[str, list[str]]]] = {
    "chain": ({"a": (), "b": ("a",), "c": ("b",)}, {"a": ["draft", GOOD], "b": [GOOD], "c": [GOOD]}),
    "diamond": ({"a": (), "b": ("a",), "c": ("a",), "d": ("b", "c")},
                {"a": [GOOD], "b": [GOOD], "c": ["nope", GOOD], "d": [GOOD]}),
    "wide": ({"r": (), **{f"w{i}": ("r",) for i in range(5)}, "z": tuple(f"w{i}" for i in range(5))},
             {"r": [GOOD], **{f"w{i}": [GOOD] for i in range(5)}, "z": [GOOD]}),
    "blocked": ({"a": (), "b": (), "c": ("a", "b"), "d": ("c",), "e": ("a", "d"), "f": ("b",), "g": ("f",),
                 "h": ("g",)},
                {"a": ["nope", "still nope"], "b": [GOOD], "c": [GOOD], "d": [GOOD], "e": [GOOD],
                 "f": [GOOD + F.CANARY], "g": [GOOD], "h": [GOOD]}),
    "skewed": ({"a": (), "b": ("a",), "c": ("b",), "d": ("a", "c")}, {t: [GOOD] for t in "abcd"}),
}


def fingerprint(res: GraphResult) -> dict[str, tuple[Any, ...]]:
    return {t: (r.status, r.attempts, r.rubric_versions, r.reason, r.score, r.exploits)
            for t, r in res.tasks.items()}


def engine_run(root: Path, engine: str, deps: dict[str, tuple[str, ...]], script: dict[str, list[str]],
               **kw: Any) -> tuple[GraphResult, Any]:
    vault, student = RubricVault(root / "vault"), F.FakeStudent(script, delay=kw.pop("delay", 0.0))
    graph = F.make_graph(vault, deps)
    extra = {"checkpoint_dir": root / "ckpt"} if engine == "maf" else {}
    fn = run_graph_maf if engine == "maf" else run_graph
    res = asyncio.run(fn(graph, bus=AgentBus(root / "bus"), vault=vault, voters_for=kw.pop("voters_for", F.voters),
                         student_factory=student, run_id="r1", pools=ResourcePools({}), **extra, **kw))
    return res, student


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_conformance_with_asyncio_engine(tmp_path: Path, name: str) -> None:
    deps, script = GRAPHS[name]
    ref, ref_student = engine_run(tmp_path / "asyncio", "asyncio", deps, script)
    got, student = engine_run(tmp_path / "maf", "maf", deps, script)
    assert fingerprint(got) == fingerprint(ref)
    assert sorted(t for t, _, _ in student.calls) == sorted(t for t, _, _ in ref_student.calls)
    assert AgentBus(tmp_path / "maf" / "bus").read(ids.run_topic("r1"))[0].kind == "manifest"
    assert any((tmp_path / "maf" / "ckpt").glob("*.json"))


def test_blocked_dependency_reaches_fan_in_without_hanging(tmp_path: Path) -> None:
    deps, script = GRAPHS["blocked"]
    got, student = engine_run(tmp_path, "maf", deps, script)
    assert {t: r.status for t, r in got.tasks.items()} == {
        "a": "rejected", "b": "committed", "c": "blocked", "d": "blocked", "e": "blocked", "f": "committed",
        "g": "aborted", "h": "blocked"}
    assert got.tasks["c"].reason == "dependency_blocked" and got.tasks["c"].attempts == 0
    assert {t for t, _, _ in student.calls} == {"a", "b", "f"} and got.tasks["g"].reason == "context_leak"


def test_exploit_and_hardening_conform(tmp_path: Path) -> None:
    def run(engine: str) -> GraphResult:
        root = tmp_path / engine
        hardener = Hardener(RubricVault(root / "vault"), root / "art", patcher=TemplatePatcher(random.Random(0)),
                            seed=0)
        res, _ = engine_run(root, engine, {"a": (), "b": ("a",)}, {"a": [EXPLOIT, GOOD], "b": [GOOD]},
                            voters_for=lambda _d: [DeterministicCheckVoter(),
                                                   CallableVoter("soft", F.gullible, measure="s1",
                                                                 criteria=["c-cites"])],
                            challenger=DeterministicChallenger(["judge_injection"]), hardener=hardener)
        return res

    ref, got = run("asyncio"), run("maf")
    assert fingerprint(got) == fingerprint(ref)
    assert got.tasks["a"].rubric_versions == ("a-rubric@v1", "a-rubric@v2") and got.tasks["a"].exploits >= 1


def test_independent_deliverables_run_concurrently_in_a_superstep(tmp_path: Path) -> None:
    deps, script = {"x": (), "y": (), "z": ("x", "y")}, {t: [GOOD] for t in "xyz"}
    _, par = engine_run(tmp_path / "par", "maf", deps, script, delay=0.4)
    (x0, x1), (y0, y1), (z0, _) = par.spans["x"], par.spans["y"], par.spans["z"]
    assert x0 < y1 and y0 < x1 and z0 >= max(x1, y1)
    _, ser = engine_run(tmp_path / "ser", "maf", deps, script, delay=0.4, max_parallel=1)
    (x0, x1), (y0, y1) = ser.spans["x"], ser.spans["y"]
    assert x1 <= y0 or y1 <= x0


def test_resume_after_crash_comes_from_the_bus(tmp_path: Path) -> None:
    vault = RubricVault(tmp_path / "vault")
    graph = F.make_graph(vault, {"a": (), "b": ("a",), "c": ("a", "b")})
    bus = AgentBus(tmp_path / "bus")
    real = bus.append

    dead: list[bool] = []

    async def crashing(topic: str, kind: Any, *a: Any, **kw: Any) -> Any:
        if topic.endswith("/b") and kind == "vote" or dead:  # crash, and nothing more reaches the bus
            dead.append(True)
            raise F.Crash
        return await real(topic, kind, *a, **kw)

    bus.append = crashing  # type: ignore[method-assign]
    first = F.FakeStudent({t: [GOOD] for t in "abc"})
    kw: dict[str, Any] = {"vault": vault, "voters_for": F.voters, "run_id": "r1", "pools": ResourcePools({}),
                          "checkpoint_dir": tmp_path / "ckpt"}
    with pytest.raises(F.Crash):
        asyncio.run(run_graph_maf(graph, bus=bus, student_factory=first, **kw))
    mid = GraphResult.from_bus(AgentBus(tmp_path / "bus"), graph, "r1")
    assert mid.tasks["a"].status == "committed" and mid.tasks["b"].status == mid.tasks["c"].status == "pending"
    assert len(AgentBus(tmp_path / "bus").state(ids.task_topic("r1", "b")).intents_without_outcome()) == 1
    stale = set((tmp_path / "ckpt").glob("*.json"))
    assert stale  # the crashed workflow left checkpoints; they are discarded, never resumed
    second = F.FakeStudent({t: ["# Report\nResources RES-7 and RES-8.\n"] for t in "abc"})
    res = asyncio.run(run_graph_maf(graph, bus=AgentBus(tmp_path / "bus"), student_factory=second, **kw))
    assert [t for t, _, _ in second.calls] == ["c"]  # a committed, b's in-flight proposal reused
    assert {t: (r.status, r.attempts) for t, r in res.tasks.items()} == {
        "a": ("committed", 1), "b": ("committed", 1), "c": ("committed", 1)}
    assert fingerprint(res) == fingerprint(GraphResult.from_bus(AgentBus(tmp_path / "bus"), graph, "r1"))
    assert not stale & set((tmp_path / "ckpt").glob("*.json"))


def test_missing_checkpoints_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ci_lab.taskgraph.maf_engine as engine

    monkeypatch.setattr(engine, "CHECKPOINT_TYPES", [])  # in-flight messages can no longer be checkpointed
    with pytest.raises(CheckpointNotWrittenError):
        engine_run(tmp_path, "maf", {"a": (), "b": ("a",)}, {"a": [GOOD], "b": [GOOD]})


def test_same_signature_as_run_graph() -> None:
    ours = dict(inspect.signature(run_graph_maf).parameters)
    assert ours.pop("checkpoint_dir").default is None
    assert ours == dict(inspect.signature(run_graph).parameters)
