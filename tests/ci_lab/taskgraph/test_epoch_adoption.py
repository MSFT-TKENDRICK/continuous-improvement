from __future__ import annotations

import asyncio
import importlib.util
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from ci_lab.adversary.challenger import DeterministicChallenger
from ci_lab.adversary.harden import Hardener, TemplatePatcher
from ci_lab.bus import ids
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.types import Author, RubricPatchBody
from ci_lab.bus.voters.local import CallableVoter, DeterministicCheckVoter
from ci_lab.bus.wal import AgentBus
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
HARD = Author("hardener", "hardener")


def gullible_voters(_d: Any) -> list[Any]:
    return [DeterministicCheckVoter(), CallableVoter("soft", F.gullible, measure="s1", criteria=["c-cites"])]


class SlowHardener(Hardener):
    """Lets the task commit first, so the patch must go to the run topic (``applies_from_epoch = 1``)."""

    async def harden(self, *a: Any, **kw: Any) -> Any:
        await asyncio.sleep(0.2)
        return await super().harden(*a, **kw)


def go(root: Path, run_id: str, script: list[str], *, engine: str = "asyncio", harden: type[Hardener] | None = None,
       **kw: Any) -> GraphResult:
    vault, bus = RubricVault(root / "vault"), AgentBus(root / "bus")
    graph = F.make_graph(vault, {"a": ()})
    if harden is not None:
        kw |= {"challenger": DeterministicChallenger(["judge_injection"]),
               "hardener": harden(vault, root / "art", patcher=TemplatePatcher(random.Random(0)), seed=0)}
    if engine == "maf":
        kw["checkpoint_dir"] = root / f"ckpt-{run_id}"
    fn = run_graph_maf if engine == "maf" else run_graph
    return asyncio.run(fn(graph, bus=bus, vault=vault, voters_for=gullible_voters,
                          student_factory=F.FakeStudent({"a": script}), run_id=run_id, pools=ResourcePools({}), **kw))


def run_log(root: Path, run_id: str) -> tuple[list[Any], list[Any]]:
    """(epoch-0 adoption patches, adoption notes) on ``<run>/_run``."""
    entries = AgentBus(root / "bus").read(ids.run_topic(run_id))
    return ([e for e in entries if e.kind == "rubric_patch" and e.body.applies_from_epoch == 0],
            [e for e in entries if e.kind == "note" and "epoch_adoption" in e.body.data])


def judged_versions(root: Path, run_id: str) -> set[str]:
    st = AgentBus(root / "bus").state(ids.task_topic(run_id, "a"))
    return {e.body.rubric_version for e in st.entries if e.kind in ("proposal", "vote", "verdict")}


def harden_first_run(root: Path, engine: str = "asyncio") -> None:
    res = go(root, "r1", ["# Report\nOrder ORD-1.\n", F.GOOD], engine=engine, harden=Hardener)
    assert res.tasks["a"].rubric_versions == ("a-rubric@v1", "a-rubric@v2")
    patches, notes = run_log(root, "r1")
    assert patches == [] and [n.body.data for n in notes] == [{"epoch_adoption": {}, "enabled": True}]


def test_epoch_patch_of_a_finished_task_is_adopted_by_the_next_run(tmp_path: Path) -> None:
    assert go(tmp_path, "r1", [F.GOOD], harden=SlowHardener).tasks["a"].rubric_versions == ("a-rubric@v1",)
    epoch = [e.body for e in AgentBus(tmp_path / "bus").read(ids.run_topic("r1")) if e.kind == "rubric_patch"]
    assert [(b.to_version, b.applies_from_epoch) for b in epoch] == [("a-rubric@v2", 1)]
    assert go(tmp_path, "r2", [F.GOOD]).tasks["a"].rubric_versions == ("a-rubric@v2",)
    assert judged_versions(tmp_path, "r2") == {"a-rubric@v2"}


def test_next_run_adopts_hardened_rubric_and_resume_does_not_readopt(tmp_path: Path) -> None:
    harden_first_run(tmp_path)
    res = go(tmp_path, "r2", [F.GOOD])
    assert res.tasks["a"].status == "committed" and res.tasks["a"].rubric_versions == ("a-rubric@v2",)
    assert judged_versions(tmp_path, "r2") == {"a-rubric@v2"}
    (patch,), (note,) = run_log(tmp_path, "r2")
    assert patch.author == Author("hardener", "epoch-adopt") and patch.body.accepted
    assert (patch.body.from_version, patch.body.to_version, patch.body.applies_from_attempt) == \
        ("a-rubric@v1", "a-rubric@v2", None)
    assert note.body.data == {"epoch_adoption": {"a-rubric": "a-rubric@v2"}, "enabled": True}
    assert note.seq > patch.seq
    # a newer hardening appearing later must not change r2's decided baseline on resume
    vault = RubricVault(tmp_path / "vault")
    v3 = vault.seal(replace(vault.versions("a-rubric")[-1], version=3))
    assert vault.open(v3, role="orchestrator").version_id == "a-rubric@v3"
    asyncio.run(AgentBus(tmp_path / "bus").append(ids.task_topic("later", "a"), "rubric_patch", HARD, RubricPatchBody(
        rubric_id="a-rubric", from_version="a-rubric@v2", to_version="a-rubric@v3", applies_from_attempt=None,
        applies_from_epoch=1, diff_sha256="0" * 64, metrics={}, accepted=True)))
    before = len(AgentBus(tmp_path / "bus").read(ids.run_topic("r2")))
    again = go(tmp_path, "r2", ["unused"])
    assert again.tasks["a"].rubric_versions == ("a-rubric@v2",)
    assert len(AgentBus(tmp_path / "bus").read(ids.run_topic("r2"))) == before
    # ...while a fresh run follows the lineage v1 -> v2 -> v3
    assert go(tmp_path, "r3", [F.GOOD]).tasks["a"].rubric_versions == ("a-rubric@v3",)
    assert run_log(tmp_path, "r3")[1][0].body.data["epoch_adoption"] == {"a-rubric": "a-rubric@v3"}


def test_adoption_can_be_disabled(tmp_path: Path) -> None:
    harden_first_run(tmp_path)
    res = go(tmp_path, "r2", [F.GOOD], adopt_epoch_patches=False)
    assert res.tasks["a"].rubric_versions == ("a-rubric@v1",) and judged_versions(tmp_path, "r2") == {"a-rubric@v1"}
    patches, (note,) = run_log(tmp_path, "r2")
    assert patches == [] and note.body.data == {"epoch_adoption": {}, "enabled": False}


def test_only_sealed_newer_descendants_of_the_pinned_rubric_are_adopted(tmp_path: Path) -> None:
    vault, bus = RubricVault(tmp_path / "vault"), AgentBus(tmp_path / "bus")
    pinned = vault.open(F.make_graph(vault, {"a": ()}).deliverables[0].rubric_commitment, role="orchestrator")
    vault.seal(replace(pinned, version=2))
    vault.seal(replace(pinned, version=7))

    def patch(frm: int, to: int, accepted: bool = True, *, epoch: bool = False) -> RubricPatchBody:
        return RubricPatchBody(rubric_id="a-rubric", from_version=f"a-rubric@v{frm}", to_version=f"a-rubric@v{to}",
                               applies_from_attempt=None if epoch else 2, applies_from_epoch=1 if epoch else None,
                               diff_sha256="0" * 64, metrics={}, accepted=accepted)

    for frm, to, ok in ((1, 9, True), (5, 7, True), (1, 7, False), (1, 2, True)):  # unsealed, foreign, rejected
        asyncio.run(bus.append(ids.task_topic("old", "a"), "rubric_patch", HARD, patch(frm, to, ok)))
    asyncio.run(bus.append(ids.task_topic("own", "a"), "rubric_patch", HARD, patch(2, 7)))
    assert go(tmp_path, "own", [F.GOOD]).tasks["a"].rubric_versions == ("a-rubric@v2",)  # own run's patch ignored


def test_maf_engine_adopts_like_asyncio(tmp_path: Path) -> None:
    for engine in ("asyncio", "maf"):
        root = tmp_path / engine
        harden_first_run(root, engine)
        res = go(root, "r2", [F.GOOD], engine=engine)
        assert res.tasks["a"].rubric_versions == ("a-rubric@v2",) and judged_versions(root, "r2") == {"a-rubric@v2"}
        (patch,), (note,) = run_log(root, "r2")
        assert patch.body.to_version == "a-rubric@v2" and note.body.data["epoch_adoption"] == {"a-rubric": "a-rubric@v2"}
        again = go(root, "r2", ["unused"], engine=engine)
        assert again.tasks["a"].rubric_versions == ("a-rubric@v2",) and len(run_log(root, "r2")[0]) == 1
