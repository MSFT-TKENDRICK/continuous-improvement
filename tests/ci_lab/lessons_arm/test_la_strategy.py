from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from ci_lab.contracts import ArmContext, ArmDirective, Edit, Profile
from ci_lab.lessons_arm.agent import LessonSynthesizer, local_builder
from ci_lab.lessons_arm.bundle import read_rule_file
from ci_lab.lessons_arm.strategy import (
    COMMIT_TRAILER,
    GuardPathViolation,
    GuardStrategy,
    git_commit,
    guard_path,
    register,
)
from ci_lab.rulespec import Fingerprint, LessonCluster
from ci_lab.testing import Call, FakeChatClient


def git(wt: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=wt, check=True, capture_output=True, text=True).stdout.strip()


def cluster(rule: str, cid: str, *, route: str = "R2", **kw: object) -> LessonCluster:
    base: dict[str, object] = {"members": ("t1", "t2"), "families": ("f1", "f2"), "slices": ("s1", "s2"),
                               "route": route, **kw}
    return LessonCluster(id=cid, fingerprint=Fingerprint(pin="p1", oracle_rules=(rule,),
                                                         tool_ngrams=(("read_file", "write_file"),)),
                         **base)  # type: ignore[arg-type]


@pytest.fixture
def env(tmp_path: Path):
    wt = tmp_path / "wt"
    (wt / "harness" / "skills").mkdir(parents=True)
    (wt / "harness" / "skills" / "changes.md").write_text("# Changes\nAlways inspect the target.\n", encoding="utf-8")
    git(wt, "init", "-q")
    git(wt, "-c", "user.name=t", "-c", "user.email=t@x", "add", "-A")
    git(wt, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "base")
    base = git(wt, "rev-parse", "HEAD")
    round_dir = tmp_path / "round"
    run_dir = round_dir / "arm-g"
    run_dir.mkdir(parents=True)
    (round_dir / "lessons").mkdir()
    return wt, base, round_dir, run_dir


def write_candidates(round_dir: Path, *items: LessonCluster | dict) -> None:
    lines = [c.model_dump_json() if isinstance(c, LessonCluster) else json.dumps(c) for c in items]
    (round_dir / "lessons" / "candidates.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ctx(wt: Path, base: str, run_dir: Path, budget: int = 1, arm: str = "arm-g") -> ArmContext:
    return ArmContext(experiment_id="exp1", directive=ArmDirective(arm=arm, strategy="guard", edit_budget=budget),
                      worktree=wt, base_commit=base, failures=[], profile=Profile.FAKE, run_dir=run_dir)


def test_template_lessons_become_one_commit_per_edit(env):
    wt, base, round_dir, run_dir = env
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig"),
                     cluster("harness.amount_exceeds_limit", "c-amt", members=("a", "b", "c")),
                     cluster("harness.requires_access", "c-id"))
    edits = asyncio.run(GuardStrategy().propose(ctx(wt, base, run_dir, budget=2)))
    assert len(edits) == 2 and all(isinstance(e, Edit) and e.component == "guard" for e in edits)
    assert edits[0].files == ("harness/guards/c-amt.yaml",)  # more members first (same rung)
    for e in edits:
        assert git(wt, "diff", "--name-only", f"{e.commit}~1", e.commit) == e.files[0]
        assert COMMIT_TRAILER in git(wt, "log", "-1", "--format=%B", e.commit)
        rf = read_rule_file(wt / e.files[0])
        assert [r.mode for r in rf.rules] == ["shadow"] and rf.rules[0].provenance.source == "template"
    assert git(wt, "rev-parse", "HEAD") == edits[1].commit
    report = json.loads((run_dir / "optimizer" / "arm-g-guard.json").read_text(encoding="utf-8"))
    assert [x["lesson_id"] for x in report["edits"]] == ["c-amt", "c-elig"]
    assert report["replay"] == "unavailable"
    assert (round_dir / "lesson_claims" / "c-amt.json").exists()


def test_zero_budget_and_no_candidates(env):
    wt, base, _round_dir, run_dir = env
    assert asyncio.run(GuardStrategy().propose(ctx(wt, base, run_dir, budget=0))) == []
    assert asyncio.run(GuardStrategy().propose(ctx(wt, base, run_dir))) == []


def test_skip_reasons(env):
    wt, base, round_dir, run_dir = env
    feats_untrusted = {"kind": "prior_call", "target_tool": "write_file", "prior_tool": "read_file",
                       "subject_arg": "resource_id", "trusted": False}
    write_candidates(
        round_dir,
        cluster("injection.followed_instruction", "c-inj"),
        cluster("harness.disallowed_write", "c-rej", status="rejected"),
        cluster("x.y", "c-prose", route="R6"),
        {"cluster": cluster("x.z", "c-usage").model_dump(mode="json"), "features": feats_untrusted,
         "route_reasons": ["usage"]},
        cluster("harness.disallowed_write", "c-other"),
        cluster("x.leftover", "c-left"),
    )
    (wt / "lessons").mkdir()
    (wt / "lessons" / "registry.yaml").write_text(yaml.safe_dump({"schema_version": 1, "lessons": [
        {"lesson_id": "c-other", "prose_anchors": ["harness/skills/changes.md#Changes"]}]}), encoding="utf-8")
    sib = round_dir / "arm-p"
    sib.mkdir()
    (sib / "proposal.json").write_text(json.dumps({"strategy": "skillopt", "edits": [
        {"component": "skill", "hypothesis": "h", "files": ["harness/skills/changes.md"], "commit": "x"}]}),
        encoding="utf-8")
    edits = asyncio.run(GuardStrategy().propose(ctx(wt, base, run_dir, budget=3)))
    assert edits == []
    skipped = json.loads((run_dir / "optimizer" / "arm-g-guard.json").read_text(encoding="utf-8"))["skipped"]
    assert skipped == {"c-inj": "injection_suspect", "c-rej": "status_rejected", "c-prose": "route_R6",
                       "c-usage": "untrusted_unlabeled", "c-other": "touched_by_other_arm",
                       "c-left": "leftover_no_synthesizer"}


def test_leftover_uses_llm_synthesizer_after_templates(env):
    wt, base, round_dir, run_dir = env
    write_candidates(round_dir, cluster("x.leftover", "c-left", human_confirmed=True),
                     cluster("harness.disallowed_write", "c-elig"))
    client = FakeChatClient([[Call("submit_rule", {
        "skeleton": "arg_constraint", "target_tool": "write_file",
        "slots": {"arg": "amount", "op": "gt", "values": [0]}})], "done"])
    syn = LessonSynthesizer(client=client, builder=local_builder)
    edits = asyncio.run(GuardStrategy(synthesizer=syn).propose(ctx(wt, base, run_dir, budget=2)))
    assert [e.files[0] for e in edits] == ["harness/guards/c-elig.yaml", "harness/guards/c-left.yaml"]
    assert read_rule_file(wt / edits[1].files[0]).rules[0].provenance.source == "synthesizer"


def test_replay_rejection_leaves_no_trace(env):
    wt, base, round_dir, run_dir = env
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig"))
    seen = []

    def replay(rules, cl, work_dir):
        seen.append((next(iter(rules.rules if hasattr(rules, "rules") else rules)).id, cl.id, work_dir.is_dir()))
        return False, ["fp_ucb 0.2 > 0.02"]

    edits = asyncio.run(GuardStrategy(replay=replay).propose(ctx(wt, base, run_dir)))
    assert edits == [] and seen == [("lsn.c-elig.prior", "c-elig", True)]
    assert not (wt / "harness" / "guards" / "c-elig.yaml").exists()
    assert git(wt, "rev-parse", "HEAD") == base
    rep = json.loads((run_dir / "optimizer" / "arm-g-guard.json").read_text(encoding="utf-8"))
    assert rep["rejected"]["c-elig"] == ["replay", "fp_ucb 0.2 > 0.02"]
    # replay pass is recorded
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig2"))
    ok = asyncio.run(GuardStrategy(replay=lambda r, c, w: (True, [])).propose(ctx(wt, base, run_dir)))
    assert "replay pass_to_closed_loop" in ok[0].hypothesis
    # required replay without a validator rejects
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig3"))
    strat = GuardStrategy(require_replay=True)
    strat.replay = None
    assert asyncio.run(strat.propose(ctx(wt, base, run_dir))) == []


def test_real_m16_replay_rejects_without_evidence(env):
    pytest.importorskip("ci_lab.lessons.replay")
    wt, base, round_dir, run_dir = env
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig"))
    edits = asyncio.run(GuardStrategy(trajectories=[]).propose(ctx(wt, base, run_dir)))
    assert edits == [] and git(wt, "rev-parse", "HEAD") == base
    rep = json.loads((run_dir / "optimizer" / "arm-g-guard.json").read_text(encoding="utf-8"))
    assert rep["rejected"]["c-elig"][0] == "replay" and len(rep["rejected"]["c-elig"]) > 1


def test_bundle_failure_reverts(env):
    wt, base, round_dir, run_dir = env
    guards = wt / "harness" / "guards"
    guards.mkdir(parents=True)
    (guards / "broken.yaml").write_text("schema_version: 1\nrules: [{id: x}]\n", encoding="utf-8")
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig"))
    assert asyncio.run(GuardStrategy().propose(ctx(wt, base, run_dir))) == []
    assert not (guards / "c-elig.yaml").exists()
    rep = json.loads((run_dir / "optimizer" / "arm-g-guard.json").read_text(encoding="utf-8"))
    assert rep["rejected"]["c-elig"][0] == "bundle"


def test_repair_reproposal_is_idempotent_and_claims_are_exclusive(env):
    wt, base, round_dir, run_dir = env
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig"))
    first = asyncio.run(GuardStrategy().propose(ctx(wt, base, run_dir)))
    again = asyncio.run(GuardStrategy().propose(ctx(wt, base, run_dir)))
    assert again == first and git(wt, "rev-parse", "HEAD") == first[0].commit
    other_run = round_dir / "arm-h"
    other_run.mkdir()
    assert asyncio.run(GuardStrategy().propose(ctx(wt, base, other_run, arm="arm-h"))) == []


def test_path_guard_and_bad_committer(env):
    wt, base, round_dir, run_dir = env
    for bad in ("../evil", "a/b", "extractors", "", ".hidden"):
        with pytest.raises(GuardPathViolation):
            guard_path(wt, bad)
    assert guard_path(wt, "c-ok")[1] == "harness/guards/c-ok.yaml"

    def sneaky(worktree, files, message):
        (worktree / "harness" / "guards" / "BUNDLE.lock").write_text("x", encoding="utf-8")
        return git_commit(worktree, [*files, "harness/guards/BUNDLE.lock"], message)

    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig"))
    with pytest.raises(GuardPathViolation, match="only harness/guards/c-elig.yaml"):
        asyncio.run(GuardStrategy(committer=sneaky).propose(ctx(wt, base, run_dir)))


def test_enforce_write_mode(env):
    wt, base, round_dir, run_dir = env
    write_candidates(round_dir, cluster("harness.disallowed_write", "c-elig"))
    e = asyncio.run(GuardStrategy(write_mode="enforce").propose(ctx(wt, base, run_dir)))
    assert read_rule_file(wt / e[0].files[0]).rules[0].mode == "enforce"


def test_register_with_strategy_registry(monkeypatch):
    try:  # don't leak "guard" into the real registry when ci_lab.strategies is installed
        import ci_lab.strategies as real
    except ImportError:
        pass
    else:
        monkeypatch.setattr(real, "_FACTORIES", dict(real._FACTORIES))
    assert register() is False or "ci_lab.strategies" in sys.modules
    registered: dict[str, object] = {}
    fake = types.ModuleType("ci_lab.strategies")
    fake.register_strategy = lambda name, factory: registered.__setitem__(name, factory)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ci_lab.strategies", fake)
    assert register() is True and registered == {"guard": GuardStrategy}
