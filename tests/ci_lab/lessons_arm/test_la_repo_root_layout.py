"""The guard arm in a repo-root slot writes where the domain's agent loads guards (review fix)."""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ci_lab.campaign.wiring import HarnessDomain
from ci_lab.contracts import (
    ArmContext,
    ArmDirective,
    EvalResult,
    EvaluatorPin,
    Profile,
    TaskScore,
)
from ci_lab.domain.harness import HarnessDomain as EvaluationHarnessDomain
from ci_lab.domain.layout import guard_extractors, guards_rel
from ci_lab.lessons_arm.paired import harness_bundle_digest, paired_eval
from ci_lab.rulespec import Fingerprint, LessonCluster
from ci_lab.strategies import get_strategy
from ci_lab.strategies.base import edit_scope_violations

HARNESS_ROOT = "harness"
GUARDS = f"{HARNESS_ROOT}/guards"
FIXTURES = Path(__file__).parents[1] / "guards" / "fixtures"
EXTRACTORS = FIXTURES / "extractors.yaml"
SEED_RULES = FIXTURES / "rules.yaml"


def git(wt: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=wt, check=True, capture_output=True, text=True).stdout.strip()


class RepoRootDomain:
    """Harness layout with a trivial, deterministic evaluate."""

    name = "harness"
    surface_globs = EvaluationHarnessDomain.surface_globs
    frozen_globs = EvaluationHarnessDomain.frozen_globs
    component_globs = EvaluationHarnessDomain.component_globs
    guard_extractors = (EXTRACTORS,)

    def __init__(self) -> None:
        self.harness_dirs: list[Path] = []

    def splits(self):
        return {"evolve": ("c1", "c2")}

    async def evaluate(self, harness_dir, split, k, *, experiment_id, variant):
        self.harness_dirs.append(Path(harness_dir))
        return EvalResult("sha256:x", split, EvaluatorPin("e", "j", "fake"),  # type: ignore[arg-type]
                          [TaskScore(c, t, "s", 1.0) for c in self.splits()[split] for t in range(k)])

    def failures(self, result):
        return []


@pytest.fixture
def slot(tmp_path: Path):
    wt = tmp_path / "wt"
    guards = wt / GUARDS
    guards.mkdir(parents=True)
    shutil.copy(SEED_RULES, guards / SEED_RULES.name)
    (wt / HARNESS_ROOT / "agent.yaml").write_text("name: harness-agent\n", encoding="utf-8")
    (wt / "README.md").write_text("repo\n", encoding="utf-8")
    git(wt, "init", "-q")
    git(wt, "-c", "user.name=t", "-c", "user.email=t@x", "add", "-A")
    git(wt, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "base")
    round_dir = tmp_path / "round"
    run_dir = round_dir / "arm-g"
    run_dir.mkdir(parents=True)
    (round_dir / "lessons").mkdir()
    cl = LessonCluster(id="c-elig", fingerprint=Fingerprint(pin="p1", oracle_rules=("action.item_unapproved",),
                                                             tool_ngrams=(("inspect_item", "apply_change"),)),
                       members=("t1", "t2"), families=("f1", "f2"), slices=("s1", "s2"), route="R2")
    candidate = {
        "cluster": cl.model_dump(mode="json"),
        "features": {
            "kind": "prior_call",
            "target_tool": "apply_change",
            "prior_tool": "inspect_item",
            "subject_arg": "item_id",
            "prior_result_equals": {"approved": True},
        },
    }
    (round_dir / "lessons" / "candidates.jsonl").write_text(
        json.dumps(candidate) + "\n",
        encoding="utf-8",
    )
    return wt, git(wt, "rev-parse", "HEAD"), run_dir


def test_harness_layout_helpers():
    domain = RepoRootDomain()
    assert guards_rel(domain) == GUARDS == "harness/guards"
    assert guards_rel(HarnessDomain(domain, HARNESS_ROOT)) == GUARDS  # type: ignore[arg-type]
    assert guard_extractors(HarnessDomain(domain, HARNESS_ROOT)) == [EXTRACTORS]  # type: ignore[arg-type]
    assert guards_rel(None) == "harness/guards"


def test_guard_arm_writes_where_the_agent_loads_and_paired_eval_sees_it(slot, tmp_path: Path):
    wt, base, run_dir = slot
    inner = RepoRootDomain()
    domain = HarnessDomain(inner, HARNESS_ROOT)  # type: ignore[arg-type]
    before = harness_bundle_digest(wt, guards_dir=guards_rel(domain), extractors=guard_extractors(domain))

    strategy = get_strategy("guard", domain=domain)
    ctx = ArmContext(experiment_id="exp1", directive=ArmDirective(arm="arm-g", strategy="guard", edit_budget=1),
                     worktree=wt, base_commit=base, failures=[], profile=Profile.FAKE, run_dir=run_dir)
    [edit] = asyncio.run(strategy.propose(ctx))
    rel = f"{GUARDS}/c-elig.yaml"
    assert edit.files == (rel,) and (wt / rel).is_file()
    assert git(wt, "diff", "--name-only", base, "HEAD") == rel
    assert edit_scope_violations("guard", [rel], guards_dir=guards_rel(domain)) == []

    after = harness_bundle_digest(wt, guards_dir=guards_rel(domain), extractors=guard_extractors(domain))
    assert after != before
    m = asyncio.run(paired_eval(domain, wt, "evolve", 1, experiment_id="exp1", variant="arm-g",
                                decisions_dir=tmp_path / "dec", stochastic=False))
    assert m.bundle_digest == after
    assert inner.harness_dirs and set(inner.harness_dirs) == {wt / HARNESS_ROOT}


@pytest.mark.parametrize("strategy", ["gepa", "skillopt", "agent"])
def test_text_strategies_may_not_write_domain_guards(strategy: str):
    files = [f"{GUARDS}/c-elig.yaml", "other/guards/x.yaml", f"{HARNESS_ROOT}/prompts/system.md"]
    assert edit_scope_violations(strategy, files, guards_dir=GUARDS) == files[:1]
    # a harness root not named "harness" is still protected via guards_dir
    assert edit_scope_violations(strategy, ["agent/guards/r.yaml", "agent/p.md"],
                                 guards_dir="agent/guards") == ["agent/guards/r.yaml"]


def test_guard_strategy_confined_to_domain_guards_dir():
    files = [f"{GUARDS}/ok.yaml", "other/guards/stray.yaml", f"{GUARDS}/BUNDLE.lock",
             f"{HARNESS_ROOT}/prompts/system.md"]
    assert edit_scope_violations("guard", files, guards_dir=GUARDS) == files[1:]
