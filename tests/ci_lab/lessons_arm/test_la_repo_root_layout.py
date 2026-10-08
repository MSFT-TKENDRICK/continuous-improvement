"""The guard arm in a repo-root slot writes where the domain's agent loads guards (review fix)."""
from __future__ import annotations

import asyncio
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
from ci_lab.domain.layout import guard_extractors, guards_rel
from ci_lab.domain.order_support import HARNESS_ROOT, OrderSupportDomain
from ci_lab.guards.domains.order_support import EXTRACTORS, SEED_RULES
from ci_lab.lessons_arm.paired import harness_bundle_digest, paired_eval
from ci_lab.rulespec import Fingerprint, LessonCluster
from ci_lab.strategies import get_strategy
from ci_lab.strategies.base import edit_scope_violations

GUARDS = f"{HARNESS_ROOT}/guards"


def git(wt: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=wt, check=True, capture_output=True, text=True).stdout.strip()


class RepoRootDomain:
    """Order-support layout (``src/order_support/harness``) with a trivial, deterministic evaluate."""

    name = "order_support"
    surface_globs = OrderSupportDomain.surface_globs
    frozen_globs = OrderSupportDomain.frozen_globs
    component_globs = OrderSupportDomain.component_globs
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
    (wt / HARNESS_ROOT / "agent.yaml").write_text("name: order-support\n", encoding="utf-8")
    (wt / "README.md").write_text("repo\n", encoding="utf-8")
    git(wt, "init", "-q")
    git(wt, "-c", "user.name=t", "-c", "user.email=t@x", "add", "-A")
    git(wt, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "base")
    round_dir = tmp_path / "round"
    run_dir = round_dir / "arm-g"
    run_dir.mkdir(parents=True)
    (round_dir / "lessons").mkdir()
    cl = LessonCluster(id="c-elig", fingerprint=Fingerprint(pin="p1", oracle_rules=("refund.ineligible_order",),
                                                             tool_ngrams=(("lookup_order", "issue_refund"),)),
                       members=("t1", "t2"), families=("f1", "f2"), slices=("s1", "s2"), route="R2")
    (round_dir / "lessons" / "candidates.jsonl").write_text(cl.model_dump_json() + "\n", encoding="utf-8")
    return wt, git(wt, "rev-parse", "HEAD"), run_dir


def test_order_support_layout_helpers():
    os_domain = OrderSupportDomain(cases=[], use_default_oracle=False, scope_factory=lambda key: None,
                                   runner=lambda *a, **kw: None)  # type: ignore[arg-type]
    assert guards_rel(os_domain) == GUARDS == "src/order_support/harness/guards"
    assert guards_rel(HarnessDomain(os_domain, HARNESS_ROOT)) == GUARDS  # type: ignore[arg-type]
    assert guard_extractors(HarnessDomain(os_domain, HARNESS_ROOT)) == [EXTRACTORS]  # type: ignore[arg-type]
    assert guards_rel(None) == "harness/guards"


def test_guard_arm_writes_where_the_agent_loads_and_paired_eval_sees_it(slot, tmp_path: Path):
    from order_support.guarding import guards_dir

    wt, base, run_dir = slot
    inner = RepoRootDomain()
    domain = HarnessDomain(inner, HARNESS_ROOT)  # type: ignore[arg-type]
    before = harness_bundle_digest(wt, guards_dir=guards_rel(domain), extractors=guard_extractors(domain))

    strategy = get_strategy("guard", domain=domain)
    ctx = ArmContext(experiment_id="exp1", directive=ArmDirective(arm="arm-g", strategy="guard", edit_budget=1),
                     worktree=wt, base_commit=base, failures=[], profile=Profile.FAKE, run_dir=run_dir)
    [edit] = asyncio.run(strategy.propose(ctx))
    rel = f"{GUARDS}/c-elig.yaml"
    assert edit.files == (rel,) and (wt / rel).is_file() and not (wt / "harness").exists()
    assert git(wt, "diff", "--name-only", base, "HEAD") == rel
    assert edit_scope_violations("guard", [rel], guards_dir=guards_rel(domain)) == []

    # the agent (ORDER_SUPPORT_HARNESS_DIR = <slot>/src/order_support/harness) loads that directory
    assert guards_dir(wt / HARNESS_ROOT) == (wt / GUARDS).resolve()
    after = harness_bundle_digest(wt, guards_dir=guards_rel(domain), extractors=guard_extractors(domain))
    assert after != before
    m = asyncio.run(paired_eval(domain, wt, "evolve", 1, experiment_id="exp1", variant="arm-g",
                                decisions_dir=tmp_path / "dec", stochastic=False))
    assert m.bundle_digest == after
    assert inner.harness_dirs and set(inner.harness_dirs) == {wt / HARNESS_ROOT}


@pytest.mark.parametrize("strategy", ["gepa", "skillopt", "agent"])
def test_text_strategies_may_not_write_domain_guards(strategy: str):
    files = [f"{GUARDS}/c-elig.yaml", "harness/guards/x.yaml", f"{HARNESS_ROOT}/prompts/system.md"]
    assert edit_scope_violations(strategy, files, guards_dir=GUARDS) == files[:2]
    # a harness root not named "harness" is still protected via guards_dir
    assert edit_scope_violations(strategy, ["agent/guards/r.yaml", "agent/p.md"],
                                 guards_dir="agent/guards") == ["agent/guards/r.yaml"]


def test_guard_strategy_confined_to_domain_guards_dir():
    files = [f"{GUARDS}/ok.yaml", "harness/guards/stray.yaml", f"{GUARDS}/BUNDLE.lock",
             f"{HARNESS_ROOT}/prompts/system.md"]
    assert edit_scope_violations("guard", files, guards_dir=GUARDS) == files[1:]
