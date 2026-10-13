from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ci_lab.contracts import (
    ArmResult,
    CriticVerdict,
    Edit,
    EvalResult,
    EvaluatorPin,
    TaskScore,
    Violation,
)
from ci_lab.oes import canonical
from ci_lab.oes.build import (
    ConfirmStats,
    Holdout,
    RrsiParams,
    Schedule,
    calibration_envelope,
    confirm_envelope,
    round_envelope,
    sleep_envelope,
)

AT = "2026-10-08T05:00:00Z"
VERSION = "0.2.0"
FIXTURES = Path(__file__).parent / "fixtures"
PIN = EvaluatorPin("e0e1e2e3e4e5e6e7e8e9e0e1e2e3e4e5e6e7e8e9", "gpt-judge", "GitHubCopilot", ("gpt-judge-2026-09",))
SPLITS = {"evolve": "sha256:" + "1" * 64, "heldout": "sha256:" + "2" * 64}
H0, H1, H2 = "a" * 40, "b" * 40, "c" * 40
T0, T1, T2 = "d0" * 20, "d1" * 20, "d2" * 20
CRIT = Violation("change.unverified_identity", "critical", "change before verification")
PARAMS = RrsiParams(delta=0.02, delta_method="aa_bootstrap_q95", beta0=0.10, beta1=44.5, ws=0.0, wc=15.0, wn=0.5)


def make_eval(tree: str, split: str = "evolve", *, base: float = 0.6, n: int = 9, missing: int = 0,
              crit: int = 0, tokens: int = 150, pin: EvaluatorPin = PIN) -> EvalResult:
    scores = []
    for i in range(n):
        score = None if i < missing else round(min(1.0, base + (i % 3 - 1) * 0.1), 3)
        scores.append(TaskScore(case_id=f"c{i:02d}", trial=0, suite=("change", "identity", "tone")[i % 3],
                                score=score, violations=(CRIT,) if i < crit else (), tokens_in=tokens - 50,
                                tokens_out=50, served_model="gpt-target"))
    return EvalResult(harness_tree=tree, split=split, pin=pin, scores=scores)  # type: ignore[arg-type]


def make_arm(name: str, head: str | None, tree: str | None, ev: EvalResult | None, *,
             passed: bool = True) -> ArmResult:
    edits = [Edit("prompt", f"{name}: clarify change policy", ("harness/prompts/system.md",),
                  head)] if head else []
    return ArmResult(arm=name, base_commit=H0, head_commit=head, harness_tree=tree, edits=edits,
                     critic=CriticVerdict(passed=passed, reasons=[] if passed else ["touches frozen path"]),
                     eval=ev, status="evaluated" if ev else "rejected")


def selection(winner: str | None = "v1") -> dict[str, Any]:
    return {
        "winner": winner, "incumbentScore": 0.6, "newIncumbentScore": 0.7 if winner else 0.6,
        "candidates": [
            {"variantId": "v1", "deltaS": 0.1, "deltaC": 0.0, "novelty": 0.2, "ciLowerBound": 0.04,
             "ciLevel": 0.95, "rule": "cost", "objective": None, "admissible": winner == "v1", "reasons": []},
            {"variantId": "v2", "deltaS": -0.05, "deltaC": 0.0, "novelty": 0.1, "ciLowerBound": -0.1,
             "ciLevel": 0.95, "rule": "weighted", "objective": -0.7, "admissible": False,
             "reasons": ["safety guard"]}]}


def build_calibration(**kw: Any) -> dict[str, Any]:
    runs = kw.pop("runs", None) or [make_eval(T0, "aa", base=0.6), make_eval(T0, "aa", base=0.61),
                                    make_eval(T0, "aa", base=0.605)]
    return calibration_envelope("tone-a1", runs, delta=0.02, delta_method="aa_bootstrap_q95", harness_commit=H0,
                                split_hashes=SPLITS, judge_errors={"rep0": 0, "rep1": 1, "rep2": 0},
                                exported_at=AT, source_version=VERSION, **kw)


def build_round(*, winner: str | None = "v1", incumbent: EvalResult | None = None, **kw: Any) -> dict[str, Any]:
    arms = kw.pop("arms", None) or [make_arm("v1", H1, T1, make_eval(T1, base=0.7)),
                                    make_arm("v2", H2, T2, make_eval(T2, base=0.55, crit=1)),
                                    make_arm("v3", None, None, None, passed=False)]
    return round_envelope(
        "tone-a1", kw.pop("round_no", 1), incumbent=incumbent or make_eval(T0), incumbent_commit=H0, arms=arms,
        selection=kw.pop("selection", None) or selection(winner),
        schedule=Schedule(budget=2, stall=False, exploration_slots=1, prune_set=("config",)), params=PARAMS,
        split_hashes=SPLITS, archive_refs={"v2": "exp-archive/tone-a1-r01/v2"},
        judge_errors={"inc": 0, "v1": 0, "v2": 1}, exported_at=AT, source_version=VERSION,
        artifacts=[{"type": "csv", "uri": "experiments/campaigns/tone-a1/r01/eval.csv"}], **kw)


def build_confirm(*, p_value: float = 0.01, final_crit: int = 0, **kw: Any) -> dict[str, Any]:
    return confirm_envelope(
        "tone-a1", baseline=make_eval(T0, "heldout", base=0.6),
        final=make_eval(T1, "heldout", base=0.7, crit=final_crit), baseline_commit=H0, final_commit=H1,
        stats=ConfirmStats(p_value=p_value, ci_lower=0.03 if p_value < 0.05 else -0.01, ci_upper=0.17),
        holdout=kw.pop("holdout", None) or Holdout(dataset_hash="sha256:" + "2" * 64, looks_used=1),
        look_ledger_ref="experiments/holdout-looks.jsonl", split_hashes=SPLITS,
        registered_at="2026-10-01T00:00:00Z", ood=(make_eval(T0, "ood", base=0.5), make_eval(T1, "ood", base=0.52)),
        accepted_rounds=["tone-a1-r01"], exported_at=AT, source_version=VERSION, **kw)


def build_sleep(*, candidate: bool = True, assert_passed: bool = True, **kw: Any) -> dict[str, Any]:
    return sleep_envelope(
        "2026-10-08", incumbent=make_eval(T0), candidate=make_eval(T1, base=0.65) if candidate else None,
        incumbent_commit=H0, skillopt_version="0.2.0",
        tasks_by_origin={"reviewed": 12, "harvested": 20}, tasks_by_split={"evolve": 32, "heldout": 0, "ood": 0},
        skillopt_gate={"passed": True, "mode": "on", "score": 0.7, "baselineScore": 0.62},
        assert_gate={"passed": assert_passed, "ciLowerBound": 0.01}, delta=0.02,
        budget_used={"tasks": 32, "rollouts": 64, "tokens": 120000, "wallClockSeconds": 1800.5},
        budget_limits={"tasks": 50, "rollouts": 200, "tokens": 500000, "wallClockSeconds": 7200},
        candidate_digest="sha256:" + "3" * 64 if candidate else None, incumbent_digest="sha256:" + "4" * 64,
        night_index=7, skill_path="harness/skills/harness-agent/SKILL.md",
        exported_at=AT, source_version=VERSION, **kw)


BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "calibration": build_calibration, "round": build_round, "confirm": build_confirm, "sleep": build_sleep}


def reseal(doc: dict[str, Any]) -> dict[str, Any]:
    """Recompute resultHash + contentHash so a mutation is judged by the semantic rules only."""
    doc = copy.deepcopy(doc)
    if "results" in doc and "resultHash" in doc.get("provenance", {}):
        doc["provenance"]["resultHash"] = canonical.digest(doc["results"])
    return canonical.seal(doc)


@pytest.fixture
def envelopes() -> dict[str, dict[str, Any]]:
    return {k: b() for k, b in BUILDERS.items()}


@pytest.fixture
def fx() -> Any:
    """This module (factories + constants); test modules can't import conftest under --import-mode=importlib."""
    import sys
    return sys.modules[__name__]
