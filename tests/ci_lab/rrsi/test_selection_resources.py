"""Resource regularization under the ``harness`` profile (contract v3.1 A13-A16)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ci_lab.contracts import (
    ArmResult,
    Edit,
    EvalResult,
    EvaluatorPin,
    TaskScore,
    Violation,
)
from ci_lab.rrsi import stats
from ci_lab.rrsi.params import profile
from ci_lab.rrsi.readjudicate import round_record, verify
from ci_lab.rrsi.selection import SelectionInputs, select

PIN = EvaluatorPin(evaluator_tree="evaltree", judge_model="judge", judge_provider="prov")
CRIT = Violation(rule_id="safety.crit", severity="critical", detail="x")
BASE = [0.5] * 8
GAIN = [0.75] * 8


def ev(scores, *, tokens=100, calls=4, wall=1000.0, complexity=100.0, crit=0, runtime=True, tree_valid=None):
    out = [TaskScore(case_id=f"c{i}", trial=0, suite="s", score=s, violations=(CRIT,) * crit if i == 0 else (),
                     tokens_in=tokens // 2, tokens_out=tokens - tokens // 2,
                     wall_ms=wall + 10 * i if runtime else 0.0, llm_calls=calls if runtime else 0,
                     tool_calls=calls // 2 if runtime else 0) for i, s in enumerate(scores)]
    surface = {} if complexity is None else {"complexity": complexity}
    if tree_valid is not None:
        surface["tree_valid"] = tree_valid
    return EvalResult(harness_tree="tree", split="evolve", pin=PIN, scores=out, surface=surface)


def arm(name, result):
    return ArmResult(arm=name, base_commit="inc", head_commit=f"h-{name}", harness_tree=f"t-{name}",
                     edits=[Edit(component="prompt", hypothesis="h", files=("p.md",), commit="c")], eval=result,
                     status="evaluated", strategy="agent")


def inputs(arms, *, prof="harness", inc=None, **kw):
    inc = inc or ev(BASE)
    s_star = stats.task_score(stats.index_scores(inc.scores), stats.universe([inc]))
    return SelectionInputs(round=1, incumbent=inc, arms=tuple(arms), s_star=s_star, delta=0.125,
                           hp=profile(prof, n_bootstrap=300, **kw))


def run(arms, **kw):
    return select(inputs(arms, **kw))


def trace(dec, name="a"):
    return next(t for t in dec.arms if t.arm == name)


def has(t, prefix):
    return any(r.startswith(prefix) for r in t.reasons)


def test_records_resource_deltas_and_params():
    dec = run([arm("a", ev(GAIN, calls=8, wall=2000.0, complexity=105.0))])
    t = trace(dec)
    assert dec.decision == "do_not_ship" and t.reasons == ("calls_cap: dCalls 1.0000 > 0.1500",)
    assert t.delta_x == pytest.approx(0.05) and t.complexity == 105.0
    assert t.calls_per_task == 12.0 and t.delta_calls == pytest.approx(1.0)
    assert t.wall_ms_p50 == pytest.approx(2035.0) and t.delta_wall == pytest.approx(1000.0 / 1035.0)
    assert not t.resources["passed"] and t.resources["caps"]["x_cap"]["passed"]
    assert set(t.resources["caps"]) == {"x_cap", "calls_cap"}
    assert dec.params["w_x"] == 0.5 and dec.params["require_resource_metrics"] is True
    json.dumps(dec.to_dict(), allow_nan=False)


def test_x_cap_rejects_cost_branch_gain():
    t = trace(run([arm("a", ev(GAIN, complexity=120.0))]))
    assert t.branch == "cost" and t.rule["passed"] and not t.admissible and has(t, "x_cap: dX 0.2000 > 0.1000")


def test_calls_cap_rejects_cost_and_weighted_branches():
    cost = trace(run([arm("a", ev(GAIN, calls=6))]))
    assert cost.branch == "cost" and not cost.admissible and has(cost, "calls_cap")
    weighted = trace(run([arm("a", ev(BASE, tokens=80, calls=6))]))
    assert weighted.branch == "weighted" and weighted.rule["passed"] and not weighted.admissible
    assert has(weighted, "calls_cap")
    assert trace(run([arm("a", ev(BASE, tokens=80, calls=6))], calls_cap=None)).admissible


def test_wall_cap_applies_only_when_set():
    slow = [arm("a", ev(BASE, tokens=80, wall=5000.0))]
    assert trace(run(slow)).admissible
    t = trace(run(slow, wall_cap=0.5))
    assert not t.admissible and has(t, "wall_cap") and t.resources["caps"]["wall_cap"]["passed"] is False


@pytest.mark.parametrize("kw", [{"complexity": None}, {"runtime": False}])
def test_missing_resource_metrics_inadmissible_when_required(kw):
    t = trace(run([arm("a", ev(GAIN, **kw))]))
    assert not t.admissible and "resource_metrics_missing" in t.reasons and t.resources["missing"]
    assert trace(run([arm("a", ev(GAIN, **kw))], require_resource_metrics=False)).admissible
    missing_inc = trace(run([arm("a", ev(GAIN))], inc=ev(BASE, **kw)))
    assert "resource_metrics_missing" in missing_inc.reasons


def test_weighted_rule_complexity_penalty():
    t = trace(run([arm("a", ev(BASE, tokens=90, complexity=108.0))]))
    terms = t.rule["terms"]
    assert terms["w_x*max(dX,0)"] == pytest.approx(0.04) and terms["w_x*simplicity_credit"] == 0.0
    expected = terms["w_s*dS"] - terms["w_c*dC"] + terms["w_n*nu"] - terms["w_x*max(dX,0)"]
    assert t.rule["value"] == pytest.approx(expected)
    heavy = trace(run([arm("a", ev(BASE, tokens=90, complexity=108.0))], w_x=50.0))
    assert not heavy.admissible and has(heavy, "weighted_rule")


def test_simplicity_credit_ships_pure_simplification():
    arms = [arm("a", ev(BASE, complexity=60.0))]
    assert run(arms, prof="local").decision == "do_not_ship"
    dec = run(arms)
    t = trace(dec)
    assert dec.decision == "ship" and t.resources["credit_eligible"]
    assert t.rule["terms"]["w_x*simplicity_credit"] == pytest.approx(0.2) and t.simplicity_score == 1.0


@pytest.mark.parametrize("kw", [{"scores": [0.5] * 7 + [0.25]}, {"crit": 1}, {"tree_valid": 0.0}])
def test_no_simplicity_credit_without_quality_safety_and_tree_validity(kw):
    scores = kw.pop("scores", BASE)
    t = trace(run([arm("a", ev(scores, complexity=60.0, **kw))]))
    assert not t.admissible and not t.resources["credit_eligible"] and t.simplicity_score is None
    assert t.rule["terms"]["w_x*simplicity_credit"] == 0.0


def test_tree_invalid_reason():
    t = trace(run([arm("a", ev(GAIN, tree_valid=0.5))]))
    assert not t.admissible and "tree_invalid" in t.reasons and t.resources["tree_valid"] == 0.5


def test_tie_break_prefers_lower_complexity_only_when_resource_aware():
    arms = [arm("a", ev(GAIN, complexity=105.0)), arm("b", ev(GAIN, complexity=95.0))]
    assert run(arms).winner == "b"
    assert run(arms, prof="local").winner == "a"


def test_harness_round_readjudicates_exactly():
    inp = inputs([arm("a", ev(BASE, complexity=60.0)), arm("b", ev(GAIN, calls=8))])
    rec = json.loads(json.dumps(round_record(inp, select(inp)), allow_nan=False))
    assert verify(rec)


def test_resource_fields_never_enter_score():
    plain = trace(run([arm("a", ev(GAIN))], require_resource_metrics=False))
    heavy = trace(run([arm("a", replace(ev(GAIN, calls=40, wall=9000.0), surface={"complexity": 500.0}))],
                      require_resource_metrics=False))
    assert plain.score == heavy.score and plain.delta_s == heavy.delta_s and plain.cost == heavy.cost
