import json
from dataclasses import replace

import pytest

from ci_lab.contracts import EvaluatorPin
from ci_lab.oes import Envelope, validate_envelope
from ci_lab.oes.build import ConfirmStats, Holdout, summarize
from ci_lab.oes.canonical import HASH_FIELD, verify
from ci_lab.oes.models import NON_COMPENSATORY, RRSI_EXT, SLEEP_EXT

KINDS = ["calibration", "round", "confirm", "sleep"]


def _results(doc, metric_id, variant=None):
    return [r for r in doc["results"]["metricResults"]
            if r["metricId"] == metric_id and (variant is None or r["comparison"]["variantId"] == variant)]


@pytest.mark.parametrize("kind", KINDS)
def test_builders_emit_valid_hash_locked_envelopes(envelopes, kind):
    doc = envelopes[kind]
    assert validate_envelope(doc) == []
    assert doc["schemaVersion"] == "0.1.0" and verify(doc)
    assert doc["provenance"]["resultHash"].startswith("sha256:")
    assert [v["id"] for v in doc["variants"] if v["role"] == "baseline"].__len__() == 1


@pytest.mark.parametrize("kind", KINDS)
def test_models_round_trip_exactly(envelopes, kind):
    doc = envelopes[kind]
    again = Envelope.from_dict(doc).to_dict()
    assert again == doc
    assert json.dumps(again, sort_keys=True) == json.dumps(doc, sort_keys=True)
    assert again[HASH_FIELD] == doc[HASH_FIELD]


@pytest.mark.parametrize("kind", KINDS)
def test_builders_are_deterministic(fx, kind):
    assert fx.BUILDERS[kind]() == fx.BUILDERS[kind]()


@pytest.mark.parametrize("kind", KINDS)
def test_metric_roles(envelopes, kind):
    roles = {m["id"]: m["role"] for m in envelopes[kind]["metrics"]}
    assert roles["safety_violations"] == "guardrail" and roles["cost_tokens_per_task"] == "guardrail"
    assert roles["missing_trial_rate"] == roles["judge_error_rate"] == "data_quality"
    assert roles["evaluator_pin"] == "invariant"
    assert any(r == "diagnostic" for r in roles.values())
    assert [m for m, r in roles.items() if r == "primary"] == [
        {"calibration": "evolve_score", "round": "evolve_score", "confirm": "heldout_score",
         "sleep": "evolve_score"}[kind]]
    safety = next(m for m in envelopes[kind]["metrics"] if m["id"] == "safety_violations")
    assert safety[NON_COMPENSATORY] is True


def test_calibration_envelope(envelopes):
    doc = envelopes["calibration"]
    assert doc["experiment"]["id"] == "tone-a1-cal"
    assert doc["design"]["type"] == "abn" and [v["id"] for v in doc["variants"]] == ["rep0", "rep1", "rep2"]
    assert doc["decision"]["outcome"] == "do_not_ship"
    ext = doc["extensions"][RRSI_EXT]
    assert ext["kind"] == "calibration" and ext["round"] == 0 and ext["repeats"] == 3 and ext["delta"] == 0.02
    noise = next(c for c in doc["qualityChecks"] if c["checkType"] == "aa_noise_band")
    assert noise["status"] == "pass" and noise["observed"]["maxAbsDeltaS"] == pytest.approx(0.01)


def test_calibration_two_repeats_is_ab_and_tree_mismatch_reruns(fx):
    assert fx.build_calibration(runs=[fx.make_eval(fx.T0, "aa"), fx.make_eval(fx.T0, "aa")])["design"]["type"] == "ab"
    doc = fx.build_calibration(runs=[fx.make_eval(fx.T0, "aa"), fx.make_eval(fx.T1, "aa")])
    assert doc["decision"]["outcome"] == "rerun" and validate_envelope(doc) == []


def test_round_envelope_ship(envelopes, fx):
    doc = envelopes["round"]
    assert doc["experiment"]["id"] == "tone-a1-r01"
    assert doc["design"]["type"] == "abn" and doc["design"]["multipleTestingPolicy"] == "custom"
    assert [(v["id"], v["role"]) for v in doc["variants"]] == [
        ("inc", "baseline"), ("v1", "treatment"), ("v2", "treatment"), ("v3", "treatment")]
    assert doc["decision"]["outcome"] == "ship" and doc["scorecard"]["primaryOutcome"]["variantId"] == "v1"
    ext = doc["extensions"][RRSI_EXT]
    assert ext["multipleTesting"] == "exploratory" and ext["selection"]["winner"] == "v1"
    assert ext["ciLowerBound"] == 0.04 and ext["lineage"]["parentExperimentId"] == "tone-a1-cal"
    assert ext["variants"]["v1"]["edits"][0] == {
        "component": "prompt", "hypothesis": "v1: clarify change policy", "commit": fx.H1,
        "files": ["harness/prompts/system.md"]}
    assert ext["variants"]["v3"]["critic"]["passed"] is False and ext["variants"]["v3"]["status"] == "rejected"
    assert ext["evaluatorPin"]["servedJudgeModels"] == ["gpt-judge-2026-09"]
    # v2 has a critical violation the incumbent lacks: its safety guardrail blocks ship
    assert _results(doc, "safety_violations", "v2")[0]["decisionImpact"] == "blocks_ship"
    assert _results(doc, "evolve_score", "v1")[0]["decisionImpact"] == "supports_ship"
    assert _results(doc, "evolve_score", "v1")[0]["confidenceInterval"]["lower"] == 0.04
    assert not _results(doc, "evolve_score", "v3")  # unevaluated arm: no results
    assert doc["results"]["sampleSizes"] == {"inc": 9, "v1": 9, "v2": 9}


def test_round_without_winner_is_do_not_ship(fx):
    doc = fx.build_round(winner=None)
    assert doc["decision"]["outcome"] == "do_not_ship" and doc["extensions"][RRSI_EXT]["ciLowerBound"] is None
    assert validate_envelope(doc) == []


def test_round_lineage_for_later_rounds(fx):
    doc = fx.build_round(round_no=3, supersedes="tone-a1-r03-old")
    ext = doc["extensions"][RRSI_EXT]
    assert doc["experiment"]["id"] == "tone-a1-r03" and ext["lineage"]["parentExperimentId"] == "tone-a1-r02"
    assert ext["supersedes"] == "tone-a1-r03-old" and validate_envelope(doc) == []


def test_round_pin_drift_reruns(fx):
    other = EvaluatorPin(fx.PIN.evaluator_tree, "gpt-judge", "GitHubCopilot", ("gpt-judge-2026-10",))
    arms = [fx.make_arm("v1", fx.H1, fx.T1, fx.make_eval(fx.T1, base=0.7, pin=other))]
    doc = fx.build_round(arms=arms)
    assert doc["decision"]["outcome"] == "rerun"
    assert next(c for c in doc["qualityChecks"] if c["checkType"] == "invariant_metric")["status"] == "fail"
    assert _results(doc, "evaluator_pin", "v1")[0]["variantValue"] == 0
    assert doc["extensions"][RRSI_EXT]["ciLowerBound"] is None and validate_envelope(doc) == []


def test_round_baseline_missing_trials_reruns(fx):
    doc = fx.build_round(incumbent=fx.make_eval(fx.T0, missing=2))
    assert doc["decision"]["outcome"] == "rerun" and doc["scorecard"]["qualityStatus"] == "invalid"
    assert validate_envelope(doc) == []


def test_round_surface_metrics_only_with_surface_data(fx, envelopes):
    assert not {"surface_complexity", "simplicity_score"} & {m["id"] for m in envelopes["round"]["metrics"]}

    def with_surface(ev, complexity):
        return replace(ev, surface={"complexity": complexity, "files": 4.0})

    arms = [fx.make_arm("v1", fx.H1, fx.T1, with_surface(fx.make_eval(fx.T1, base=0.7), 80.0)),
            fx.make_arm("v2", fx.H2, fx.T2, fx.make_eval(fx.T2, base=0.55, crit=1)),
            fx.make_arm("v3", None, None, None, passed=False)]
    doc = fx.build_round(arms=arms, incumbent=with_surface(fx.make_eval(fx.T0), 100.0))
    assert validate_envelope(doc) == [] and verify(doc)
    roles = {m["id"]: (m["role"], m["direction"]) for m in doc["metrics"]}
    assert roles["surface_complexity"] == ("diagnostic", "decrease_is_good")
    assert roles["simplicity_score"] == ("diagnostic", "increase_is_good")
    cx = _results(doc, "surface_complexity", "v1")[0]
    assert (cx["baselineValue"], cx["variantValue"], cx["relativeDifference"]) == (100.0, 80.0, -0.2)
    simp = _results(doc, "simplicity_score", "v1")[0]
    assert (simp["baselineValue"], simp["variantValue"]) == (0.5, 1.0)
    assert not _results(doc, "surface_complexity", "v2")  # no surface data on v2
    assert Envelope.from_dict(doc).to_dict() == doc


def test_confirm_envelope(envelopes):
    doc = envelopes["confirm"]
    assert doc["experiment"]["id"] == "tone-a1-confirm" and doc["design"]["type"] == "ab"
    assert doc["design"]["alpha"] == 0.05 and doc["design"]["peekingPolicy"] == "fixed_horizon"
    assert doc["decision"]["outcome"] == "ship"
    ext = doc["extensions"][RRSI_EXT]
    assert ext["multipleTesting"] == "confirmatory" and ext["split"] == "heldout"
    assert ext["preRegistration"]["sided"] == "one" and ext["preRegistration"]["nonInferiorityMargin"] == 0
    assert ext["holdout"] == {"datasetHash": "sha256:" + "2" * 64, "plannedLooks": 1, "looksUsed": 1}
    assert _results(doc, "heldout_score", "final")[0]["pValue"] == 0.01
    assert _results(doc, "ood_score", "final")[0]["role"] == "secondary"


@pytest.mark.parametrize(("kw", "reason"), [({"p_value": 0.2}, "not significant"),
                                            ({"final_crit": 1}, "safety inferior")])
def test_confirm_do_not_ship(fx, kw, reason):
    doc = fx.build_confirm(**kw)
    assert doc["decision"]["outcome"] == "do_not_ship" and reason in doc["decision"]["rationale"]
    assert validate_envelope(doc) == []


def test_confirm_margin_allows_bounded_safety_regression(fx):
    assert fx.build_confirm(final_crit=1, non_inferiority_margin=1)["decision"]["outcome"] == "ship"


def test_sleep_envelope(envelopes):
    doc = envelopes["sleep"]
    assert doc["experiment"]["id"] == "sleep-20261008" and doc["decision"]["outcome"] == "ship"
    ext = doc["extensions"][SLEEP_EXT]
    assert ext["tasks"]["total"] == 32 and ext["gate"]["assert"]["deltaS"] == pytest.approx(0.05)
    assert ext["gate"]["assert"]["safetyViolations"] == {"baseline": 0, "candidate": 0}
    assert ext["budget"]["used"]["wallClockSeconds"] == 1800.5 and ext["candidateDigest"].startswith("sha256:")


@pytest.mark.parametrize("kw", [{"candidate": False}, {"assert_passed": False}])
def test_sleep_do_not_ship(fx, kw):
    doc = fx.build_sleep(**kw)
    assert doc["decision"]["outcome"] == "do_not_ship" and validate_envelope(doc) == []


def test_sleep_with_adoption_pr(fx):
    pr = {"branch": "exp/sleep-20261008-7/cand", "number": 42, "url": "https://github.com/o/r/pull/42"}
    doc = fx.build_sleep(adoption_pr=pr)
    assert doc["extensions"][SLEEP_EXT]["adoptionPr"] == pr and validate_envelope(doc) == []


def _ungated_sleep(fx, **kw):
    from ci_lab.oes.build import sleep_envelope

    base = {"incumbent": None, "candidate": None, "incumbent_commit": fx.H0, "skillopt_version": "0.1",
            "tasks_by_origin": {"reviewed": 4}, "tasks_by_split": {"evolve": 4},
            "skillopt_gate": {"passed": False, "reasons": ["no candidate"]},
            "assert_gate": {"passed": False, "reasons": ["no candidate"]}, "delta": 0.02,
            "budget_used": {"tasks": 4}, "evaluator_pin": fx.PIN, "incumbent_digest": "sha256:" + "e" * 64,
            "exported_at": fx.AT}
    return sleep_envelope("2026-10-08", **{**base, **kw})


def test_sleep_without_incumbent_eval_is_valid_no_change(fx):
    doc = _ungated_sleep(fx)
    assert validate_envelope(doc) == []
    assert doc["decision"]["outcome"] == "do_not_ship" and "control retained" in doc["decision"]["rationale"]
    assert [v["id"] for v in doc["variants"]] == ["incumbent"] and "results" not in doc
    assert "no-change" in doc["experiment"]["tags"]
    ext = doc["extensions"][SLEEP_EXT]
    assert ext["evaluatorPin"]["evaluatorTree"] == fx.PIN.evaluator_tree and ext["candidateDigest"] is None
    assert "safetyViolations" not in ext["gate"]["assert"]


def test_sleep_without_incumbent_eval_budget_stop_is_rerun(fx):
    doc = _ungated_sleep(fx, rerun_reason="rollouts budget exceeded")
    assert validate_envelope(doc) == []
    assert doc["decision"]["outcome"] == "rerun" and doc["scorecard"]["qualityStatus"] == "invalid"


@pytest.mark.parametrize(("kw", "match"), [({"evaluator_pin": None}, "evaluator_pin"),
                                           ({"candidate": "eval"}, "incumbent eval")])
def test_sleep_without_incumbent_eval_rejects_bad_inputs(fx, kw, match):
    if kw.get("candidate") == "eval":
        kw["candidate"] = fx.make_eval(fx.T1)
    with pytest.raises(ValueError, match=match):
        _ungated_sleep(fx, **kw)

def test_summarize_counts_missing_as_zero(fx):
    s = summarize(fx.make_eval(fx.T0, base=0.6, missing=3, crit=2))
    assert s.trials == 9 and s.missing_rate == pytest.approx(1 / 3, abs=1e-6)
    assert s.score == pytest.approx(0.4) and s.critical_violations == 2 and s.tokens_per_task == 150


@pytest.mark.parametrize(("kind", "kw", "match"), [
    ("calibration", {"runs": "one"}, "at least 2"),
    ("round", {"round_no": 0}, "rounds start at 1"),
    ("round", {"incumbent": "heldout"}, "evolve split"),
    ("round", {"arms": "dupe"}, "arm names"),
    ("round", {"winner": "v3"}, "not an evaluated arm"),
    ("confirm", {"holdout": Holdout("sha256:" + "2" * 64, looks_used=2)}, "look budget"),
    ("confirm", {"holdout": Holdout("sha256:" + "2" * 64, looks_used=0)}, "look budget"),
])
def test_builders_reject_invalid_inputs(fx, kind, kw, match):
    if kw.get("runs") == "one":
        kw["runs"] = [fx.make_eval(fx.T0, "aa")]
    if kw.get("incumbent") == "heldout":
        kw["incumbent"] = fx.make_eval(fx.T0, "heldout")
    if kw.get("arms") == "dupe":
        kw["arms"] = [fx.make_arm("v1", fx.H1, fx.T1, fx.make_eval(fx.T1))] * 2
    with pytest.raises(ValueError, match=match):
        fx.BUILDERS[kind](**kw)


def test_builder_rejects_bad_ids(fx):
    from ci_lab.oes.build import calibration_envelope, confirm_envelope, sleep_envelope
    with pytest.raises(ValueError, match="campaign"):
        calibration_envelope("BAD ID", [fx.make_eval(fx.T0)] * 2, delta=0.02, delta_method="x",
                             harness_commit=fx.H0, split_hashes=fx.SPLITS)
    with pytest.raises(ValueError, match="held-out"):
        confirm_envelope("tone-a1", baseline=fx.make_eval(fx.T0), final=fx.make_eval(fx.T1), baseline_commit=fx.H0,
                         final_commit=fx.H1, stats=ConfirmStats(0.01, 0.03),
                         holdout=Holdout("sha256:" + "2" * 64, 1), look_ledger_ref="x", split_hashes=fx.SPLITS)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        sleep_envelope("20261008", incumbent=fx.make_eval(fx.T0), candidate=None, incumbent_commit=fx.H0,
                       skillopt_version="0", tasks_by_origin={}, tasks_by_split={}, skillopt_gate={},
                       assert_gate={}, delta=0.02, budget_used={})
