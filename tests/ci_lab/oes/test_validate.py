"""Mutation tests: each mutation is re-sealed so only the targeted rule can fire."""

import copy
import json

import pytest

from ci_lab.oes import validate_envelope
from ci_lab.oes.models import RRSI_EXT, SLEEP_EXT
from ci_lab.oes.validate import iter_rule_ids, read_look_ledger, validate_file


def rules(doc, **kw):
    return iter_rule_ids(validate_envelope(doc, **kw))


def mutate(fx, doc, fn):
    doc = copy.deepcopy(doc)
    fn(doc)
    return fx.reseal(doc)


def _result(doc, metric_id, variant):
    return next(r for r in doc["results"]["metricResults"]
                if r["metricId"] == metric_id and r["comparison"]["variantId"] == variant)


# ---------------------------------------------------------------- envelope-level

def test_non_object_and_unreadable(tmp_path):
    assert rules([1]) == {"schema"}
    bad = tmp_path / "x.json"
    bad.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    assert iter_rule_ids(validate_file(bad)) == {"json"}
    bad.write_text('{"a": NaN}', encoding="utf-8")
    assert iter_rule_ids(validate_file(bad)) == {"json"}


def test_core_schema_violation(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["design"].update(type="split_test"))
    errs = validate_envelope(doc)
    assert iter_rule_ids(errs) == {"schema"} and any("$.design.type" in e for e in errs)


def test_unknown_extensions_are_ignored(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["extensions"].update({"org.example": {"anything": [1]}}))
    assert validate_envelope(doc) == []


@pytest.mark.parametrize(("kind", "fn", "path"), [
    ("round", lambda d: d["extensions"][RRSI_EXT].update(pruneSet=["vibes"]), "pruneSet"),
    ("round", lambda d: d["extensions"][RRSI_EXT].update(extra=1), "extra"),
    ("round", lambda d: d["extensions"][RRSI_EXT].pop("costRule"), "costRule"),
    ("round", lambda d: d["extensions"][RRSI_EXT].update(multipleTesting="confirmatory"), "multipleTesting"),
    ("calibration", lambda d: d["extensions"][RRSI_EXT].update(round=1), "round"),
    ("confirm", lambda d: d["extensions"][RRSI_EXT].pop("lookLedgerRef"), "lookLedgerRef"),
    ("confirm", lambda d: d["extensions"][RRSI_EXT]["evaluatorPin"].pop("servedJudgeModels"), "servedJudgeModels"),
    ("sleep", lambda d: d["extensions"][SLEEP_EXT]["gate"].pop("assert"), "assert"),
    ("sleep", lambda d: d["extensions"][SLEEP_EXT].update(adoptionPr={"branch": "main", "number": 1,
                                                                      "url": "https://x/1"}), "branch"),
])
def test_extension_schema_violations(fx, envelopes, kind, fn, path):
    errs = validate_envelope(mutate(fx, envelopes[kind], fn))
    assert iter_rule_ids(errs) == {"extension-schema"}
    assert any(path in e for e in errs), errs


def test_schema_version(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d.update(schemaVersion="0.2.0"))
    assert "schema-version" in rules(doc)


# ---------------------------------------------------------------- hash lock

@pytest.mark.parametrize("kind", ["calibration", "round", "confirm", "sleep"])
def test_content_hash_detects_tampering(envelopes, kind):
    doc = copy.deepcopy(envelopes[kind])
    doc["decision"]["rationale"] += " (edited)"
    assert rules(doc) == {"content-hash"}


def test_content_hash_required_when_decided(envelopes):
    doc = copy.deepcopy(envelopes["round"])
    del doc["contentHash"]
    assert rules(doc) == {"content-hash"}


def test_content_hash_optional_when_undecided(fx, envelopes):
    def undecide(d):
        d["experiment"]["status"] = "running"
        d["decision"] = {"status": "pending"}
        d.pop("scorecard")
    doc = mutate(fx, envelopes["calibration"], undecide)
    del doc["contentHash"]
    assert validate_envelope(doc) == []


def test_result_hash(fx, envelopes):
    def tamper(d):
        d["results"]["sampleSizes"]["inc"] = 99
    doc = copy.deepcopy(envelopes["round"])
    tamper(doc)
    from ci_lab.oes.canonical import seal
    assert rules(seal(doc)) == {"result-hash"}


# ---------------------------------------------------------------- baseline / references

def test_two_baselines(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["variants"][1].update(role="baseline"))
    assert "baseline" in rules(doc)


def test_zero_baselines(fx, envelopes):
    doc = mutate(fx, envelopes["confirm"], lambda d: d["variants"][0].update(role="treatment"))
    assert "baseline" in rules(doc)


def test_duplicate_variant_ids(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["variants"][2].update(id="v1"))
    assert "baseline" in rules(doc)


def test_control_role_counts_as_baseline(fx, envelopes):
    doc = mutate(fx, envelopes["sleep"], lambda d: d["variants"][0].update(role="control"))
    assert validate_envelope(doc) == []


@pytest.mark.parametrize("fn", [
    lambda d: d["results"]["metricResults"][0].update(metricId="nope"),
    lambda d: d["results"]["metricResults"][0]["comparison"].update(variantId="ghost"),
    lambda d: d["results"]["metricResults"][0]["comparison"].update(baselineVariantId="v2"),
    lambda d: d["results"]["sampleSizes"].update(ghost=3),
])
def test_references(fx, envelopes, fn):
    assert rules(mutate(fx, envelopes["round"], fn)) == {"references"}


# ---------------------------------------------------------------- decision consistency

def test_experiment_decided_but_decision_pending(fx, envelopes):
    doc = mutate(fx, envelopes["calibration"], lambda d: d["decision"].update(status="pending"))
    assert "decision" in rules(doc)


def test_decided_without_outcome(fx, envelopes):
    doc = mutate(fx, envelopes["calibration"], lambda d: d["decision"].pop("outcome"))
    assert rules(doc) == {"decision"}


def test_outcome_outside_ci_lab_vocabulary(fx, envelopes):
    def iterate(d):
        d["decision"]["outcome"] = "iterate"
        d["scorecard"]["recommendedAction"] = "iterate"
    assert rules(mutate(fx, envelopes["calibration"], iterate)) == {"decision"}


def test_recommended_action_mismatch(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["scorecard"].update(recommendedAction="do_not_ship"))
    assert rules(doc) == {"decision"}


def test_ship_with_failing_critical_check(fx, envelopes):
    doc = mutate(fx, envelopes["confirm"], lambda d: d["qualityChecks"][0].update(status="fail",
                                                                                  severity="critical"))
    assert rules(doc) == {"decision"}


def test_ship_without_treatment(fx, envelopes):
    def ship(d):
        d["decision"]["outcome"] = "ship"
        d["scorecard"]["recommendedAction"] = "ship"
    assert "decision" in rules(mutate(fx, envelopes["calibration"], ship))


# ---------------------------------------------------------------- non-compensatory safety

def test_non_compensatory_guardrail_worsened(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: _result(d, "safety_violations", "v1").update(variantValue=1))
    assert rules(doc) == {"non-compensatory"}


def test_non_compensatory_blocking_result(fx, envelopes):
    doc = mutate(fx, envelopes["confirm"],
                 lambda d: _result(d, "evaluator_pin", "final").update(decisionImpact="blocks_ship"))
    assert rules(doc) == {"non-compensatory"}


def test_non_compensatory_missing_guardrail_result(fx, envelopes):
    def drop(d):
        d["results"]["metricResults"] = [r for r in d["results"]["metricResults"]
                                         if not (r["metricId"] == "safety_violations"
                                                 and r["comparison"]["variantId"] == "candidate")]
    assert rules(mutate(fx, envelopes["sleep"], drop)) == {"non-compensatory"}


def test_non_compensatory_respects_confirm_margin(fx):
    doc = fx.build_confirm(final_crit=1, non_inferiority_margin=1)
    assert validate_envelope(doc) == []
    doc = mutate(fx, doc, lambda d: d["extensions"][RRSI_EXT]["preRegistration"].update(nonInferiorityMargin=0))
    assert rules(doc) == {"non-compensatory"}


def test_non_compensatory_ignored_when_not_shipping(fx):
    doc = fx.build_round(winner=None)
    assert _result(doc, "safety_violations", "v2")["decisionImpact"] == "blocks_ship"
    assert validate_envelope(doc) == []


# ---------------------------------------------------------------- rrsi rounds

def test_rrsi_ship_without_winner(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["extensions"][RRSI_EXT]["selection"].update(winner=None))
    assert "rrsi" in rules(doc)


def test_rrsi_do_not_ship_with_winner(fx, envelopes):
    def keep(d):
        d["decision"]["outcome"] = "do_not_ship"
        d["scorecard"]["recommendedAction"] = "do_not_ship"
    assert rules(mutate(fx, envelopes["round"], keep)) == {"rrsi"}


def test_rrsi_inadmissible_winner(fx, envelopes):
    doc = mutate(fx, envelopes["round"],
                 lambda d: d["extensions"][RRSI_EXT]["selection"]["candidates"][0].update(admissible=False))
    assert rules(doc) == {"rrsi"}


def test_rrsi_baseline_winner(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["extensions"][RRSI_EXT]["selection"].update(winner="inc"))
    assert "rrsi" in rules(doc)


def test_rrsi_ci_lower_bound_mismatch(fx, envelopes):
    doc = mutate(fx, envelopes["round"], lambda d: d["extensions"][RRSI_EXT].update(ciLowerBound=0.5))
    assert rules(doc) == {"rrsi"}


@pytest.mark.parametrize(("kind", "fn"), [
    ("round", lambda d: d["experiment"].update(id="tone-a1-r02")),
    ("calibration", lambda d: d["experiment"].update(id="tone-a1-r00")),
    ("round", lambda d: d["design"].update(type="ab")),
    ("round", lambda d: d["design"].update(multipleTestingPolicy="none")),
    ("round", lambda d: d["extensions"][RRSI_EXT]["variants"].pop("v3")),
])
def test_rrsi_structure(fx, envelopes, kind, fn):
    assert rules(mutate(fx, envelopes[kind], fn)) == {"rrsi"}


# ---------------------------------------------------------------- held-out looks + confirmation

def test_holdout_looks_exceed_planned(fx, envelopes):
    doc = mutate(fx, envelopes["confirm"], lambda d: d["extensions"][RRSI_EXT]["holdout"].update(looksUsed=2))
    assert rules(doc) == {"holdout-looks"}


def test_holdout_confirm_is_a_look(fx, envelopes):
    doc = mutate(fx, envelopes["confirm"], lambda d: d["extensions"][RRSI_EXT]["holdout"].update(looksUsed=0))
    assert rules(doc) == {"holdout-looks"}


def test_holdout_look_ledger(envelopes, tmp_path):
    doc = envelopes["confirm"]
    h = doc["extensions"][RRSI_EXT]["holdout"]["datasetHash"]
    assert validate_envelope(doc, look_counts={h: 1}) == []
    assert rules(doc, look_counts={h: 2}) == {"holdout-looks"}
    assert validate_envelope(doc, look_counts={"sha256:other": 5}) == []
    ledger = tmp_path / "looks.jsonl"
    ledger.write_text("\n".join(json.dumps({"datasetHash": x, "experimentId": "e"}) for x in (h, h, "o")) + "\n",
                      encoding="utf-8")
    assert read_look_ledger(ledger) == {h: 2, "o": 1}


def test_confirm_ship_requires_significance(fx, envelopes):
    doc = mutate(fx, envelopes["confirm"],
                 lambda d: d["extensions"][RRSI_EXT]["preRegistration"]["stats"].update(pValue=0.05))
    assert rules(doc) == {"confirm"}


def test_confirm_ship_requires_positive_ci(fx, envelopes):
    doc = mutate(fx, envelopes["confirm"],
                 lambda d: d["extensions"][RRSI_EXT]["preRegistration"]["stats"].update(ciLower=0))
    assert rules(doc) == {"confirm"}


@pytest.mark.parametrize("fn", [lambda d: d["design"].update(alpha=0.1),
                                lambda d: d["design"].update(peekingPolicy="sequential")])
def test_confirm_design_matches_preregistration(fx, envelopes, fn):
    assert rules(mutate(fx, envelopes["confirm"], fn)) == {"confirm"}


# ---------------------------------------------------------------- sleep

def test_sleep_ship_with_failed_gate(fx, envelopes):
    doc = mutate(fx, envelopes["sleep"], lambda d: d["extensions"][SLEEP_EXT]["gate"]["assert"].update(passed=False))
    assert rules(doc) == {"sleep"}


def test_sleep_ship_without_candidate_digest(fx, envelopes):
    doc = mutate(fx, envelopes["sleep"], lambda d: d["extensions"][SLEEP_EXT].update(candidateDigest=None))
    assert rules(doc) == {"sleep"}


def test_sleep_heldout_tasks_rejected(fx, envelopes):
    doc = mutate(fx, envelopes["sleep"], lambda d: d["extensions"][SLEEP_EXT]["tasks"]["bySplit"].update(heldout=3))
    assert rules(doc) == {"sleep"}


def test_sleep_task_total(fx, envelopes):
    doc = mutate(fx, envelopes["sleep"], lambda d: d["extensions"][SLEEP_EXT]["tasks"].update(total=31))
    assert rules(doc) == {"sleep"}


def test_sleep_experiment_id(fx, envelopes):
    doc = mutate(fx, envelopes["sleep"], lambda d: d["extensions"][SLEEP_EXT].update(night="2026-10-09"))
    assert rules(doc) == {"sleep"}


def test_sleep_adoption_pr_requires_ship(fx):
    pr = {"branch": "exp/sleep-20261008-7/cand", "number": 42, "url": "https://github.com/o/r/pull/42"}
    doc = mutate(fx, fx.build_sleep(assert_passed=False),
                 lambda d: d["extensions"][SLEEP_EXT].update(adoptionPr=pr))
    assert rules(doc) == {"sleep"}
