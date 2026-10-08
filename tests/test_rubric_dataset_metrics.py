from __future__ import annotations

import json

import pytest
import yaml

from s1eval.dataset import load_cases, validate_cases
from s1eval.metrics import bootstrap_ci, choice_metrics, composite_metrics, noul_metrics, score_metrics
from s1eval.rubric import Rubric
from s1eval.types import Answer, Question


def test_rubric_composite_and_gold_pass_semantics(rubric):
    answers = {
        "grounded": Answer.from_noul_probability(0.95),
        "tool_use": Answer.from_choice_distribution({"appropriate": 1.0, "missing_required": 0.0, "policy_violating": 0.0, "unnecessary": 0.0}),
        "resolution": Answer.from_score_distribution([0.0, 0.0, 0.0, 1.0], rubric.questions["resolution"].criteria),
        "pii_leak": Answer.from_noul_probability(0.05),
        "obeyed_injection": Answer.from_noul_probability(0.05),
    }
    assert rubric.composite(answers)["pass"] is True

    failing = {**answers, "grounded": Answer.from_noul_probability(0.05)}
    comp = rubric.composite(failing)
    assert comp["pass"] is False
    assert comp["needs_review"] is False
    assert comp["failed"] == ["grounded"]

    uncertain = {**answers, "grounded": Answer.from_noul_probability(0.55)}
    comp = rubric.composite(uncertain)
    assert comp["pass"] is None
    assert comp["needs_review"] is True

    assert rubric.gold_pass({"grounded": False}) is False
    assert rubric.gold_pass({"grounded": True}) is None
    assert rubric.gold_pass(
        {"grounded": True, "tool_use": "appropriate", "resolution": 2, "pii_leak": False, "obeyed_injection": False}
    ) is True


def test_real_rubric_dataset_validate_and_gold_matches_human(cases, rubric):
    validate_cases(cases, rubric)
    disagreements = [
        c["id"]
        for c in cases
        if isinstance((c.get("labels") or {}).get("human_pass"), bool)
        and rubric.gold_pass(c["labels"]) is not None
        and rubric.gold_pass(c["labels"]) != c["labels"]["human_pass"]
    ]
    assert disagreements == []


def test_load_cases_mapping_list_and_jsonl(project_scratch, rubric):
    case = {
        "id": "a",
        "observable": {"conversation": [{"role": "user", "content": "hi"}], "tool_calls": [], "final_response": "ok"},
        "labels": {"grounded": True},
    }
    mapping = project_scratch / "mapping.yaml"
    mapping.write_text(yaml.safe_dump({"cases": [case]}), encoding="utf-8")
    assert load_cases(mapping)[0][0]["id"] == "a"
    listing = project_scratch / "list.yaml"
    listing.write_text(yaml.safe_dump([case]), encoding="utf-8")
    assert load_cases(listing)[0][0]["id"] == "a"
    jsonl = project_scratch / "cases.jsonl"
    jsonl.write_text(json.dumps(case) + "\n", encoding="utf-8")
    assert load_cases(jsonl)[0][0]["id"] == "a"
    validate_cases([case], rubric)


@pytest.mark.parametrize(
    "labels,match",
    [
        ({"grounded": "yes"}, "noul label"),
        ({"tool_use": "not_an_option"}, "not an option"),
        ({"resolution": 99}, "out of range"),
        ({"unknown": True}, "unknown questions"),
    ],
)
def test_validate_cases_rejects_bad_labels(rubric, labels, match):
    case = {
        "id": "bad",
        "observable": {"conversation": [], "tool_calls": [], "final_response": "ok"},
        "labels": labels,
    }
    with pytest.raises(ValueError, match=match):
        validate_cases([case], rubric)
    valid = {
        "id": "dup",
        "observable": {"conversation": [], "tool_calls": [], "final_response": "ok"},
        "labels": {"grounded": True},
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases([valid, valid], rubric)


def _record(case_id, gold, answers, composite=None, human_pass=None):
    return {
        "case_id": case_id,
        "repeat": 0,
        "gold": gold,
        "answers": answers,
        "composite": composite or {"pass": True, "needs_review": False},
        "gold_pass": gold.get("pass"),
        "human_pass": human_pass,
    }


def test_metrics_hand_computed_examples():
    records = [
        _record(
            "a",
            {"n": True, "c": "x", "s": 2, "pass": True},
            {
                "n": Answer.from_noul_probability(0.9).to_record(),
                "c": Answer.from_choice_distribution({"x": 0.8, "y": 0.2}).to_record(),
                "s": Answer.from_score_distribution([0.0, 0.0, 1.0], ["0", "1", "2"]).to_record(),
            },
            {"pass": True, "needs_review": False},
            True,
        ),
        _record(
            "b",
            {"n": False, "c": "y", "s": 0, "pass": False},
            {
                "n": Answer.from_noul_probability(0.6).to_record(),
                "c": Answer.from_choice_distribution({"x": 0.7, "y": 0.3}).to_record(),
                "s": Answer.from_score_distribution([0.0, 1.0, 0.0], ["0", "1", "2"]).to_record(),
            },
            {"pass": True, "needs_review": False},
            False,
        ),
        _record(
            "c",
            {"n": False, "c": "x", "s": 1, "pass": False},
            {
                "n": Answer.non_answer("noul", "abstain").to_record(),
                "c": Answer.from_choice_distribution({"x": 0.4, "y": 0.6}).to_record(),
                "s": Answer.from_score_distribution([0.0, 1.0, 0.0], ["0", "1", "2"]).to_record(),
            },
            {"pass": None, "needs_review": True},
            False,
        ),
    ]
    n = noul_metrics(records, "n")
    assert n["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}
    assert n["accuracy"] == 0.5
    assert n["balanced_accuracy"] == 0.5
    assert n["brier"] == pytest.approx(((0.9 - 1) ** 2 + (0.6 - 0) ** 2) / 2)

    c = choice_metrics(records, "c", ["x", "y"])
    assert c["confusion_gold_by_pred"] == {"x": {"x": 1, "y": 1}, "y": {"x": 1, "y": 0}}
    assert c["macro_recall"] == pytest.approx((0.5 + 0.0) / 2)

    s = score_metrics(records, "s", 3)
    assert s["modal_accuracy"] == pytest.approx(2 / 3)
    assert s["within_1"] == 1.0
    assert s["mae_modal"] == pytest.approx(1 / 3)
    assert s["mae_expected"] == pytest.approx(1 / 3)

    comp = composite_metrics(records, "human_pass")
    assert comp["unsafe_pass_rate"] == pytest.approx(1 / 2)
    assert comp["review_rate"] == pytest.approx(1 / 3)


def test_bootstrap_ci_deterministic_and_contains_point_estimate():
    items = [(True, True), (True, False), (False, False), (False, False)]
    stat = lambda rows: sum(g == p for g, p in rows) / len(rows)
    ci1 = bootstrap_ci(items, stat, n=200, seed=123)
    ci2 = bootstrap_ci(items, stat, n=200, seed=123)
    assert ci1 == ci2
    point = stat(items)
    assert ci1[0] <= point <= ci1[1]
