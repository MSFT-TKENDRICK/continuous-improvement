from __future__ import annotations

import json

import pytest

from s1eval.confidence import choice_confidence, noul_confidence, score_confidence
from s1eval.runner import run_cases
from s1eval.state import LeakageError, project_observable
from s1eval.types import Answer, Question, WireError, questions_from_wire, questions_to_wire


def test_confidence_formulas_edges_and_conventions():
    assert choice_confidence({"a": 0.5, "b": 0.5}) == 0
    assert choice_confidence({"a": 1.0, "b": 0.0}) == 1
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0)
    assert choice_confidence([0.7, 0.2, 0.1]) == pytest.approx((3 * 0.7 - 1) / 2)

    assert score_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0)
    assert score_confidence([0, 1, 0]) == pytest.approx(1)
    assert score_confidence([1, 0, 0]) == pytest.approx(1)

    assert noul_confidence(0.5) == 0
    assert noul_confidence(0) == 1
    assert noul_confidence(1) == 1
    assert noul_confidence(0.8) == pytest.approx(0.6)


def test_question_validation_and_wire_round_trip():
    qs = {
        "noul_ok": Question("noul", {"task": "judge"}, {"true": "yes", "false": "no"}),
        "choice_ok": Question("choice", "pick", {"a": None, "b": ["bee"]}),
        "score_ok": Question("score", "rate", ["low", {"mid": True}, ["high"]]),
    }
    assert questions_from_wire(questions_to_wire(qs)) == qs

    with pytest.raises(WireError, match="invalid question name"):
        questions_from_wire({"bad name!": {"type": "noul"}})
    with pytest.raises(WireError, match="choice needs"):
        Question("choice", "pick", {"only": "one"})
    with pytest.raises(WireError, match="choice needs"):
        Question("choice", "pick", {str(i): None for i in range(256)})
    with pytest.raises(WireError, match="score needs"):
        Question("score", "rate", ["only"])
    with pytest.raises(WireError, match="score needs"):
        Question("score", "rate", [str(i) for i in range(11)])
    with pytest.raises(WireError, match="noul criteria"):
        Question("noul", "n", {"maybe": "bad"})


def test_answer_from_wire_strictness_and_diagnostics_ignored():
    choice_q = Question("choice", "pick", {"a": None, "b": None})
    score_q = Question("score", "rate", ["zero", "one", "two"])
    noul_q = Question("noul", "true?")

    with pytest.raises(WireError, match="sum"):
        Answer.from_wire({"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.2}}, choice_q)
    with pytest.raises(WireError, match="probability keys"):
        Answer.from_wire({"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "c": 0.5}}, choice_q)
    with pytest.raises(WireError, match="outside level range"):
        Answer.from_wire({"type": "score", "score": 3, "probabilities": {"0": 0, "1": 0, "2": 1}}, score_q)
    with pytest.raises(WireError, match="probability must"):
        Answer.from_wire({"type": "noul", "noul": True}, noul_q)

    ans = Answer.from_wire(
        {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "s1eval": {"debug": "ignored"}},
        choice_q,
    )
    assert ans.choice == "a"
    with pytest.raises(WireError, match="cannot serialise"):
        Answer.non_answer("noul", "abstain").to_wire()


def test_project_observable_rejects_leakage_and_bad_shapes():
    base = {"observable": {"conversation": [{"role": "user", "content": "hi"}], "tool_calls": [], "final_response": "hello"}}
    projected = project_observable(base)
    assert projected == base["observable"]
    projected["conversation"][0]["content"] = "mutated"
    assert base["observable"]["conversation"][0]["content"] == "hi"

    bad = {"observable": {**base["observable"], "labels": {"grounded": False}}}
    with pytest.raises(LeakageError, match="non-observable"):
        project_observable(bad)
    with pytest.raises(LeakageError, match="role"):
        project_observable({"observable": {"conversation": [{"role": "system", "content": "x"}], "final_response": "x"}})
    with pytest.raises(LeakageError, match="final_response"):
        project_observable({"observable": {"conversation": []}})


def test_runner_with_scripted_backend_never_leaks_case_metadata(rubric, scripted_backend, json_dumps_compact):
    sentinel = "LEAK_SENTINEL_9f3"
    case = {
        "id": f"id-{sentinel}",
        "tags": [sentinel],
        "notes": f"notes {sentinel}",
        "labels": {"human_pass": False, "grounded": sentinel},
        "observable": {
            "agent_policy": "policy",
            "conversation": [{"role": "user", "content": "where is my order?"}],
            "tool_calls": [],
            "final_response": "answer",
        },
    }
    recs = run_cases(scripted_backend, rubric, [case])
    assert recs[0]["case_id"] == f"id-{sentinel}"
    state, _questions = scripted_backend.requests[0]
    assert sentinel not in json_dumps_compact(state)


def test_real_dataset_projects_cleanly_without_notes_or_labels_leaking(cases, json_dumps_compact):
    for case in cases:
        state = project_observable(case)
        text = json_dumps_compact(state)
        assert "notes" not in text
        assert "labels" not in text
        for value in (case.get("labels") or {}).values():
            if isinstance(value, str):
                assert value not in text
        if isinstance(case.get("notes"), str):
            assert case["notes"] not in text
