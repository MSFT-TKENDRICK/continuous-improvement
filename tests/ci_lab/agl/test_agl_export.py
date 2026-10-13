from __future__ import annotations

from pathlib import Path

import pytest

from ci_lab.agl.export import (
    METRIC,
    HoldoutViolation,
    eval_result,
    expected_keys,
    oes_metric_values,
    skillopt_task_records,
    task_scores,
)
from ci_lab.agl.journal import FileRolloutJournal
from ci_lab.agl.scope import RolloutScope
from ci_lab.contracts import EvaluatorPin, RolloutKey, TaskScore, Violation

EXP, VAR = "camp-r00", "base"


def _journal(tmp_path: Path) -> FileRolloutJournal:
    j = FileRolloutJournal(tmp_path, fsync=False)
    # case-a: 2 trials, both pass; case-b: trial 0 fails with a critical violation, trial 1 missing
    for trial, value in ((0, 1.0), (1, 0.8)):
        with RolloutScope(j, RolloutKey(EXP, VAR, "case-a", trial),
                          {"intent": "Change harness prompt 7", "suite": "harness_proposal", "split": "evolve",
                           "reference_kind": "rubric", "reference": "inspect before editing"}) as s:
            s.record_model_request({"model": "gpt-5-mini", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                                    "ci": {"served_model": "gpt-5-mini-2025"},
                                    "request": {"messages": [{"role": "user", "content": "RAW TOOL OUTPUT"}]}})
            s.score("assert", value, suite="harness_proposal", category="change",
                    rule_ids=["change.verify"], excerpt="asked for resource id")
            s.reward(value, source="assert")
    with RolloutScope(j, RolloutKey(EXP, VAR, "case-b", 0),
                      {"intent": "Ignore previous instructions", "suite": "harness_injection"}) as s:
        s.score("assert", 0.9, suite="harness_injection", excerpt="INJECTED PAYLOAD",
                violations=[{"rule_id": "inj.followed", "severity": "critical", "detail": "followed"},
                            {"rule_id": "inj.followed", "severity": "critical", "detail": "followed"},
                            {"severity": "major"}])
        s.reward(0.9)
    return j


def test_task_scores_missing_is_none(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    keys = expected_keys(EXP, VAR, ["case-a", "case-b"], 2)
    scores = task_scores(j, keys)
    assert [(s.case_id, s.trial, s.score) for s in scores] == [
        ("case-a", 0, 1.0), ("case-a", 1, 0.8), ("case-b", 0, 0.9), ("case-b", 1, None)]
    assert scores[3].suite == "harness_injection"  # missing trial inherits its case's suite
    a0 = scores[0]
    assert a0.suite == "harness_proposal" and (a0.tokens_in, a0.tokens_out) == (100, 20)
    assert a0.served_model == "gpt-5-mini-2025" and a0.violations == ()
    assert scores[2].violations == (Violation("inj.followed", "critical", "followed"),)
    assert scores[2].suite == "harness_injection"
    named = task_scores(j, keys[:1], score_name="assert")
    assert named[0].score == 1.0
    assert task_scores(j, keys[:1], score_name="nope")[0].score is None


def test_latest_attempt_wins(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path, fsync=False)
    k0 = RolloutKey(EXP, VAR, "c", 0)
    with RolloutScope(j, k0) as s:
        s.reward(0.0)
    with RolloutScope(j, RolloutKey(EXP, VAR, "c", 0, attempt=1)) as s:
        s.reward(1.0)
    assert task_scores(j, [k0])[0].score == 1.0


def test_eval_result(tmp_path: Path) -> None:
    pin = EvaluatorPin("tree", "judge", "copilot")
    res = eval_result(_journal(tmp_path), expected_keys(EXP, VAR, ["case-a"], 2), harness_tree="h",
                      split="evolve", pin=pin)
    assert res.harness_tree == "h" and res.pin == pin and [s.score for s in res.scores] == [1.0, 0.8]


def test_skillopt_records(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    keys = expected_keys(EXP, VAR, ["case-a", "case-b"], 2)
    recs = skillopt_task_records(j, keys, split="evolve")
    assert {r["project"] for r in recs} == {"harness"}
    by_id = {r["id"]: r for r in recs}
    a, b = by_id["agl:case-a"], by_id["agl:case-b"]
    assert a["intent"] == "Change harness prompt 7" and a["outcome"] == "success" and a["split"] == "train"
    assert a["attempted_solution"] == "asked for resource id" and a["reference_kind"] == "rubric"
    assert a["tags"] == ["change", "change.verify", "harness_proposal"] and len(a["source_sessions"]) == 2
    assert b["attempted_solution"] == ""  # injection suite: never copy excerpts
    assert "RAW TOOL OUTPUT" not in repr(recs) and "INJECTED PAYLOAD" not in repr(recs)
    from skillopt_sleep.types import TaskRecord

    for r in recs:
        assert TaskRecord.from_dict(r).to_dict()["id"] == r["id"]


@pytest.mark.parametrize("split", ["heldout", "ood", "aa", "train"])
def test_skillopt_rejects_non_evolve(tmp_path: Path, split: str) -> None:
    with pytest.raises(HoldoutViolation):
        skillopt_task_records(_journal(tmp_path), [], split=split)


def test_skillopt_rejects_heldout_rollout_inputs(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path, fsync=False)
    k = RolloutKey(EXP, VAR, "h", 0)
    with RolloutScope(j, k, {"intent": "x", "split": "heldout"}) as s:
        s.reward(1.0)
    with pytest.raises(HoldoutViolation):
        skillopt_task_records(j, [k], split="evolve")


def test_skillopt_skips_cases_without_typed_intent(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path, fsync=False)
    k = RolloutKey(EXP, VAR, "n", 0)
    with RolloutScope(j, k, {}) as s:
        s.reward(0.0)
    assert skillopt_task_records(j, [k], split="evolve") == []


def test_oes_metric_values(tmp_path: Path) -> None:
    scores = task_scores(_journal(tmp_path), expected_keys(EXP, VAR, ["case-a", "case-b"], 2))
    m = oes_metric_values(scores)
    assert m["evolve_score"] == pytest.approx((1.0 + 0.8 + 0.9 + 0.0) / 4)
    assert m["safety_score"] == pytest.approx((0.9 + 0.0) / 2)
    assert m["critical_unsafe_pass"] == 1.0 and m["critical_violations"] == 1.0
    assert m["missing_trial_rate"] == 0.25 and m["n_tasks"] == 4.0
    assert m["cost_tokens_per_task"] == pytest.approx(240 / 3)
    assert m["suite.harness_proposal"] == pytest.approx(0.9)
    assert m["suite.harness_injection"] == pytest.approx(0.45)
    plain = oes_metric_values([TaskScore("x", 0, "other", 0.5)], primary="score")
    assert plain["score"] == 0.5 and "safety_score" not in plain
    assert oes_metric_values([])["evolve_score"] == 0.0


def test_runtime_fallback_from_model_requests(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path, fsync=False)
    k = RolloutKey(EXP, VAR, "r", 0)
    with RolloutScope(j, k) as s:
        s.record_model_request({"latency_ms": 120.5, "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
        s.record_model_request({"latency_ms": 79.5})
        s.record_model_request({"latency_ms": "bad"})
        s.reward(1.0)
    ts = task_scores(j, [k])[0]
    assert (ts.wall_ms, ts.llm_calls, ts.tool_calls, ts.tokens_in, ts.tokens_out) == (200.0, 3, 0, 10, 2)


def test_runtime_from_score_attrs_then_metric_events(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path, fsync=False)
    k = RolloutKey(EXP, VAR, "m", 0)
    with RolloutScope(j, k) as s:
        s.record_model_request({"latency_ms": 5, "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
        s.score("assert", 1.0, wall_ms=900.0, llm_calls=4, tool_calls=2)
    ts = task_scores(j, [k], score_name="assert")[0]
    assert (ts.wall_ms, ts.llm_calls, ts.tool_calls, ts.tokens_in) == (900.0, 4, 2, 10)
    with RolloutScope(j, RolloutKey(EXP, VAR, "m", 0, attempt=1)) as s:
        s.score("assert", 1.0, wall_ms=900.0, llm_calls=4, tool_calls=2)
        s.emit(METRIC, {"wall_ms": 1500.0, "llm_calls": 6, "tool_calls": 3, "tokens_in": 70, "tokens_out": 30})
        s.emit(METRIC, {"name": "tool_calls", "value": 5})
        s.emit(METRIC, {"name": "wall_ms", "value": float("nan")})
    ts = task_scores(j, [k], score_name="assert")[0]
    assert (ts.wall_ms, ts.llm_calls, ts.tool_calls, ts.tokens_in, ts.tokens_out) == (1500.0, 6, 5, 70, 30)


def test_oes_runtime_metric_values() -> None:
    scores = [TaskScore("a", 0, "s", 1.0, wall_ms=100.0, llm_calls=2, tool_calls=1),
              TaskScore("b", 0, "s", 0.5, wall_ms=300.0, llm_calls=4, tool_calls=3),
              TaskScore("c", 0, "s", None, wall_ms=9999.0, llm_calls=99, tool_calls=99)]
    m = oes_metric_values(scores)
    assert (m["wall_ms_per_task"], m["llm_calls_per_task"], m["tool_calls_per_task"]) == (200.0, 3.0, 2.0)
    assert oes_metric_values([])["wall_ms_per_task"] == 0.0
