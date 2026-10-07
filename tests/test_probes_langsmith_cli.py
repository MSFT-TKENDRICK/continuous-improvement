from __future__ import annotations

import json
import shutil
import inspect

import pytest

from s1eval.backends.scripted import ScriptedBackend
from s1eval.cli import main
from s1eval.langsmith_integration import default_to_observable, make_evaluator
from s1eval.probes import probe_choice_order, probe_complement, probe_distractor
from s1eval.runner import run_cases
from s1eval.types import Answer, Question


def _mini_rubric():
    from s1eval.rubric import Condition, Rubric

    return Rubric(
        name="mini",
        version="1",
        questions={
            "ok": Question("noul", "Is it ok?"),
            "pick": Question("choice", "Pick one", {"a": None, "b": None, "c": None, "d": None}),
        },
        complements={"ok": Question("noul", "Is it not ok?")},
        pass_rule=[Condition("ok", expect=True), Condition("pick", allowed=["a", "b", "c", "d"])],
    )


def _case(cid="case", final_response="ok"):
    return {"id": cid, "observable": {"conversation": [{"role": "user", "content": "hi"}], "tool_calls": [], "final_response": final_response}}


def test_complement_probe_detects_yes_biased_judge():
    rubric = _mini_rubric()
    backend = ScriptedBackend(lambda _state, _name, q: Answer.from_noul_probability(0.9) if q.type == "noul" else Answer.from_choice_distribution({k: 1 / len(q.criteria) for k in q.criteria}))
    cases = [_case("a")]
    base = run_cases(backend, rubric, cases)
    out = probe_complement(backend, rubric, cases, base)
    assert out["ok"]["contradiction_rate"] == 1


def test_choice_order_probe_detects_first_option_bias_and_invariant_judge():
    rubric = _mini_rubric()
    cases = [_case("a")]
    first = ScriptedBackend(
        lambda _state, _name, q: Answer.from_choice_distribution({k: (1.0 if i == 0 else 0.0) for i, k in enumerate(q.criteria)})
        if q.type == "choice"
        else Answer.from_noul_probability(0.9)
    )
    first_base = run_cases(first, rubric, cases)
    first_out = probe_choice_order(first, rubric, cases, first_base)["pick"]
    assert first_out["first_shown_rate"] == 1.0
    assert first_out["flip_rate"] == 1.0

    invariant = ScriptedBackend(
        lambda _state, _name, q: Answer.from_choice_distribution({k: (1.0 if k == "a" else 0.0) for k in q.criteria})
        if q.type == "choice"
        else Answer.from_noul_probability(0.9)
    )
    inv_base = run_cases(invariant, rubric, cases)
    inv_out = probe_choice_order(invariant, rubric, cases, inv_base)["pick"]
    assert inv_out["flip_rate"] == 0.0
    assert inv_out["first_shown_rate"] == 0.25


def test_distractor_probe_flips_when_backend_reacts_to_distractor_key():
    rubric = _mini_rubric()

    def fn(state, _name, q):
        if q.type == "choice":
            return Answer.from_choice_distribution({"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0})
        return Answer.from_noul_probability(0.1 if "distract" in state else 0.9)

    backend = ScriptedBackend(fn)
    cases = [_case("a")]
    base = run_cases(backend, rubric, cases)
    out = probe_distractor(backend, rubric, cases, base, [{"key": "distract", "text": "irrelevant"}])
    assert out["flip_rate"]["ok"] == 1.0


def test_distractor_probe_separates_abstentions_from_flips():
    """ok -> abstain must not dilute flip_rate's denominator nor be silently ignored (review finding)."""
    rubric = _mini_rubric()

    def fn(state, _name, q):
        if q.type == "choice":
            return Answer.from_choice_distribution({"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0})
        if "distract" in state:
            return Answer.non_answer("noul", "abstain", reason="test")
        return Answer.from_noul_probability(0.9)

    backend = ScriptedBackend(fn)
    cases = [_case("a"), _case("b")]
    base = run_cases(backend, rubric, cases)
    out = probe_distractor(backend, rubric, cases, base, [{"key": "distract", "text": "irrelevant"}])
    assert out["n_comparable"]["ok"] == 0
    assert out["flip_rate"]["ok"] is None
    assert out["status_change_rate"]["ok"] == 1.0
    assert out["flip_rate"]["pick"] == 0.0 and out["n_comparable"]["pick"] == 4


def test_langsmith_evaluator_shape_signature_and_default_mapping(rubric, scripted_backend):
    evaluator = make_evaluator(scripted_backend, rubric)
    sig = inspect.signature(evaluator)
    assert list(sig.parameters) == ["inputs", "outputs"]
    result = evaluator({"question": "Where is order X?"}, {"output": "It shipped."})
    keys = {r["key"] for r in result["results"]}
    assert set(rubric.questions) | {"pass", "needs_review"} <= keys
    obs = default_to_observable({"question": "Q"}, {"output": "A", "tool_calls": [{"name": "lookup", "arguments": {}, "result": {}, "hidden": "x"}]})
    assert obs["conversation"] == [{"role": "user", "content": "Q"}]
    assert "hidden" not in obs["tool_calls"][0]
    assert obs["final_response"] == "A"


def test_cli_validate_export_run_and_report_offline(repo_root, project_scratch, monkeypatch, capsys):
    monkeypatch.chdir(repo_root)
    assert main(["validate"]) == 0
    validate_out = json.loads(capsys.readouterr().out)
    assert validate_out["cases"] == 30
    assert validate_out["rule_vs_human_pass_disagreements"] == []

    assert main(["export-langsmith"]) == 0
    exported = capsys.readouterr().out
    for name in ["grounded", "tool_use", "resolution", "pii_leak", "obeyed_injection"]:
        assert name in exported

    out_dir = project_scratch / "run"
    assert main(["run", "--backend", "scripted", "--probes", "none", "--limit", "3", "--out", str(out_dir)]) == 0
    assert (out_dir / "records.jsonl").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "report.md").exists()
    first_records = (out_dir / "records.jsonl").read_text(encoding="utf-8")
    assert len([line for line in first_records.splitlines() if line.strip()]) == 3

    (out_dir / "report.md").unlink()
    assert main(["report", "--dir", str(out_dir)]) == 0
    assert (out_dir / "report.md").exists()
