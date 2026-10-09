"""Every recorded sleep night emits a schema-valid OES envelope (``ci_lab.oes.sleep_envelope``),
including nights where no candidate reached the ASSERT gate (no change, control retained)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ci_lab.contracts import EvaluatorPin
from ci_lab.oes import SLEEP_EXT, validate_envelope
from ci_lab.sleep.budget import BudgetLimits
from ci_lab.sleep.fakes import (
    FAKE_EVALUATOR_TREE,
    RULE_TEXT,
    FakeOracle,
    FakeReflector,
    fake_run_target,
    make_fake_assert_eval,
)
from ci_lab.sleep.night import SKILL_REL, SleepConfig, SleepDeps, run_night
from ci_lab.sleep.registry import HARNESS_EDITING
from ci_lab.sleep.runner import sequential_runner

SHA = "a" * 40
CASES = {f"c{i}": "verify_identity" for i in range(6)}
ENVELOPE = "experiments/sleep/envelopes/sleep-20260921-1.json"


def _cfg(repo: Path, tmp_path: Path, **kw) -> SleepConfig:
    return SleepConfig(repo_root=repo, out_dir=tmp_path / "out" / "sleep-bundle", night_date="20260921",
                       base_sha=SHA, n_boot=300, targets=[HARNESS_EDITING], **kw)


def _deps(**kw) -> SleepDeps:
    kw.setdefault("run_target", fake_run_target)
    kw.setdefault("oracle", FakeOracle())
    kw.setdefault("reflector", FakeReflector())
    kw.setdefault("assert_eval", make_fake_assert_eval(CASES))
    kw.setdefault("latest_delta", lambda: 0.05)
    kw.setdefault("run_workflow", sequential_runner)
    kw.setdefault("clock", lambda: datetime(2026, 9, 21, 7, 17, tzinfo=UTC))
    return SleepDeps(**kw)


def _envelope(cfg: SleepConfig, h, repo: Path, tmp_path: Path) -> dict:
    """The envelope as the publisher would land it: apply the bundle's patch to the checkout."""
    patch = (cfg.out_dir / "candidate.patch").read_text(encoding="utf-8")
    (tmp_path / "p.patch").write_text(patch, encoding="utf-8", newline="\n")
    h.git(repo, "apply", str(tmp_path / "p.patch"))
    env = json.loads((repo / ENVELOPE).read_text(encoding="utf-8"))
    experiment = json.loads((cfg.out_dir / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["envelope"] == env
    return env


def _no_candidate_deps(**kw) -> SleepDeps:
    # the target already follows the rule, so SkillOpt finds nothing to fix and proposes no candidate
    return _deps(reflector=FakeReflector(extra=[]),
                 run_target=lambda t, s, m: fake_run_target(t, s + RULE_TEXT["verify_identity"], m), **kw)


def test_no_candidate_night_records_valid_control_retained_envelope(sleep_repo, tmp_path, h):
    calls: list[str] = []
    fake = make_fake_assert_eval(CASES)

    def assert_eval(skill, memory, variant):
        calls.append(variant)
        return fake(skill, memory, variant)

    assert_eval.evaluator_pin = fake.evaluator_pin
    cfg = _cfg(sleep_repo, tmp_path)
    res = run_night(cfg, _no_candidate_deps(assert_eval=assert_eval))

    assert res.status == "rejected" and calls == [], res.error
    env = _envelope(cfg, h, sleep_repo, tmp_path)
    assert validate_envelope(env) == []
    assert env["experiment"]["id"] == "sleep-20260921"
    assert env["decision"]["outcome"] == "do_not_ship"
    assert "control retained" in env["decision"]["rationale"]
    assert [v["id"] for v in env["variants"]] == ["incumbent"] and env["variants"][0]["role"] == "baseline"
    assert "results" not in env
    ext = env["extensions"][SLEEP_EXT]
    assert ext["candidateDigest"] is None and ext["adoptionPr"] is None
    assert ext["gate"]["assert"]["passed"] is False
    assert ext["gate"]["assert"]["reasons"] == ["SkillOpt pre-filter produced no candidate"]
    assert ext["gate"]["skillopt"]["passed"] is False
    assert ext["evaluatorPin"]["evaluatorTree"] == FAKE_EVALUATOR_TREE
    assert ext["skillPath"] == SKILL_REL and ext["nightIndex"] == 1 and ext["tasks"]["total"] == 6
    assert ext["incumbentDigest"].startswith("sha256:")
    assert {c["checkType"]: c["status"] for c in env["qualityChecks"]}["assert_gate"] == "not_run"


def test_candidate_elected_night_records_valid_ship_envelope(sleep_repo, tmp_path, h):
    cfg = _cfg(sleep_repo, tmp_path)
    res = run_night(cfg, _deps())

    assert res.status == "accepted", res.error
    env = _envelope(cfg, h, sleep_repo, tmp_path)
    assert validate_envelope(env) == []
    assert env["decision"]["outcome"] == "ship"
    assert {v["id"]: v["role"] for v in env["variants"]} == {"incumbent": "baseline", "candidate": "treatment"}
    ext = env["extensions"][SLEEP_EXT]
    assert ext["gate"]["assert"]["passed"] is True and ext["candidateDigest"] != ext["incumbentDigest"]


def test_budget_exceeded_before_gate_records_valid_rerun_envelope(sleep_repo, tmp_path, h):
    cfg = _cfg(sleep_repo, tmp_path, limits=BudgetLimits(max_rollouts=2))
    res = run_night(cfg, _deps())

    assert res.status == "budget_exceeded"
    env = _envelope(cfg, h, sleep_repo, tmp_path)
    assert validate_envelope(env) == []
    assert env["decision"]["outcome"] == "rerun"
    assert env["scorecard"]["qualityStatus"] == "invalid"
    assert any(c["checkType"] == "sleep_budget" and c["status"] == "fail" for c in env["qualityChecks"])


def test_no_tasks_night_records_valid_envelope(sleep_repo, tmp_path, h):
    h.write_tasks(sleep_repo / HARNESS_EDITING.tasks_file, [])
    h.git(sleep_repo, "commit", "-qam", "no tasks")
    cfg = _cfg(sleep_repo, tmp_path)
    res = run_night(cfg, _deps())

    assert res.status == "no_tasks", res.error
    env = _envelope(cfg, h, sleep_repo, tmp_path)
    assert validate_envelope(env) == []
    assert env["decision"]["outcome"] == "do_not_ship"
    assert env["extensions"][SLEEP_EXT]["tasks"]["total"] == 0


def test_explicit_evaluator_pin_dep_wins(sleep_repo, tmp_path, h):
    pin = EvaluatorPin(evaluator_tree="c0ffee" * 6 + "c0ff", judge_model="j/m", judge_provider="j")
    cfg = _cfg(sleep_repo, tmp_path)
    res = run_night(cfg, _no_candidate_deps(evaluator_pin=lambda: pin))

    assert res.status == "rejected", res.error
    env = _envelope(cfg, h, sleep_repo, tmp_path)
    assert validate_envelope(env) == []
    assert env["extensions"][SLEEP_EXT]["evaluatorPin"]["judgeModel"] == "j/m"


def test_night_without_evaluator_pin_fails_closed(sleep_repo, tmp_path):
    def assert_eval(skill, memory, variant):
        raise AssertionError("must not run")

    cfg = _cfg(sleep_repo, tmp_path)
    res = run_night(cfg, _no_candidate_deps(assert_eval=assert_eval))

    assert res.status == "error" and "evaluator pin" in res.error
    assert not res.ledger_update


def test_custom_envelope_builder_output_is_validated(sleep_repo, tmp_path):
    cfg = _cfg(sleep_repo, tmp_path)
    res = run_night(cfg, _deps(build_envelope=lambda payload: {"schemaVersion": "0.1.0", "objectType": "experiment",
                                                                "experiment": {"id": "x", "title": "x"},
                                                                "extensions": {SLEEP_EXT: {"night": "bad"}}}))

    assert res.status == "error" and "invalid OES envelope" in res.error
    assert not res.ledger_update
