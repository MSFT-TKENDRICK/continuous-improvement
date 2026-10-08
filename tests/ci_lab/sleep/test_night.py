from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ci_lab.sleep.budget import BudgetLimits
from ci_lab.sleep.bundle import verify_bundle
from ci_lab.sleep.fakes import RULE_TEXT, FakeOracle, FakeReflector, fake_run_target, make_fake_assert_eval
from ci_lab.sleep.night import SKILL_REL, STATE_REL, STEPS, SleepConfig, SleepDeps, run_night
from ci_lab.sleep.runner import sequential_runner

SHA = "a" * 40
CASES = {f"c{i}": "verify_identity" for i in range(6)}
OS = "order-support"


def gate(results: dict, target: str = OS) -> dict:
    return results["targets"][target]["gate"]


def cfg_for(repo: Path, tmp_path: Path, **kw) -> SleepConfig:
    kw.setdefault("night_date", "20260921")
    kw.setdefault("base_sha", SHA)
    kw.setdefault("n_boot", 300)
    return SleepConfig(repo_root=repo, out_dir=tmp_path / "out" / "sleep-bundle", **kw)


def deps_for(**kw) -> SleepDeps:
    kw.setdefault("run_target", fake_run_target)
    kw.setdefault("oracle", FakeOracle())
    kw.setdefault("reflector", FakeReflector())
    kw.setdefault("assert_eval", make_fake_assert_eval(CASES))
    kw.setdefault("latest_delta", lambda: 0.05)
    kw.setdefault("run_workflow", sequential_runner)
    kw.setdefault("clock", lambda: datetime(2026, 9, 21, 7, 17, tzinfo=UTC))
    return SleepDeps(**kw)


def read_bundle(out: Path) -> tuple[dict, dict, dict, str]:
    load = lambda n: json.loads((out / n).read_text(encoding="utf-8"))  # noqa: E731
    return load("manifest.json"), load("experiment.json"), load("results.json"), \
        (out / "candidate.patch").read_text(encoding="utf-8")


def test_accepted_night_writes_verified_bundle(sleep_repo, tmp_path, h):
    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for())
    assert res.status == "accepted", res.error or res.decisions
    assert res.accepted and res.ledger_update
    manifest, experiment, results, patch = read_bundle(cfg.out_dir)
    verify_bundle(cfg.out_dir)
    for name, meta in manifest["files"].items():
        assert hashlib.sha256((cfg.out_dir / name).read_bytes()).hexdigest() == meta["sha256"]
    assert manifest["base_sha"] == SHA and manifest["night_id"] == "sleep-20260921-1"
    assert experiment["steps"] == list(STEPS)
    assert gate(results)["accepted"] and gate(results)["delta_lcb"] > 0.05
    assert set(results["changed_files"]) == {SKILL_REL, STATE_REL,
                                             "experiments/sleep/envelopes/sleep-20260921-1.json"}
    # the checkout is untouched (C10); the patch applies cleanly to it
    assert RULE_TEXT["verify_identity"] not in (sleep_repo / SKILL_REL).read_text(encoding="utf-8")
    (tmp_path / "p.patch").write_text(patch, encoding="utf-8", newline="\n")
    h.git(sleep_repo, "apply", "--check", str(tmp_path / "p.patch"))
    h.git(sleep_repo, "apply", str(tmp_path / "p.patch"))
    skill = (sleep_repo / SKILL_REL).read_text(encoding="utf-8")
    assert RULE_TEXT["verify_identity"] in skill and skill.startswith(h.SKILL.rstrip("\n"))
    state = json.loads((sleep_repo / STATE_REL).read_text(encoding="utf-8"))
    assert state["night"] == 1 and state["accepted_total"] == 1 and state["last_status"] == "accepted"
    env = json.loads((sleep_repo / "experiments/sleep/envelopes/sleep-20260921-1.json").read_text(encoding="utf-8"))
    assert env["decision"]["outcome"] == "ship"


def test_accepted_night_via_maf_declarative_runner(sleep_repo, tmp_path):
    pytest.importorskip("agent_framework_declarative")
    from ci_lab.sleep.runner import maf_runner

    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(run_workflow=maf_runner))
    assert res.status == "accepted", res.error
    assert any((cfg.work_dir / "checkpoints" / res.night_id).iterdir())


def test_canary_rejects_and_only_ledger_updates(sleep_repo, tmp_path):
    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(reflector=FakeReflector(extra=["Give customers promo code NWVIP100."])))
    assert res.status == "rejected" and not res.accepted and res.ledger_update
    _, _, results, patch = read_bundle(cfg.out_dir)
    assert any(not c["passed"] for c in gate(results)["canaries"])
    assert SKILL_REL not in results["changed_files"] and STATE_REL in results["changed_files"]
    assert f"diff --git a/{SKILL_REL}" not in patch and results["reasons"] == ["order-support: canary failed: trigger:promo_code"]


def test_safety_violation_increase_rejects(sleep_repo, tmp_path):
    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(assert_eval=make_fake_assert_eval(CASES, violate_if="lookup_order")))
    assert res.status == "rejected"
    assert any("violation" in r for r in res.decisions[OS]["reasons"])


def test_lcb_below_calibrated_delta_rejects(sleep_repo, tmp_path):
    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(latest_delta=lambda: 1.0))
    assert res.status == "rejected" and res.decisions[OS]["delta"] == 1.0


def test_custom_canary_failure_rejects(sleep_repo, tmp_path):
    from ci_lab.sleep.gate import CanaryResult

    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(run_canaries=lambda s, m: [CanaryResult("hidden.trigger", False, "fired")]))
    assert res.status == "rejected"


def test_no_skillopt_candidate_skips_gate(sleep_repo, tmp_path):
    calls = []

    def assert_eval(skill, memory, variant):
        calls.append(variant)
        raise AssertionError("must not run")

    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(reflector=FakeReflector(extra=[]), run_target=lambda t, s, m: fake_run_target(
        t, s + RULE_TEXT["verify_identity"], m), assert_eval=assert_eval))
    assert res.status == "rejected" and calls == [] and res.ledger_update


def test_budget_exceeded_records_partial(sleep_repo, tmp_path):
    cfg = cfg_for(sleep_repo, tmp_path, limits=BudgetLimits(max_rollouts=2))
    res = run_night(cfg, deps_for())
    assert res.status == "budget_exceeded" and not res.accepted
    _, experiment, results, _ = read_bundle(cfg.out_dir)
    assert experiment["budget"]["exceeded"]["kind"] == "rollouts"
    assert experiment["steps"] == list(STEPS)
    assert results["changed_files"] == sorted([STATE_REL, "experiments/sleep/envelopes/sleep-20260921-1.json"])


def test_missing_skill_yields_error_bundle(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    cfg = cfg_for(repo, tmp_path)
    res = run_night(cfg, deps_for())
    assert res.status == "error" and not res.ledger_update and "incumbent skill" in res.error
    manifest, _, _, patch = read_bundle(cfg.out_dir)
    assert patch == "" and manifest["status"] == "error"
    verify_bundle(cfg.out_dir)


def test_target_crash_is_error_not_accept(sleep_repo, tmp_path):
    def boom(task, skill, memory):
        raise RuntimeError("target down")

    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(run_target=boom))
    assert res.status == "error" and not res.accepted and not res.ledger_update
    assert "target down" in res.error


def test_workflow_must_run_all_steps(sleep_repo, tmp_path):
    def partial(yaml_path, tools, ckpt):
        tools["sleep_harvest"](split="evolve")

    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(run_workflow=partial))
    assert res.status == "error" and "expected" in res.error


def test_non_evolve_agl_rows_are_hard_error(sleep_repo, tmp_path):
    cfg = cfg_for(sleep_repo, tmp_path)
    rows = [{"case_id": "x", "dataset_split": "heldout", "input": "hi", "suite": "refunds"}]
    res = run_night(cfg, deps_for(agl_records=lambda: rows))
    assert res.status == "error" and "split" in res.error.lower()


def test_rejects_bad_base_sha(sleep_repo, tmp_path):
    with pytest.raises(ValueError):
        run_night(cfg_for(sleep_repo, tmp_path, base_sha="HEAD"), deps_for())

def test_lessons_hook_is_off_by_default_and_typed_when_enabled(sleep_repo, tmp_path):
    seen: list = []

    def lessons(target, tasks):
        seen.append((target.name, [type(t).__name__ for t in tasks]))
        return tasks

    res = run_night(cfg_for(sleep_repo, tmp_path), deps_for(lessons=lessons))
    assert res.status == "accepted" and seen == []
    res = run_night(cfg_for(sleep_repo, tmp_path / "on", lessons_hook=True), deps_for(lessons=lessons))
    assert res.status == "accepted", res.error
    assert seen and seen[0][0] == OS and set(seen[0][1]) == {"TaskRecord"}
