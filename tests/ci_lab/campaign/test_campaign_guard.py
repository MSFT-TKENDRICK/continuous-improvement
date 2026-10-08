"""21b wiring: guard arms (v2.4 §13) run ``arm_guard.yaml`` inside fake-profile campaign rounds."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.fakes import fake_deps
from ci_lab.lessons_arm.paired import guard_paired_eval_step
from ci_lab.workflows import ARM_YAMLS, function_names

CID = "guard-camp"


def _new(tmp_path: Path, hyper: dict[str, Any] | None = None, **kw: Any):
    deps = fake_deps(tmp_path, **kw)
    camp = Campaign.new(CID, "fake", {"arms": 2, "aa_repeats": 2, "max_rounds": 4,
                                      "strategies": ["agent", "guard"], **(hyper or {})},
                        deps=deps, run_root=tmp_path / "runs")
    return camp, deps


def _guarded(marker: str):
    """c1 violates unless guards are enforced and the harness carries the guard arm's edit."""

    def fn(edits: Sequence[str], case: str, trial: int) -> str | None:
        if case != "c1":
            return None
        guarded = os.environ.get("CI_GUARDS") == "enforce" and any(marker in e for e in edits)
        return None if guarded else "stub.unsafe"

    return fn


def test_arm_guard_yaml_is_bound() -> None:
    from ci_lab import lessons_arm

    path = ARM_YAMLS["guard"]
    assert path == Path(lessons_arm.__file__).resolve().parent / "workflows" / "arm_guard.yaml"
    assert path.is_file()
    assert "guard_paired_eval" in function_names(path)


def test_guard_paired_eval_step_signature_matches_binding() -> None:
    params = inspect.signature(guard_paired_eval_step).parameters
    for name in ("domain", "worktree", "run_dir", "split", "k", "experiment_id", "variant", "trials",
                 "stochastic", "incumbent", "margin"):
        assert name in params


def test_agent_and_guard_arms_round(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path)
    deps.domain.violation_fn = _guarded("tweak v2")
    asyncio.run(camp.calibrate())
    out = asyncio.run(camp.run(rounds=1))
    assert out["rounds"][0]["winner"] == "v1"  # the agent arm's "boost" edit wins on score

    run_dir = tmp_path / "runs" / f"{CID}-r01"
    begin = json.loads((run_dir / "begin.json").read_text())
    assert [(d["strategy"], d["component"]) for d in begin["directives"]] == [("agent", "skill"),
                                                                                ("guard", "guard")]
    guard = json.loads((run_dir / "v2" / "guard_eval.json").read_text())
    assert guard["skipped"] is False and guard["ship"] == {"ok": True, "reasons": []}
    assert guard["metrics"]["paired"] is True
    assert guard["metrics"]["delivered_violation_rate"] == 0.0
    assert guard["metrics"]["attempted_violation_rate"] == pytest.approx(0.25)
    inc = json.loads((run_dir / "inc-guard" / "guard_eval.json").read_text())
    assert inc["metrics"]["delivered_violation_rate"] == pytest.approx(0.25)

    done = json.loads((run_dir / "v2" / "arm.done").read_text())
    assert done["result"]["strategy"] == "guard" and done["result"]["status"] == "evaluated"
    assert done["guard"]["ship"]["ok"] is True
    assert done["result"]["edits"][0]["files"] == ["harness/guards/v2.yaml"]
    assert not (run_dir / "v1" / "guard_eval.json").exists()  # agent arms skip the guard gate
    # Paired runs: guard-off / guard-on of the arm and of the incumbent, identical cases.
    variants = [v for e, v, _ in deps.domain.calls if e == f"{CID}-r01"]
    assert {"v2-off-0", "v2-on-0", "inc-guard-off-0", "inc-guard-on-0"} <= set(variants)


def test_guard_arm_without_effect_fails_ship_rule(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path)
    deps.domain.violation_fn = _guarded("never-matches")
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))
    run_dir = tmp_path / "runs" / f"{CID}-r01"
    done = json.loads((run_dir / "v2" / "arm.done").read_text())
    assert done["result"]["status"] == "rejected"
    assert done["reason"].startswith("guard_ship_rule: delivered_violation_rate")
    sel = json.loads((run_dir / "selection.json").read_text())
    assert next(t for t in sel["trace"] if t["arm"] == "v2")["admissible"] is False


def test_guard_eval_is_stochastic_on_copilot_profile(tmp_path: Path) -> None:
    from ci_lab.contracts import Profile
    from ci_lab.workflows.steps import CampaignEnv, guard_eval_options

    deps = fake_deps(tmp_path)
    env = CampaignEnv("c", Profile.COPILOT, {"k": 1}, deps, tmp_path)
    assert guard_eval_options(env) == {"k": 1, "trials": 3, "stochastic": True, "margin": 0.0}
    env = CampaignEnv("c", Profile.FAKE, {"k": 2, "guard_trials": 4, "guard_margin": 0.05}, deps, tmp_path)
    assert guard_eval_options(env) == {"k": 2, "trials": 4, "stochastic": False, "margin": 0.05}
