"""Campaign model preflight (copilot profile): the model plan, the driver hook and the CLI exit.
No network: listings are injected."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ci_lab.campaign import cli as campaign_cli
from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.preflight import campaign_model_plan, make_preflight
from ci_lab.cli import main
from ci_lab.contracts import Profile
from ci_lab.meta.spec_loader import AGENTS
from ci_lab.providers.models import ModelPreflightError

META = "claude-sonnet-5"
AVAILABLE = [META, "gpt-5-mini", "gpt-5.5"]


def _lister(ids=AVAILABLE, calls: list[int] | None = None):
    async def list_models():
        if calls is not None:
            calls.append(1)
        return [{"id": m} for m in ids]

    return list_models


def _live_env(**extra: str) -> dict[str, str]:
    return {
        "CI_LAB_TARGET_MODEL": "gpt-5-mini",
        "CI_LAB_JUDGE_MODEL": "s1/llamacpp/qwen",
        "CI_S1_LLAMA_URL": "http://127.0.0.1:8081",
        **extra,
    }


def test_plan_agent_only_is_meta_agents_and_target() -> None:
    plan = campaign_model_plan({"strategies": ["agent"]}, env=_live_env())
    users = {u.user for u in plan.copilot}
    assert {f"meta agent {a}" for a in AGENTS} <= users
    assert {u.model for u in plan.copilot} == {META, "gpt-5-mini"}
    assert all("CI_META_MODEL" in u.override for u in plan.copilot if u.user.startswith("meta agent "))
    assert not any("synthesizer" in u or "optimizer" in u for u in users)
    assert plan.served == []  # no offline endpoint configured


def test_plan_follows_ci_meta_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI_META_MODEL", "gpt-5.5")
    plan = campaign_model_plan({"strategies": ["agent", "guard"]}, env=_live_env())
    assert {u.model for u in plan.copilot} == {"gpt-5.5", "gpt-5-mini"}
    assert any(u.user == "lesson synthesizer (guard)" for u in plan.copilot)


def test_plan_rejects_disallowed_meta_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI_META_MODEL", "gpt-6-luna")
    with pytest.raises(ModelPreflightError, match="meta-agent model selection is invalid"):
        campaign_model_plan({}, env=_live_env())


def test_plan_optimizer_checks_copilot_and_the_serve_endpoint(tmp_path: Path) -> None:
    key = tmp_path / "serve.key"
    key.write_text("sekret\n", encoding="utf-8")
    env = _live_env(CI_LAB_OPTIMIZER_MODEL="gpt-5.4-mini",
                    CI_COPILOT_SERVE_URL="http://127.0.0.1:8090/v1/",
                    CI_COPILOT_SERVE_KEY_FILE=str(key))
    plan = campaign_model_plan({"strategies": "agent,gepa,skillopt"}, env=env)
    [opt] = [u for u in plan.copilot if u.user.startswith("optimizer")]
    assert opt.model == "gpt-5.4-mini" and opt.user == "optimizer LM (gepa/skillopt)"
    [(base, k, uses)] = plan.served
    assert (base, k, [u.model for u in uses]) == ("http://127.0.0.1:8090/v1", "sekret", ["gpt-5.4-mini"])
    # An AGL proxy is not a copilot-serve: only the Copilot listing is checked.
    plan = campaign_model_plan({"strategies": ["gepa"]}, env={**env, "AGL_OPENAI_BASE_URL": "http://agl/v1"},
                               )
    assert plan.served == [] and any(u.user.startswith("optimizer") for u in plan.copilot)


def test_plan_agl_optimizer_uses_copilot_without_serve_endpoint() -> None:
    plan = campaign_model_plan({"strategies": ["agl"]},
                               env=_live_env(CI_LAB_OPTIMIZER_MODEL="gpt-5.4-mini"))
    [optimizer] = [u for u in plan.copilot if u.user.startswith("optimizer")]
    assert (optimizer.model, optimizer.user) == ("gpt-5.4-mini", "optimizer LM (agl)")
    assert plan.served == []


def test_plan_unreadable_serve_key_is_a_preflight_error(tmp_path: Path) -> None:
    env = _live_env(CI_COPILOT_SERVE_KEY_FILE=str(tmp_path / "missing.key"))
    with pytest.raises(ModelPreflightError, match="copilot-serve key"):
        campaign_model_plan({"strategies": ["gepa"]}, env=env)


@pytest.mark.parametrize("profile", [Profile.FAKE, Profile.OFFLINE, "fake", "offline"])
def test_make_preflight_skips_non_copilot_profiles(profile: Any) -> None:
    assert make_preflight(profile) is None


def test_make_preflight_passes_and_fails() -> None:
    kw = {"env": _live_env()}
    asyncio.run(make_preflight(Profile.COPILOT, list_models=_lister(), **kw)({}))  # type: ignore[misc]
    with pytest.raises(ModelPreflightError) as ei:
        asyncio.run(make_preflight("copilot", list_models=_lister(["gpt-5-mini"]), **kw)({}))  # type: ignore[misc]
    msg = str(ei.value)
    assert f"'{META}' used by meta agent analyst" in msg and "CI_META_MODEL=<id>" in msg
    assert "Available: gpt-5-mini" in msg


def test_harness_plan_requires_pinned_target_and_s1_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    from ci_lab.campaign import preflight

    monkeypatch.setattr(preflight, "_meta_uses", lambda strategies, meta_harness_dir=None: [])
    with pytest.raises(ModelPreflightError, match="CI_LAB_TARGET_MODEL"):
        campaign_model_plan({"strategies": ["agent"]}, env={})
    env = {
        "CI_LAB_TARGET_MODEL": "gpt-5-mini",
        "CI_LAB_JUDGE_MODEL": "s1/llamacpp/qwen3.5-4b",
        "CI_S1_LLAMA_URL": "http://127.0.0.1:8081",
    }
    plan = campaign_model_plan({"strategies": ["agent"]}, env=env)
    [target] = [use for use in plan.copilot if use.user == "self-hosted harness target"]
    assert target.model == "gpt-5-mini" and target.override == "CI_LAB_TARGET_MODEL=<copilot model id>"
    assert plan.served == []


# ---------------------------------------------------------------- driver + CLI


def _fake_deps(tmp_path: Path):
    return campaign_cli.load_deps(Profile.FAKE, run_root=tmp_path, ledger_dir=None, repo="o/r",
                                  dry_run_publish=True)


def test_driver_runs_preflight_once_before_spending(tmp_path: Path) -> None:
    deps = _fake_deps(tmp_path)
    seen: list[dict[str, Any]] = []

    async def preflight(hyper: Any) -> None:
        seen.append(dict(hyper))

    deps.preflight = preflight
    camp = Campaign.new("pf-camp", Profile.FAKE, {"aa_repeats": 2, "max_rounds": 1}, deps=deps, run_root=tmp_path)
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(1))
    assert len(seen) == 1 and seen[0]["aa_repeats"] == 2
    again = Campaign.load("pf-camp", deps=deps, run_root=tmp_path)
    asyncio.run(again.calibrate())  # cached: no spend, no preflight
    assert len(seen) == 1
    asyncio.run(again.confirm())
    assert len(seen) == 2


def test_driver_preflight_failure_spends_nothing(tmp_path: Path) -> None:
    deps = _fake_deps(tmp_path)

    async def preflight(hyper: Any) -> None:
        raise ModelPreflightError("model 'x' is not available")

    deps.preflight = preflight
    camp = Campaign.new("pf-fail", Profile.FAKE, {"aa_repeats": 2}, deps=deps, run_root=tmp_path)
    with pytest.raises(ModelPreflightError):
        asyncio.run(camp.calibrate())
    assert not camp.status()["calibrated"]
    deps.preflight = None
    asyncio.run(Campaign.load("pf-fail", deps=deps, run_root=tmp_path).calibrate())
    deps.preflight = preflight
    camp = Campaign.load("pf-fail", deps=deps, run_root=tmp_path)
    with pytest.raises(ModelPreflightError):
        asyncio.run(camp.run(1))
    assert not (tmp_path / "pf-fail-r01").exists()
    assert camp.status()["frontier"].get("round", 0) == 0


def test_cli_preflight_failure_exits_2_with_json(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    common = ["--profile", "fake", "--run-dir", str(tmp_path), "--dry-run-publish"]
    assert main(["campaign", "new", "pf-cli", *common, "--hyper", "aa_repeats=2"]) == 0
    capsys.readouterr()
    real = campaign_cli.load_deps

    def load_deps(*a: Any, **kw: Any) -> Any:
        deps = real(*a, **kw)

        async def preflight(hyper: Any) -> None:
            raise ModelPreflightError("Copilot model preflight: 'm' missing\nAvailable: a, b")

        deps.preflight = preflight
        return deps

    monkeypatch.setattr(campaign_cli, "load_deps", load_deps)
    assert main(["campaign", "calibrate", "pf-cli", *common]) == 2
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["error"] == "model preflight failed" and "Available: a, b" in out["detail"]
    assert "ci-lab campaign calibrate" in captured.err
