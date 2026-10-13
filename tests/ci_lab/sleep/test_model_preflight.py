"""Sleep and ``ci-lab doctor --models`` model preflights (no network: listings are faked)."""

from __future__ import annotations

import argparse
import json

import pytest

from ci_lab import cli as root_cli
from ci_lab.contracts import Profile
from ci_lab.providers import models as provider_models
from ci_lab.sleep.wiring import model_uses


@pytest.fixture(autouse=True)
def _default_sleep_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CI_LAB_SLEEP_TARGET_MODEL", "CI_LAB_SLEEP_REFLECTOR_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _ids(*ids: str):
    async def list_models():
        return [{"id": m} for m in ids]

    return list_models


def test_model_uses_copilot_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert model_uses(Profile.FAKE) == [] and model_uses(Profile.OFFLINE) == []
    monkeypatch.delenv("CI_LAB_SLEEP_TARGET_MODEL", raising=False)
    monkeypatch.setenv("CI_LAB_SLEEP_REFLECTOR_MODEL", "gpt-5.5")
    uses = {u.user: u for u in model_uses(Profile.COPILOT)}
    assert uses["sleep target"].model == "gpt-5-mini"
    assert uses["sleep reflector"].model == "gpt-5.5"
    assert uses["sleep reflector"].override == "CI_LAB_SLEEP_REFLECTOR_MODEL=<id>"


def test_sleep_run_fails_before_the_night_when_a_model_is_missing(tmp_path, monkeypatch, capsys) -> None:
    from ci_lab.sleep import cli

    monkeypatch.setattr("ci_lab.sleep.wiring.build_deps", lambda *a, **kw: object())
    monkeypatch.setattr(provider_models, "copilot_model_ids", _ids("claude-sonnet-5"))

    def no_night(*a, **kw):
        raise AssertionError("the night must not start")

    monkeypatch.setattr("ci_lab.sleep.night.run_night", no_night)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    parser = argparse.ArgumentParser()
    cli.register(parser.add_subparsers(dest="cmd"))
    args = parser.parse_args(["sleep", "run", "--profile", "copilot", "--out", str(tmp_path / "out"),
                              "--date", "20260921", "--base-sha", "a" * 40])
    assert args.func(args) == 1
    err = capsys.readouterr().err
    assert "model preflight failed" in err and "'gpt-5-mini' used by sleep target, sleep reflector" in err
    assert "CI_LAB_SLEEP_TARGET_MODEL=<id>" in err and "Available: claude-sonnet-5" in err


def test_doctor_models_reports_every_copilot_model(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"OPENAI_API_BASE": "http://h/v1", "CI_COPILOT_SERVE_KEY": "k"}
    served = {"http://h/v1": ["local"], "http://127.0.0.1:8765/v1": ["gpt-5-mini"]}
    report = root_cli.doctor_models(env=env, list_models=_ids("claude-sonnet-5", "gpt-5-mini"),
                                    fetch=lambda base, key: served[base])
    assert report["ok"] is True, report
    users = {r["user"] for r in report["required"]}
    assert {"meta agent analyst", "lesson synthesizer (guard)", "sleep target"} <= users
    assert any(u.startswith("optimizer LM") for u in users)
    assert {s["base_url"] for s in report["served"]} == set(served)

    report = root_cli.doctor_models(env=env, list_models=_ids("gpt-5-mini"), fetch=lambda base, key: [])
    assert report["ok"] is False
    assert {r["model"] for r in report["required"] if not r["available"]} == {"claude-sonnet-5"}
    assert all(not r["available"] for s in report["served"] for r in s["required"])

    async def down():
        raise OSError("no cli")

    assert root_cli.doctor_models(env=env, list_models=down)["ok"] is False


def test_doctor_models_flag(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(root_cli, "doctor_models", lambda: {"ok": False, "error": "x"})
    assert root_cli.main(["doctor", "--models"]) == 1
    assert json.loads(capsys.readouterr().out)["models"] == {"ok": False, "error": "x"}
    monkeypatch.setattr(root_cli, "doctor_models", lambda: {"ok": True})
    assert root_cli.main(["doctor", "--models"]) == 0
