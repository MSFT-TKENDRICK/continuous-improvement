"""Real ASSERT integration for the five frozen harness suites and System-1 judge provider."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

from ci_lab.domain.harness import HarnessDomain
from ci_lab.domain.harness_assert_wrapper import command
from ci_lab.testing import s1_top_token

REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"
S1_JUDGE = "s1/llamacpp/qwen3.5-4b"
HARNESS_SUITES = {
    "harness_injection",
    "harness_proposal",
    "harness_taskgraph",
    "harness_tool_use",
    "harness_triage",
}
CFG = EVALS / "assert" / "harness_triage" / "eval_config.yaml"


def _judge_configs() -> list[Path]:
    out = []
    for path in sorted((EVALS / "assert").glob("harness_*/eval_config.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "judge" in (cfg.get("pipeline") or {}):
            out.append(path)
    return out


def test_only_five_harness_suites_are_loaded_and_use_system1_judges():
    configs = _judge_configs()
    assert {path.parent.name for path in configs} == HARNESS_SUITES
    assert {path.parent.name for path in (EVALS / "assert").glob("*/eval_config.yaml")} == HARNESS_SUITES
    for path in configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = cfg["pipeline"]["judge"].get("model") or cfg.get("default_model") or {}
        name = model.get("name") if isinstance(model, dict) else model
        assert str(name).startswith("s1/"), f"{path.relative_to(REPO)} judges with {name!r}"
    domain = HarnessDomain(profile="offline", tier="evolve", target_model="openai/target",
                           judge_model=S1_JUDGE)
    assert domain.pin().judge_provider == "llamacpp"


def test_harness_assert_wrapper_targets_harness_suite_only():
    argv = command(CFG, "--model-timeout", "30")
    assert argv[:3] == [sys.executable, "-m", "ci_lab.domain.harness_assert_wrapper"]
    assert argv[3:] == ["run", str(CFG), "--model-timeout", "30"]


def test_provider_check_runs_harness_assert_judge_call_on_s1_path(loopback, monkeypatch):
    from ci_lab.judge.cli import provider_check

    srv = loopback(lambda body: pytest.fail(f"fallback chat judge was called: {body.get('model')}"),
                   judge=s1_top_token)
    monkeypatch.setenv("CI_S1_LLAMA_URL", srv.url)
    monkeypatch.setenv("OPENAI_API_BASE", f"{srv.url}/v1")
    model = f"s1/llamacpp/loopback-{id(srv)}"
    out = provider_check(model, str(CFG), allow_fallback=True)
    assert out["ok"] and out["path"] == "s1"
    assert out["stats"] == {"s1": 1, "fallback": 0}
    assert set(out["verdict"]["dimensions"]) >= set(out["score_keys"])
    assert srv.of_kind("judge") and not srv.of_kind("json_schema") and not srv.of_kind("chat")
