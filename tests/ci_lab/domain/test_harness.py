from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ci_lab.contracts import Profile
from ci_lab.domain import DEFAULT_DOMAIN, DOMAIN_CHOICES, get_domain
from ci_lab.domain.harness import (
    FROZEN_GLOBS,
    JUDGE_MODEL_ENV,
    PROFILE_ENV,
    SURFACE_GLOBS,
    TARGET_MODEL_ENV,
    HarnessDomain,
)
from ci_lab.domain.harness_assert_wrapper import current_case_id
from ci_lab.domain.harness_runner import run_case
from ci_lab.domain.harness_scoring import grade_case, load_rubric
from ci_lab.domain.layout import harness_root
from ci_lab.harness_tree import HarnessTree, HarnessTreeError, tree_digest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    path = tmp_path / "harness"
    shutil.copytree(ROOT / "harness", path)
    return path


def _domain_for(case_ids: list[str], **kwargs: Any) -> HarnessDomain:
    return HarnessDomain(profile="fake", tier="ci", split_map={"evolve": case_ids}, **kwargs)


def _evaluate(domain: HarnessDomain, candidate: Path, k: int = 1):
    return asyncio.run(domain.evaluate(
        candidate, "evolve", k, experiment_id="harness-test-r00", variant="candidate"))


def test_manifest_surface_splits_and_registry_defaults() -> None:
    domain = HarnessDomain()
    assert domain.name == "harness"
    assert DEFAULT_DOMAIN == "order_support"
    assert DOMAIN_CHOICES == ("order_support", "harness")
    assert isinstance(get_domain("harness"), HarnessDomain)
    assert "harness/harness.yaml" not in SURFACE_GLOBS
    assert {"src/**", "evals/**", "schemas/**", ".github/**", "third_party/**",
            "harness/harness.yaml"} <= set(FROZEN_GLOBS)
    assert {name: len(ids) for name, ids in domain.splits().items()} == {
        "evolve": 15, "heldout": 15, "ood": 20, "aa": 15}
    assert domain.component_globs == HarnessTree(ROOT / "harness").component_globs()
    assert harness_root(domain) == "harness"


def test_fake_smoke_all_five_suites_and_surface(candidate: Path) -> None:
    ids = [f"harness_{kind}_001" for kind in ("triage", "proposal", "taskgraph", "tool_use", "injection")]
    result = _evaluate(_domain_for(ids, concurrency=3), candidate)
    assert {score.suite for score in result.scores} == {
        "harness_triage", "harness_proposal", "harness_taskgraph",
        "harness_tool_use", "harness_injection"}
    assert all(score.score == 1.0 for score in result.scores)
    assert all(score.llm_calls >= 1 for score in result.scores)
    assert result.surface["tree_valid"] == 1.0
    assert result.surface["complexity"] > 0
    assert result.surface["component.prompt.complexity"] > 0
    assert result.pin.judge_model == "s1/scripted/default"
    assert result.pin.judge_provider == "scripted"


def test_split_k_aggregation(candidate: Path) -> None:
    result = _evaluate(_domain_for(["harness_triage_001", "harness_injection_001"]), candidate, k=2)
    assert [(score.case_id, score.trial) for score in result.scores] == [
        ("harness_triage_001", 0), ("harness_triage_001", 1),
        ("harness_injection_001", 0), ("harness_injection_001", 1),
    ]


def test_candidate_copy_and_scrubbed_environment(candidate: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("GIT_ASKPASS", "steal")
    observed: dict[str, Any] = {}

    def process(command: Any, cwd: Path, env: Any, timeout: float):
        observed.update(command=list(command), cwd=Path(cwd), env=dict(env), timeout=timeout)
        copied = Path(command[command.index("--harness-dir") + 1])
        assert copied != candidate.resolve() and copied.is_dir()
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps({
            "score": 1.0, "violations": [], "metrics": {
                "wall_ms": 1, "llm_calls": 1, "tool_calls": 0, "tokens_in": 2, "tokens_out": 3,
                "tokens": 5, "output_chars": 2, "output_lines": 1},
            "rubric_scores": {"exact-labels": 1}, "subscores": {"resource_score": 1},
            "served_model": "fake-model", "judge_model": "s1/scripted/default", "excerpt": "ok",
        }), encoding="utf-8")
        return 0, b"", b"", False

    before = tree_digest(candidate)
    result = _evaluate(_domain_for(["harness_triage_001"], process_runner=process), candidate)
    assert result.scores[0].score == 1.0
    assert tree_digest(candidate) == before
    assert not observed["cwd"].is_relative_to(ROOT)
    assert {"GH_TOKEN", "GITHUB_TOKEN", "GIT_ASKPASS"}.isdisjoint(observed["env"])
    assert set(observed["env"]) <= {
        "PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONPATH",
        "CI_COPILOT_SERVE_URL", "CI_COPILOT_SERVE_KEY", "CI_S1_LLAMA_URL",
        PROFILE_ENV, TARGET_MODEL_ENV, JUDGE_MODEL_ENV,
    }
    assert observed["env"][PROFILE_ENV] == "fake"
    assert observed["env"][TARGET_MODEL_ENV] == "fake-model"
    assert observed["env"][JUDGE_MODEL_ENV] == "s1/scripted/default"


def test_invalid_tree_fails_before_model_calls(candidate: Path) -> None:
    (candidate / "harness.yaml").write_text("format: changed\n", encoding="utf-8")
    calls = 0

    def process(*_args: Any):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    with pytest.raises(HarnessTreeError, match="invalid candidate harness"):
        _evaluate(_domain_for(["harness_triage_001"], process_runner=process), candidate)
    assert calls == 0


def test_budget_exceeded_scores_zero(candidate: Path, tmp_path: Path) -> None:
    domain = HarnessDomain()
    row = dict(domain.case("harness_tool_use_001").row)
    row["fake_script"] = [
        {"tool_calls": [{"name": "list_components", "arguments": {}} for _ in range(121)]},
        {"text": row["expected"]["answer"]},
    ]
    row = domain._copy_case(row, tmp_path)
    result = asyncio.run(run_case(
        row, harness_dir=candidate, profile="fake", tier="ci",
        target_model="fake-model", judge_model="s1/scripted/default"))
    assert result["score"] == 0.0
    assert any(v["rule_id"] == "budget.exceeded" for v in result["violations"])
    assert result["metrics"]["tool_calls"] == 121


def test_model_mismatch_is_invalid_trial(candidate: Path) -> None:
    def process(command: Any, _cwd: Path, _env: Any, _timeout: float):
        Path(command[command.index("--out") + 1]).write_text(json.dumps({
            "score": 1, "violations": [], "metrics": {
                "wall_ms": 1, "llm_calls": 1, "tool_calls": 0, "tokens_in": 1, "tokens_out": 1,
                "tokens": 2, "output_chars": 1, "output_lines": 1},
            "rubric_scores": {}, "subscores": {},
            "served_model": "wrong-model", "judge_model": "s1/scripted/default", "excerpt": "",
        }), encoding="utf-8")
        return 0, b"", b"", False

    score = _evaluate(_domain_for(["harness_triage_001"], process_runner=process), candidate).scores[0]
    assert score.score is None
    assert {v.rule_id for v in score.violations} == {"model.mismatch"}


def test_parent_enforces_runner_budget_and_missing_result(candidate: Path) -> None:
    def over_budget(command: Any, _cwd: Path, _env: Any, _timeout: float):
        Path(command[command.index("--out") + 1]).write_text(json.dumps({
            "score": 1, "violations": [], "metrics": {
                "wall_ms": 1, "llm_calls": 1, "tool_calls": 121, "tokens_in": 1, "tokens_out": 1,
                "tokens": 2, "output_chars": 1, "output_lines": 1},
            "rubric_scores": {}, "subscores": {}, "served_model": "fake-model",
            "judge_model": "s1/scripted/default", "excerpt": "",
        }), encoding="utf-8")
        return 0, b"", b"", False

    score = _evaluate(_domain_for(["harness_triage_001"], process_runner=over_budget), candidate).scores[0]
    assert score.score == 0.0
    assert "budget.exceeded" in {v.rule_id for v in score.violations}

    missing = _domain_for(
        ["harness_triage_001"],
        process_runner=lambda *_args: (3, b"ignored", b"ignored", False),
    )
    score = _evaluate(missing, candidate).scores[0]
    assert score.score is None
    assert "eval.missing_result" in {v.rule_id for v in score.violations}


def test_live_tier_requires_judge_endpoint(candidate: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI_S1_LLAMA_URL", raising=False)
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    domain = HarnessDomain(profile="offline", tier="evolve",
                           target_model="openai/target", judge_model="s1/llamacpp/judge",
                           split_map={"evolve": ["harness_triage_001"]})
    with pytest.raises(ValueError, match="CI_S1_LLAMA_URL"):
        _evaluate(domain, candidate)
    monkeypatch.setenv("CI_S1_LLAMA_URL", "http://127.0.0.1:8081")
    missing_judge = HarnessDomain(profile="offline", tier="evolve", target_model="openai/target",
                                  judge_model="", split_map={"evolve": ["harness_triage_001"]})
    with pytest.raises(ValueError, match="CI_LAB_JUDGE_MODEL"):
        _evaluate(missing_judge, candidate)


def test_quality_score_excludes_missing_resource_metrics() -> None:
    rubric = load_rubric(ROOT, "harness_proposal")
    grade = grade_case(
        rubric, deterministic_score=1.0, text="Removed one duplicate and preserved safety.",
        measurements={}, profile="fake", judge_model="s1/scripted/default")
    assert grade.score == 1.0
    assert grade.subscores["resource_score"] == 0.0
    assert all(value == 0.0 for key, value in grade.subscores.items() if key.startswith("resource."))
    assert {v["rule_id"] for v in grade.violations} >= {"metric.missing"}


def test_injection_failures_never_include_excerpt(candidate: Path) -> None:
    def process(command: Any, _cwd: Path, _env: Any, _timeout: float):
        Path(command[command.index("--out") + 1]).write_text(json.dumps({
            "score": 0.5, "violations": [{"rule_id": "injection.obeyed", "severity": "critical",
                                           "detail": "contained"}],
            "metrics": {"wall_ms": 1, "llm_calls": 1, "tool_calls": 0, "tokens_in": 1,
                        "tokens_out": 1, "tokens": 2, "output_chars": 1, "output_lines": 1},
            "rubric_scores": {"critical-containment": 0},
            "subscores": {}, "served_model": "fake-model",
            "judge_model": "s1/scripted/default", "excerpt": "RAW TOOL OUTPUT secret",
        }), encoding="utf-8")
        return 0, b"", b"", False

    domain = _domain_for(["harness_injection_001"], process_runner=process)
    result = _evaluate(domain, candidate)
    (failure,) = domain.failures(result)
    assert failure.excerpt == ""
    assert "injection.obeyed" in failure.rule_ids


def test_assert_wrapper_binds_real_case_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from ci_lab.domain import harness_assert_wrapper as wrapper

    seen: list[str | None] = []

    async def original(test_case: dict[str, Any]) -> str:
        seen.append(current_case_id())
        return "ok"

    module = SimpleNamespace(_run_prompt_test_case=original)
    monkeypatch.setattr(wrapper, "register_judge", lambda: None)
    monkeypatch.setattr(wrapper, "_attach_token", object())
    wrapper.install((module,))
    assert asyncio.run(module._run_prompt_test_case({"test_case_id": "harness_triage_001"})) == "ok"
    assert seen == ["harness_triage_001"]
    assert current_case_id() is None
    assert wrapper.command("evals/assert/harness_triage/eval_config.yaml")[:3] == [
        wrapper.sys.executable, "-m", "ci_lab.domain.harness_assert_wrapper"]


def test_campaign_fake_domain_choice_is_registered(tmp_path: Path) -> None:
    from ci_lab.campaign.cli import load_deps

    harness = load_deps(Profile.FAKE, run_root=tmp_path / "h", ledger_dir=None,
                        repo="example/harness", dry_run_publish=True, domain_name="harness")
    default = load_deps(Profile.FAKE, run_root=tmp_path / "d", ledger_dir=None,
                        repo="example/harness", dry_run_publish=True)
    assert harness.domain.name == "harness"
    assert default.domain.name == "order_support"
