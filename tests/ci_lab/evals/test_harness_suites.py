from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from assert_ai.runner import _load_context

from ci_lab.bus.cli import _load_rubrics
from ci_lab.metrics.rubric import validate_metric_check
from ci_lab.taskgraph.model import load_graph
from ci_lab.taskgraph.validate import validate_graph, validate_rubric

ROOT = Path(__file__).resolve().parents[3]
ASSERT_ROOT = ROOT / "evals" / "assert"
RUBRIC_ROOT = ROOT / "evals" / "rubrics" / "harness"
DATASET = ROOT / "evals" / "datasets" / "harness.yaml"
SUITES = (
    "harness_triage",
    "harness_proposal",
    "harness_taskgraph",
    "harness_tool_use",
    "harness_injection",
)
RUNTIME_METRICS = {"wall_ms", "llm_calls", "tokens"}
OUTPUT_FORBIDDEN = ("wall_ms", "llm_calls", "tokens_in", "tokens_out", "complexity_delta", "edit_lines")
CASE_CAPS = {"max_llm_calls", "max_tool_calls", "max_tokens", "timeout_s"}


def _rows(suite: str) -> list[dict[str, Any]]:
    path = ASSERT_ROOT / suite / "test_set.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _taxonomy(suite: str) -> dict[str, Any]:
    return json.loads((ASSERT_ROOT / suite / "taxonomy.json").read_text(encoding="utf-8"))


def _rubric(suite: str):
    (rubric,) = _load_rubrics([RUBRIC_ROOT / f"{suite}.yaml"])
    return rubric


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def test_real_loaders_parse_all_harness_suites(tmp_path: Path) -> None:
    for suite in SUITES:
        cfg = ASSERT_ROOT / suite / "eval_config.yaml"
        ctx = _load_context(config=str(cfg), overrides=[f"artifacts_root={tmp_path / suite}"])
        assert dict(ctx["stages"])["test_set"]["taxonomy_path"] == "taxonomy.json"
        taxonomy = _taxonomy(suite)
        assert taxonomy["behavior"]["name"] == suite
        rubric = _rubric(suite)
        assert validate_rubric(rubric) == []

    graph = load_graph(ROOT / "evals" / "fixtures" / "harness" / "taskgraph" / "graph.yaml")
    assert validate_graph(graph) == []


def test_cases_are_distinct_scripted_and_do_not_supply_measurements() -> None:
    ids: list[str] = []
    for suite in SUITES:
        rows = _rows(suite)
        assert len(rows) >= 10
        assert {row["behavior"] for row in rows} == {suite}
        assert all(row.get("seed") and row.get("expected") for row in rows)
        ids += [row["test_case_id"] for row in rows]
        for row in rows:
            assert not (CASE_CAPS & row.keys()), row["test_case_id"]
            script = row.get("fake_script")
            assert isinstance(script, list) and script
            assert "measurements" not in row
            for step in script:
                assert set(step) in ({"text"}, {"tool_calls"})
                if "tool_calls" in step:
                    assert step["tool_calls"] and all(
                        isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)
                        for call in step["tool_calls"]
                    )
                candidate_material = "\n".join(_strings(step))
                assert not any(field in candidate_material for field in OUTPUT_FORBIDDEN)
    assert len(ids) == len(set(ids)) == 50


def test_dataset_tiers_cover_once_without_leaks() -> None:
    doc = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    assert doc["format"] == "ci_lab.harness.dataset.v1"
    assert doc["budget_source"] == {
        "manifest": "src/ci_lab/harness_tree/manifest.yaml",
        "key": "caps.eval",
    }
    tiers = doc["tiers"]
    assert tiers["ci"] == {"profile": "fake", "k": 1, "selection": "all", "judge": "fake"}
    assert tiers["evolve"]["k"] == 1 and tiers["evolve"]["max_cases_per_suite"] == 3
    assert tiers["confirm"]["k"] == 2 and tiers["confirm"]["finalists_only"] is True
    assert doc["splits"]["aa"] == {"alias": "evolve"}
    assert doc["critical_suites"] == ["harness_injection"]

    all_ids = {row["test_case_id"] for suite in SUITES for row in _rows(suite)}
    assigned = {name: set(doc["splits"][name]) for name in ("evolve", "heldout", "ood")}
    assert not (assigned["evolve"] & assigned["heldout"])
    assert not (assigned["evolve"] & assigned["ood"])
    assert not (assigned["heldout"] & assigned["ood"])
    assert set().union(*assigned.values()) == all_ids
    assert sum(map(len, assigned.values())) == len(all_ids)
    evolve_counts = Counter(case_id.rsplit("_", 1)[0] for case_id in assigned["evolve"])
    assert set(evolve_counts) == set(SUITES)
    assert all(count <= 3 for count in evolve_counts.values())


def test_assert_configs_use_supported_target_and_judge_pins() -> None:
    for suite in SUITES:
        cfg = yaml.safe_load((ASSERT_ROOT / suite / "eval_config.yaml").read_text(encoding="utf-8"))
        assert cfg["default_model"]["name"] == "openai/local"
        assert cfg["pipeline"]["judge"]["model"]["name"] == "s1/llamacpp/qwen3.5-4b"
        assert not ({"target_model", "judge_model", "pins"} & cfg.keys())


def test_rubrics_separate_quality_from_measured_resources() -> None:
    for suite in SUITES:
        rubric = _rubric(suite)
        quality = rubric.quality()
        resources = rubric.resources()
        assert any(c.measure in {"assert", "deterministic"} for c in quality)
        assert any(c.measure == "s1" for c in quality)
        assert resources and all(c.role == "resource" for c in resources)
        assert all(c.role == "quality" for c in quality)
        metric_resources = [c for c in resources if c.measure == "metric"]
        assert any(c.check["metric"] in RUNTIME_METRICS for c in metric_resources)
        for criterion in metric_resources:
            validate_metric_check(criterion.check)
        if suite in {"harness_proposal", "harness_taskgraph"}:
            assert any(c.check["metric"] == "complexity_delta" for c in metric_resources)
        else:
            efficiency = next(c for c in resources if c.id == "resource-efficiency")
            assert efficiency.check["metric"] in {"llm_calls", "tokens", "tool_calls"}
            assert all(c.check["metric"] != "complexity_delta" for c in metric_resources)

        scores = {c.id: 1.0 for c in quality} | {c.id: 0.0 for c in resources}
        assert rubric.quality_score(scores) == 1.0
        subscores = rubric.resource_subscores(scores)
        assert subscores["resource_score"] == 0.0
        assert set(subscores) == {*(f"resource.{c.id}" for c in resources), "resource_score"}


def test_suite_specific_contracts() -> None:
    assert all({"component", "reason_code"} <= row["expected"].keys() for row in _rows("harness_triage"))
    assert all(row["expected"]["edit_count"] == 1 for row in _rows("harness_proposal"))
    assert all(row["graph"].endswith("taskgraph/graph.yaml") for row in _rows("harness_taskgraph"))
    assert all(row["expected"]["tool_sequence"] for row in _rows("harness_tool_use"))

    injection = _rows("harness_injection")
    assert {row["dimensions"]["injection_source"] for row in injection} == {"digest", "tool_output", "skill_text"}
    assert all(row["safety"]["classification"] == "critical" for row in injection)
    assert all(row["expected"]["allowed_writes"] == [] and row["expected"]["forbidden_tools"] for row in injection)
