"""L1 contract additions: TaskScore runtime fields/subscores, EvalResult.surface, component owners, and
round-trips through every TaskScore/EvalResult codec (old records without the new fields still load)."""

from __future__ import annotations

import json

import pytest

from ci_lab.cache.evalcache import EvalCache
from ci_lab.campaign import records
from ci_lab.contracts import (
    COMPONENT_OWNERS,
    COMPONENTS,
    STRATEGIES,
    TEXT_COMPONENTS,
    EvalResult,
    EvaluatorPin,
    TaskScore,
    Violation,
    strategy_may_edit,
)
from ci_lab.rrsi import codec

TREE = "a" * 40
PIN = EvaluatorPin("etree", "judge-m", "prov", ("served-j",))
FULL = TaskScore("c01", 1, "core", 0.75, (Violation("r.x", "major", "d"),), tokens_in=10, tokens_out=5,
                 served_model="m", wall_ms=12.5, llm_calls=3, tool_calls=2,
                 subscores={"resource.speed": 0.4, "resource_score": 0.4})
RESULT = EvalResult("htree", "evolve", PIN, [FULL, TaskScore("c02", 0, "core", None)],
                    surface={"complexity": 42.0, "files": 7.0, "component.prompt.complexity": 1.5})


def test_task_score_defaults_and_hashable() -> None:
    s = TaskScore("c", 0, "s", 1.0)
    assert (s.wall_ms, s.llm_calls, s.tool_calls, s.subscores) == (0.0, 0, 0, {})
    assert hash(FULL) == hash(FULL.__class__(**{**FULL.__dict__, "subscores": {}}))
    assert EvalResult("h", "evolve", PIN).surface == {}


def test_components_and_owners() -> None:
    assert COMPONENTS[-4:] == ("agent", "loop", "workflow", "mcp")
    assert TEXT_COMPONENTS == tuple(c for c in COMPONENTS if c != "guard")
    assert set(COMPONENT_OWNERS) == set(COMPONENTS)
    assert COMPONENT_OWNERS["prompt"] == "gepa" and COMPONENT_OWNERS["skill"] == "skillopt"
    assert COMPONENT_OWNERS["guard"] == "guard"
    for c in ("agent", "loop", "workflow", "mcp", "client_tool", "config", "context_mgmt", "memory"):
        assert COMPONENT_OWNERS[c] == "agl"
    assert "agl" not in STRATEGIES


@pytest.mark.parametrize(("strategy", "component", "ok"), [
    ("agent", "prompt", True), ("agent", "workflow", True), ("agent", "guard", False),
    ("guard", "guard", True), ("guard", "prompt", False),
    ("gepa", "prompt", True), ("gepa", "skill", False),
    ("skillopt", "skill", True), ("skillopt", "prompt", False),
    ("agl", "loop", True), ("agl", "memory", True), ("agl", "prompt", False), ("agl", "guard", False),
    ("nope", "prompt", False), ("gepa", "unknown", False)])
def test_strategy_may_edit(strategy: str, component: str, ok: bool) -> None:
    assert strategy_may_edit(strategy, component) is ok


def test_rrsi_codec_round_trip() -> None:
    d = json.loads(json.dumps(codec.eval_to_dict(RESULT)))
    assert codec.eval_from_dict(d) == RESULT
    assert codec.task_score_from_dict(codec.task_score_to_dict(FULL)) == FULL


def test_records_round_trip() -> None:
    d = json.loads(json.dumps(records.eval_to_dict(RESULT)))
    assert records.eval_from_dict(d) == RESULT


def test_eval_cache_round_trip(tmp_path) -> None:
    cache = EvalCache(tmp_path)
    cache.put(PIN, TREE, "evolve", FULL)
    assert cache.get(PIN, TREE, "evolve", "c01", 1) == FULL


def _legacy(d: dict) -> dict:
    d = dict(d)
    d.pop("surface", None)
    d["scores"] = [{k: v for k, v in s.items() if k not in ("wall_ms", "llm_calls", "tool_calls", "subscores")}
                   for s in d["scores"]]
    return d


@pytest.mark.parametrize("mod", [codec, records])
def test_legacy_json_loads(mod) -> None:
    old = _legacy(json.loads(json.dumps(mod.eval_to_dict(RESULT))))
    got = mod.eval_from_dict(old)
    assert got.surface == {}
    s = got.scores[0]
    assert (s.score, s.tokens_in, s.tokens_out, s.served_model) == (0.75, 10, 5, "m")
    assert (s.wall_ms, s.llm_calls, s.tool_calls, s.subscores) == (0.0, 0, 0, {})
