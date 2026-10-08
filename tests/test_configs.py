"""Static checks that every ASSERT suite config, taxonomy and judge contract is valid."""

import json
from pathlib import Path

import pytest
from assert_ai.core.judge import build_judge_contract
from assert_ai.runner import _load_context
from assert_ai.stages.judge import JUDGE_SYSTEM_PROMPT

from order_support import data, replay

ASSERT_DIR = replay.REPO_ROOT / "evals" / "assert"
CONFIGS = sorted(ASSERT_DIR.glob("*/eval_config.yaml"))
LIVE_SUITES = [p for p in CONFIGS if p.parent.name != "judge_replay"]


def _ctx(path: Path, tmp_path: Path):
    return _load_context(config=str(path), overrides=[f"artifacts_root={tmp_path}"])


def test_expected_suites_present():
    assert {p.parent.name for p in CONFIGS} == {
        "judge_replay", "refund_authorization", "identity_verification",
        "indirect_prompt_injection", "grounding", "tool_selection"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_config_loads_and_judge_contract_builds(path, tmp_path):
    ctx = _ctx(path, tmp_path)
    taxonomy = json.loads((path.parent / "taxonomy.json").read_text(encoding="utf-8"))
    assert ctx["behavior_name"] == taxonomy["behavior"]["name"]
    assert ctx["suite_id"] == f"order_support_{path.parent.name}"
    judge = ctx["evaluation"].judge
    contract = build_judge_contract(template=JUDGE_SYSTEM_PROMPT, policy_raw=taxonomy,
                                    judge_dimensions=judge.dimensions,
                                    disabled_dimensions=judge.disabled_dimensions)
    for category in taxonomy["behavior_categories"]:
        assert category["name"] in contract["system_prompt"]
    assert "policy_violation" in contract["score_keys"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_taxonomy_schema(path):
    taxonomy = json.loads((path.parent / "taxonomy.json").read_text(encoding="utf-8"))
    cats = taxonomy["behavior_categories"]
    names = [c["name"] for c in cats]
    assert len(names) == len(set(names)) and len(cats) >= 5
    assert all(isinstance(c["permissible"], bool) and c["definition"] and c["examples"] for c in cats)
    assert any(c["permissible"] for c in cats) and not all(c["permissible"] for c in cats)
    # Taxonomies are generic policy, not answer keys for the labelled replay cases.
    assert not any(f"c{i:02d}" in json.dumps(taxonomy) for i in range(1, 31))


# The policy routes only a verified customer's eligible, over-limit refund to a human, so a
# permissible ("required help") node must never reward escalation for anything else.
ESCALATION_CHECKED = [p for p in LIVE_SUITES if p.parent.name == "indirect_prompt_injection"]


@pytest.mark.parametrize("path", ESCALATION_CHECKED, ids=lambda p: p.parent.name)
def test_permissible_nodes_reward_only_policy_routed_escalation(path):
    taxonomy = json.loads((path.parent / "taxonomy.json").read_text(encoding="utf-8"))
    for category in taxonomy["behavior_categories"]:
        text = json.dumps(category).lower()
        if category["permissible"] and "escalat" in text:
            assert "over-limit" in text or "exceeds the refund limit" in text, category["name"]


@pytest.mark.parametrize("path", LIVE_SUITES, ids=lambda p: p.parent.name)
def test_live_suite_targets_agent_and_context_matches_fixtures(path, tmp_path):
    ctx = _ctx(path, tmp_path)
    stages = dict(ctx["stages"])
    assert list(stages) == ["test_set", "inference", "judge"]
    target = stages["inference"]["target"]
    assert target["callable"] == "order_support.agent:chat"
    assert target["trace"]["backend"] == "otel"
    context = ctx["context"]
    assert data.fixture_catalog().strip() in context
    assert data.load_policy().strip() in context
    assert data.TODAY.isoformat() in context
    for dim in stages["test_set"]["stratify"]["dimensions"]:
        assert len(dim["levels"]) >= 2  # explicit levels: no stratification LLM call


def test_replay_config_is_judge_only(tmp_path):
    stages = dict(_ctx(replay.REPLAY_DIR / "eval_config.yaml", tmp_path)["stages"])
    assert list(stages) == ["judge"]
    assert stages["judge"]["inference_set_path"] == "inference_set.jsonl"


def test_agent_policy_is_dataset_policy():
    from order_support import agent

    assert agent.SYSTEM_PROMPT.startswith(data.load_policy())
