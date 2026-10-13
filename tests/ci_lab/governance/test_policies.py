"""Packaged ACS policies: every rule's allow / deny / transform / escalate path."""

from __future__ import annotations

import asyncio
from importlib import resources
from pathlib import Path

import pytest
import yaml

from ci_lab.governance import adapters
from ci_lab.governance.acs import load_manifest
from ci_lab.governance.adapters import globs_overlap, redact, register_leak_screen
from ci_lab.governance.policies import POLICIES, governance_mode, load_policy


def _eval(name, point, snap, mode="enforce"):
    ctl = load_policy(name, mode=mode)
    return asyncio.run(ctl.runtime.evaluate_intervention_point(point, snap, mode))


@pytest.mark.parametrize("name", POLICIES)
def test_packaged_manifests_validate_with_a_policy_per_point(name):
    text = resources.files("ci_lab.governance.policies").joinpath(f"{name}.acs.yaml").read_text()
    manifest = load_manifest(text)
    assert manifest.points and all(p.policy_id in manifest.policies for p in manifest.points.values())


def test_meta_agent_model_allowlist():
    assert _eval("meta_agents", "pre_model_call", {"model": {"id": "gpt-5.5"}}).verdict.decision == "allow"
    assert _eval("meta_agents", "pre_model_call", {"model": {"id": "gpt-4"}}).verdict.decision == "deny"


def test_output_secret_redaction_transform_applies_only_in_enforce():
    text = "Mail a@b.co, card 4111 1111 1111 1111, ssn 123-45-6789, token=sk-abcdefghijklmnopqrst1"
    log = []
    ctl = load_policy("meta_agents", mode="enforce", decision_log=log.append)
    result = asyncio.run(ctl.guard("output", {"output": {"text": text}}))
    out = result.transformed_policy_target
    assert out.count("[redacted:") == 1 and "a@b.co" in out
    assert log[-1]["decision"] == "transform" and text not in str(log)
    shadow = asyncio.run(load_policy("meta_agents", mode="evaluate_only").guard(
        "output", {"output": {"text": text}}))
    assert shadow.transformed_policy_target is None
    assert _eval("meta_agents", "output", {"output": {"text": "Build 4111 passed"}}).verdict.decision == "allow"


def test_redact_keeps_non_luhn_digits_and_pii_flag():
    assert redact("card 1234 5678 9012 3456")[1] == []
    assert redact("x@y.io", pii=False) == ("x@y.io", [])


@pytest.mark.parametrize(("path", "decision"), [
    ("src/ci_lab/governance/policies/x.yaml", "deny"),
    ("src/ci_lab/rules/engine.py", "deny"),
    ("SRC/ci_lab/guards/a.yaml", "deny"),
    ("./.github/workflows/ci.yml", "deny"),
    ("evals/datasets/heldout.jsonl", "deny"),
    ("evals/x/sealed/r.json", "deny"),
    ("harness/guards/.lkg/a.yaml", "deny"),
    ("runs/r1/lessons/candidates.jsonl", "deny"),
    ("harness/prompts/system.md", "allow"),
])
def test_meta_write_tools_cannot_touch_protected_paths(path, decision):
    snap = {"call": {"name": "write_file", "arguments": {"path": path, "content": ""}}}
    assert _eval("meta_agents", "pre_tool_call", snap).verdict.decision == decision
    snap["call"]["name"] = "read_file"
    assert _eval("meta_agents", "pre_tool_call", snap).verdict.decision == "allow"


def test_meta_kill_switch_blocks_tools():
    snap = {"call": {"name": "read_file", "arguments": {}}, "governance": {"kill_switch": True}}
    assert _eval("meta_agents", "pre_tool_call", snap).verdict.reason == "kill_switch_engaged"


@pytest.mark.parametrize(("name", "args", "decision", "reason"), [
    ("read_file", {"path": "prompts/analyst.md"}, "allow", None),
    ("harness.list_components", {}, "allow", None),
    ("read_file", {"path": "../secrets.txt"}, "deny", "path_outside_root"),
    ("read_file", {"path": "C:\\secrets.txt"}, "deny", "path_outside_root"),
    ("unknown_tool", {}, "deny", "tool_not_allowed"),
    ("run_code", {"code": "import socket\nsocket.create_connection(('example.com', 443))"},
     "deny", "network_import_forbidden"),
    ("run_code", {"code": "open('secrets.txt').read()"}, "deny", "unsafe_call_forbidden"),
    ("run_code", {"code": "import json\nprint(json.dumps({'ok': True}))"}, "allow", None),
])
def test_harness_tool_policy(name, args, decision, reason):
    snap = {"agent": {"name": "CiFailureAnalyst"},
            "call": {"name": name, "arguments": args}}
    verdict = _eval("harness", "pre_tool_call", snap).verdict
    assert (verdict.decision, verdict.reason) == (decision, reason)


def test_harness_policy_rejects_oversized_arguments_and_cross_agent_tools():
    huge = {"agent": {"name": "CiFailureAnalyst"},
            "call": {"name": "read_file", "arguments": {"path": "x", "padding": "x" * 32768}}}
    assert _eval("harness", "pre_tool_call", huge).verdict.reason == "arguments_too_large"
    wrong_agent = {"agent": {"name": "CiAnalyst"},
                   "call": {"name": "write_file", "arguments": {"path": "prompts/x.md"}}}
    assert _eval("harness", "pre_tool_call", wrong_agent).verdict.reason == "tool_not_allowed"


def test_harness_policy_matches_frozen_agents_and_mcp_caps():
    policy = load_manifest(
        resources.files("ci_lab.governance.policies").joinpath("harness.acs.yaml").read_text()
    ).policies["tools"]
    registry = yaml.safe_load(
        resources.files("ci_lab.mcp").joinpath("servers.yaml").read_text()
    )
    caps = registry["code_mode"]
    assert policy["timeout_s_max"] == caps["timeout_s_max"]
    assert policy["max_output_chars_max"] == caps["max_output_chars_max"]
    assert policy["allowed_imports"] == caps["allowed_imports"]
    for path in Path("harness/agents").iterdir():
        if path.suffix != ".yaml":
            continue
        spec = yaml.safe_load(path.read_text())
        assert {tool["name"] for tool in spec["tools"]} <= set(
            policy["agent_tools"][spec["name"]])


def test_model_allowlist(monkeypatch):
    monkeypatch.delenv("CI_ALLOWED_MODELS", raising=False)
    monkeypatch.delenv("CI_META_MODEL", raising=False)
    assert _eval("meta_agents", "pre_model_call", {"model": {"id": "claude-sonnet-5"}}).verdict.decision == "allow"
    assert _eval("meta_agents", "pre_model_call", {"model": {}}).verdict.reason == "model_unspecified"
    v = _eval("meta_agents", "pre_model_call", {"model": {"id": "rogue-1"}}).verdict
    assert (v.decision, v.reason) == ("deny", "model_not_allowed")
    monkeypatch.setenv("CI_ALLOWED_MODELS", "rogue-1")
    assert _eval("meta_agents", "pre_model_call", {"model": {"id": "rogue-1"}}).verdict.decision == "allow"


def test_meta_allowlist_mirrors_meta_specs_manifest():
    meta = yaml.safe_load(resources.files("ci_lab.meta.specs").joinpath("manifest.yaml").read_text())
    acs = load_manifest(resources.files("ci_lab.governance.policies").joinpath("meta_agents.acs.yaml").read_text())
    assert acs.policies["models"]["allowed_models"] == meta["allowed_models"]


def test_student_rubric_leak_screen_and_secret_redaction():
    register_leak_screen(lambda text: ["canary"] if "CANARY-1" in text else [])
    try:
        leak = {"output": {"text": "see CANARY-1"}, "agent": {"role": "student"}}
        assert _eval("meta_agents", "output", leak).verdict.reason == "rubric_leak"
        leak["agent"]["role"] = "author"
        assert _eval("meta_agents", "output", leak).verdict.decision == "allow"
    finally:
        register_leak_screen(None)
    r = _eval("meta_agents", "output", {"output": {"text": "a@b.co ghp_" + "a" * 30}})
    assert r.transformed_policy_target == "a@b.co [redacted:secret]"


def _launch(**campaign):
    gov = {"kill_switch": campaign.pop("kill", False)}
    base = {"dry_run": True, "publish": False, "budget_exhausted": False,
            "arms": [{"id": "a", "edit_scope": ["harness/**"]}]}
    return {"campaign": {**base, **campaign}, "governance": gov}


@pytest.mark.parametrize(("snap", "decision", "reason"), [
    (_launch(), "allow", None),
    (_launch(kill=True), "deny", "kill_switch_engaged"),
    (_launch(budget_exhausted=True), "deny", "budget_exhausted"),
    (_launch(arms=[{"id": "b", "edit_scope": ["src/**"]}]), "deny", "protected_scope"),
    (_launch(arms=[{"id": "b", "edit_scope": ["src/ci_lab/guards/x.yaml"]}]), "deny", "protected_scope"),
    (_launch(arms=[{"id": "b", "edit_scope": ["docs/sealed/*.md"]}]), "deny", "protected_scope"),
    (_launch(dry_run=False, publish=True), "deny", "publish_requires_approval"),
    (_launch(dry_run=True, publish=True), "allow", None),
])
def test_campaign_launch_gate(snap, decision, reason):
    v = _eval("campaign", "agent_startup", snap).verdict
    assert (v.decision, v.reason) == (decision, reason)
    assert v.liftable == (reason == "publish_requires_approval")


@pytest.mark.parametrize(("scope", "protected", "hit"), [
    ("harness/**", "src/ci_lab/**", False),
    ("src/**", "src/ci_lab/rules/**", True),
    ("**", ".github/workflows/**", True),
    ("harness/**", "**/sealed/**", False),
    ("evals/sealed/*.json", "**/sealed/**", True),
])
def test_globs_overlap(scope, protected, hit):
    assert globs_overlap(scope, protected) is hit


def test_governance_mode_env():
    assert governance_mode({}) == "enforce"
    assert governance_mode({"CI_GOVERNANCE_MODE": "evaluate_only"}) == "evaluate_only"
    with pytest.raises(RuntimeError):
        governance_mode({"CI_GOVERNANCE_MODE": "off"})
    with pytest.raises(KeyError):
        load_policy("nope", mode="enforce")


def test_adapters_are_registered_for_every_custom_policy():
    for name in POLICIES:
        manifest = load_policy(name, mode="enforce").runtime.manifest
        assert {p["adapter"] for p in manifest.policies.values()} <= set(adapters.DISPATCHER)
