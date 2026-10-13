"""Pinned public API (FLEET.md Wave 3) — M15–M17 code against these names/signatures."""

import inspect
from dataclasses import FrozenInstanceError

import pytest

import ci_lab.rules as R
from ci_lab.rulespec import GuardView, RuleSpec


def test_public_api_names_and_signatures():
    for name in ("Bundle", "RuleLoadError", "load_bundle", "load_with_lkg", "default_templates", "compute_flags",
                 "Match", "evaluate", "evaluate_trajectory", "redact", "write_lkg", "build_bundle"):
        assert hasattr(R, name), name
    assert list(inspect.signature(R.load_bundle).parameters) == ["rule_paths", "extractor_paths", "templates"]
    assert list(inspect.signature(R.load_with_lkg).parameters)[:3] == ["guards_dir", "lock", "extractor_paths"]
    assert list(inspect.signature(R.evaluate).parameters) == ["bundle", "view", "on"]
    assert list(inspect.signature(R.redact).parameters) == ["text", "matches", "bundle"]
    assert {"rule", "message", "fix", "see", "step_index"} <= set(R.Match.__dataclass_fields__)
    assert {"rules", "extractors", "templates", "digest"} <= set(R.Bundle.__dataclass_fields__)
    assert R.RuleLoadError(["a", "b"]).problems == ["a", "b"]


def test_seed_bundle_loads_and_is_immutable(seeds, tb):
    assert [r.rung for r in seeds.rules] == sorted(r.rung for r in seeds.rules)
    assert seeds.digest.startswith("sha256:") and seeds.config_digest.startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        seeds.digest = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        seeds.templates["x"] = None  # type: ignore[index]
    pending = tb.call("issue_refund", order_id="A1", amount=5)
    got = R.evaluate(seeds, GuardView(steps=(), pending=pending), on="tool_call")
    assert [m.rule.id for m in got] == ["refund.amount_exceeds_eligible", "refund.ineligible_order",
                                        "refund.requires_verified_identity"]
    m = got[-1]
    assert isinstance(m.rule, RuleSpec) and m.step_index == pending.i
    assert "verify_identity" in m.fix and "{" not in m.message + m.fix
