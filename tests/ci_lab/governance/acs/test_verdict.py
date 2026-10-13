"""ACS verdict normalization (spec §13, §14, §16)."""

from __future__ import annotations

import pytest

from ci_lab.governance.acs import (
    AcsError,
    Decision,
    Limits,
    action_identity,
    denied,
    normalize_verdict,
)

INVALID = "runtime_error:policy_output_invalid"
T = "transform"


def test_allow_deny_transform_pass_through():
    verdict = normalize_verdict({"decision": "deny", "reason": "r", "message": "m"})
    assert (verdict.decision, verdict.reason, verdict.message) == (
        Decision.DENY,
        "r",
        "m",
    )
    assert verdict.approval is None and not verdict.liftable
    body = {"path": "$target.text", "value": "x"}
    verdict = normalize_verdict(
        {"decision": T, "transform": body, "result_labels": None}
    )
    assert verdict.decision is Decision.TRANSFORM and verdict.transform == body
    assert verdict.result_labels == ()
    evidence = {"artefact": "sha256:00", "verification_pointers": {"k": "https://x"}}
    assert (
        normalize_verdict({"decision": "allow", "evidence": evidence}).evidence
        == evidence
    )


def test_warn_normalizes_to_allow_with_warning():
    verdict = normalize_verdict(
        {
            "decision": "warn",
            "reason": "w",
            "message": "m",
            "result_labels": ["internal"],
        }
    )
    assert (verdict.decision, verdict.reason) == (Decision.ALLOW, "w")
    assert verdict.warnings == ({"reason": "w", "message": "m"},)
    assert verdict.result_labels == ("internal",)


def test_escalate_normalizes_to_liftable_deny():
    verdict = normalize_verdict({"decision": "escalate", "reason": "review"})
    assert (verdict.decision, verdict.reason, verdict.approval) == (
        Decision.DENY,
        "review",
        {},
    )
    assert verdict.liftable
    explicit = normalize_verdict({"decision": "deny", "approval": {"queue": "ops"}})
    assert explicit.liftable and explicit.approval == {"queue": "ops"}


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("nope", INVALID),
        (None, INVALID),
        ({"reason": "missing decision"}, INVALID),
        ({"decision": "block"}, INVALID),
        ({"decision": "deny", "reason": "runtime_error:x"}, INVALID),
        ({"decision": "deny", "reason": "host_error:x"}, INVALID),
        ({"decision": "deny", "reason": 3}, INVALID),
        ({"decision": "allow", "effects": []}, INVALID),
        ({"decision": "allow", "transform": {"path": "$target", "value": 1}}, INVALID),
        ({"decision": T}, INVALID),
        ({"decision": "allow", "approval": {}}, INVALID),
        ({"decision": "allow", "evidence": []}, INVALID),
        ({"decision": "allow", "result_labels": [1]}, INVALID),
        ({"decision": "allow", "x": float("nan")}, INVALID),
        ({"decision": T, "transform": {"path": "$target.a"}}, "runtime_error:transform_invalid"),
        ({"decision": T, "transform": {"path": "$target..a", "value": 1}}, "runtime_error:transform_invalid"),
        ({"decision": T, "transform": {"path": 4, "value": 1}}, "runtime_error:transform_invalid"),
        ({"decision": T, "transform": {"path": "$snap.a", "value": 1}}, "runtime_error:transform_target_forbidden"),
        ({"decision": T, "transform": {"path": "$pi.tool", "value": 1}}, "runtime_error:transform_target_forbidden"),
        ({"decision": "allow", "message": "x" * 80}, "runtime_error:resource_limit_exceeded"),
    ],
)  # fmt: skip
def test_unnormalizable_output_fails_closed(output, reason):
    with pytest.raises(AcsError) as exc:
        normalize_verdict(output, limits=Limits(max_policy_output_bytes=80))
    assert exc.value.reason == reason


def test_denied_carries_policy_input_identity():
    assert denied("runtime_error:tool_unknown").input_identity is None
    pi = {"intervention_point": "input"}
    result = denied("runtime_error:tool_unknown", pi)
    assert result.verdict.decision is Decision.DENY and result.policy_input is pi
    assert result.input_identity == result.enforced_identity == action_identity(pi)
    assert result.action_identity == result.enforced_identity
    assert result.transformed_policy_target is None
