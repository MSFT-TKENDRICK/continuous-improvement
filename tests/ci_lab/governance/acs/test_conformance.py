"""Upstream ACS conformance corpus run against the pure-Python runtime.

Ports the fixtures of upstream ``tests/conformance/run_python.py`` (FixtureAnnotator,
FixturePolicy, QueueRuntime) onto :mod:`ci_lab.governance.acs`; the corpus itself is vendored
unmodified under ``../acs_conformance`` (see NOTICE there).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from ci_lab.governance.acs import (
    AcsError,
    AcsRuntime,
    AgentControl,
    AgentControlBlocked,
    ApprovalResolution,
    Decision,
    InterventionPointResult,
    Verdict,
    action_identity,
    reserved_reasons,
)
from ci_lab.governance.acs.canonical import DEFAULT_LIMITS

CORPUS = Path(__file__).parents[1] / "acs_conformance"
CASES = [
    json.loads(p.read_text(encoding="utf-8"))
    for p in sorted((CORPUS / "cases").glob("*.json"))
]
PARITY = json.loads((CORPUS / "fail_closed_error_parity.json").read_text("utf-8"))


class FixtureAnnotator:
    def __init__(self, case: dict) -> None:
        self.case, self.seen = case, []

    def dispatch(self, name: str, config: Any, preliminary_policy_input: Any) -> Any:
        self.seen.append(name)
        behavior = self.case.get("annotator_behavior")
        if behavior == "timeout":
            raise TimeoutError(name)
        if behavior == "error":
            raise RuntimeError(name)
        return self.case.get("annotator_outputs", {}).get(name, {"ok": True})


class FixturePolicy:
    def __init__(self, case: dict) -> None:
        self.case = case

    def evaluate(self, invocation: dict) -> Any:
        if self.case.get("policy_behavior") == "error":
            raise RuntimeError("policy failed")
        return self.case["policy_response"]


class QueueRuntime:
    """Always returns a liftable deny, as upstream's approval-mismatch fixture does."""

    async def evaluate_intervention_point(
        self, point, snapshot, mode="enforce", tool=None
    ):
        policy_input = {"intervention_point": point, "snapshot": dict(snapshot)}
        identity = action_identity(policy_input)
        return InterventionPointResult(
            Verdict(Decision.DENY, reason="human_review", approval={}),
            policy_input=policy_input,
            input_identity=identity,
            enforced_identity=identity,
        )


def _runtime(case: dict) -> tuple[AcsRuntime, FixtureAnnotator]:
    annotator = FixtureAnnotator(case)
    limits = replace(DEFAULT_LIMITS, **case.get("limits", {}))
    rt = AcsRuntime(
        case["manifest_yaml"],
        dispatcher=FixturePolicy(case),
        annotator=annotator,
        limits=limits,
    )
    return rt, annotator


def _evaluate(case: dict, rt: Any) -> InterventionPointResult:
    return asyncio.run(
        rt.evaluate_intervention_point(
            case["intervention_point"], case["snapshot"], case.get("mode", "enforce")
        )
    )


def _check(case: dict, result: InterventionPointResult, seen: list[str]) -> None:
    expected, verdict, pi = case["expected"], result.verdict, result.policy_input
    assert str(verdict.decision) == expected["decision"]
    if "reason" in expected:
        assert verdict.reason == expected["reason"]
    if "transformed_policy_target" in expected:
        assert result.transformed_policy_target == expected["transformed_policy_target"]
    if "policy_target" in expected:
        assert pi["policy_target"]["value"] == expected["policy_target"]
    if "annotations" in expected:
        assert pi["annotations"] == expected["annotations"]
    if "annotator_order" in expected:
        assert seen == expected["annotator_order"]
    if "tool_name" in expected:
        assert pi["tool"]["name"] == expected["tool_name"]
    present = {
        "action_identity": (result.action_identity or "").startswith("sha256:"),
        "warnings": bool(verdict.warnings),
        "approval": verdict.approval is not None,
    }
    for key, ok in present.items():
        if expected.get(key) == "present":
            assert ok, key


def _approval_mismatch(rt: Any) -> str | None:
    async def resolver(point, result):
        approved = result.action_identity
        result.policy_input["snapshot"]["input"] = "mutated"
        return ApprovalResolution.allow(approved)

    try:
        asyncio.run(AgentControl(rt, approval_resolver=resolver).run("hi", lambda v: v))
    except AgentControlBlocked as exc:
        return exc.result.verdict.reason
    return None


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_conformance_case(case):
    assert case["sdk_support"]["python"] in {"required", "optional", "skip"}
    if case["operation"] == "approval_action_mismatch":
        for rt in (QueueRuntime(), _runtime(case)[0]):
            assert _approval_mismatch(rt) == case["expected"]["reason"]
        return
    assert case["operation"] == "evaluate"
    rt, annotator = _runtime(case)
    _check(case, _evaluate(case, rt), annotator.seen)


@pytest.mark.parametrize(
    "case", PARITY["cases"], ids=[c["id"] for c in PARITY["cases"]]
)
def test_fail_closed_error_parity(case):
    assert case["expected_reason"] in reserved_reasons()
    if case["operation"] == "build":
        with pytest.raises(AcsError) as info:
            _runtime(case)
        assert info.value.reason == case["expected_reason"]
        return
    result = _evaluate(case, _runtime(case)[0])
    assert result.verdict.decision is Decision.DENY
    assert result.verdict.reason == case["expected_reason"]
    assert result.transformed_policy_target is None


def test_corpus_is_well_formed():
    validator = Draft202012Validator(
        json.loads((CORPUS / "cases.schema.json").read_text("utf-8"))
    )
    for case in CASES:
        validator.validate(case)
    assert len(CASES) == 25
    assert set(PARITY["reserved_reasons"]) <= reserved_reasons()


def test_native_runtime_parity():
    """Cross-check decisions/reasons with the upstream native runtime when it is installed."""
    pytest.importorskip("agent_control_specification._native")
    from agent_control_specification import AgentControl as NativeControl

    for case in [c for c in CASES if c["operation"] == "evaluate"]:
        native = NativeControl.from_native(
            case["manifest_yaml"], FixtureAnnotator(case), FixturePolicy(case)
        )
        theirs = _evaluate(case, native).verdict
        ours = _evaluate(case, _runtime(case)[0]).verdict
        assert (
            str(getattr(theirs.decision, "value", theirs.decision)),
            theirs.reason,
        ) == (
            str(ours.decision),
            ours.reason,
        ), case["id"]
