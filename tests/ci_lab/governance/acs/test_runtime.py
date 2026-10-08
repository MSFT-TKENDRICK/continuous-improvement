"""ACS runtime evaluation order, normalization, transforms, annotators and limits (spec §5-§16)."""

from __future__ import annotations

import asyncio

import pytest
import yaml

from ci_lab.governance.acs import AcsRuntime, Decision, Limits, action_identity
from ci_lab.governance.acs.canonical import DEFAULT_LIMITS

MANIFEST = """\
agent_control_specification_version: 0.4.0-alpha.1
policies:
  p: {type: rego, query: data.acs.verdict}
  c: {type: custom, adapter: fixture}
intervention_points:
  input:
    policy: {id: p}
    policy_target: $snap.input
    annotations:
      beta: {from: $target.text}
      alpha: {from: $snap.input}
  agent_startup: {policy: {id: c}, policy_target: $snap}
  pre_tool_call:
    tool_name_from: $snap.tool_call.name
    policy: {id: p}
    policy_target: $snap.tool_call.args
    policy_target_kind: tool_args
  post_tool_call: {policy: {id: p}, policy_target: $snap.result}
annotators:
  alpha: {type: classifier}
  beta: {type: llm, model: m}
tools:
  search: {clearance: internal}
"""
SNAP = {"input": {"text": "card 1234", "meta": {"n": 1}}}


class Annotator:
    def __init__(self, outputs: dict | None = None, error: BaseException | None = None):
        self.outputs, self.error, self.calls = outputs or {}, error, []

    def dispatch(self, name, config, preliminary):
        self.calls.append((name, config, preliminary))
        preliminary["snapshot"]["input"] = "mutated"
        if self.error is not None:
            raise self.error
        return self.outputs.get(name, {"ok": True})


def _run(
    response,
    point="input",
    snap=SNAP,
    mode="enforce",
    annotator=None,
    limits=DEFAULT_LIMITS,
    tool=None,
    dispatcher=None,
):
    invocations = []

    async def policy(invocation):
        invocations.append(invocation)
        if isinstance(response, Exception):
            raise response
        return response

    rt = AcsRuntime(
        MANIFEST,
        dispatcher=dispatcher or policy,
        annotator=annotator or Annotator(),
        limits=limits,
    )
    result = asyncio.run(rt.evaluate_intervention_point(point, snap, mode, tool))
    return result, invocations


def test_allow_builds_the_five_member_policy_input():
    ann = Annotator({"beta": {"label": "pii"}})
    result, [inv] = _run({"decision": "allow"}, annotator=ann)
    pi = result.policy_input
    assert (
        result.verdict.decision is Decision.ALLOW and result.verdict.result_labels == ()
    )
    assert set(pi) == {
        "intervention_point",
        "policy_target",
        "snapshot",
        "annotations",
        "tool",
    }
    assert pi["policy_target"] == {
        "kind": None,
        "path": "$snap.input",
        "value": SNAP["input"],
    }
    assert pi["annotations"] == {"alpha": {"ok": True}, "beta": {"label": "pii"}}
    assert [c[0] for c in ann.calls] == ["alpha", "beta"]
    assert ann.calls[1][1] == {
        "type": "llm",
        "model": "m",
        "from": "$target.text",
        "value": "card 1234",
    }
    assert ann.calls[1][2]["annotations"] == {} and pi["snapshot"] == SNAP
    assert (
        inv["input"] == pi and inv["policy_id"] == "p" and inv["binding"] == {"id": "p"}
    )
    assert result.input_identity == result.enforced_identity == action_identity(pi)


@pytest.mark.parametrize("mode", ["enforce", "evaluate_only"])
def test_transform_confined_to_target_and_applied_only_in_enforce(mode):
    out = {
        "decision": "transform",
        "reason": "pii",
        "transform": {"path": "$target.text", "value": "card [X]"},
    }
    result, _ = _run(out, mode=mode)
    assert (
        result.verdict.decision is Decision.TRANSFORM
        and SNAP["input"]["text"] == "card 1234"
    )
    if mode == "enforce":
        assert result.transformed_policy_target == {
            "text": "card [X]",
            "meta": {"n": 1},
        }
        assert result.enforced_identity != result.input_identity
    else:
        assert result.transformed_policy_target is None
        assert result.enforced_identity == result.input_identity


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("nope", "runtime_error:policy_output_invalid"),
        ({"decision": "transform", "transform": {"path": "$snap.input", "value": 1}}, "runtime_error:transform_target_forbidden"),
        ({"decision": "transform", "transform": {"path": "$target.no.such", "value": 1}}, "host_error:transform_invalid"),
        ({"decision": "transform", "transform": {"path": "$target[0]", "value": 1}}, "host_error:transform_invalid"),
        ({"decision": "allow", "message": "x" * 100}, "runtime_error:resource_limit_exceeded"),
        (RuntimeError("boom"), "runtime_error:policy_invocation_failed"),
    ],
)  # fmt: skip
def test_policy_failures_fail_closed_with_policy_input(response, reason):
    result, _ = _run(response, limits=Limits(max_policy_output_bytes=90))
    assert (result.verdict.decision, result.verdict.reason) == (Decision.DENY, reason)
    assert result.transformed_policy_target is None and not result.verdict.liftable
    pi = result.policy_input
    assert pi is not None and result.input_identity == action_identity(pi)
    assert pi["annotations"] == {"alpha": {"ok": True}, "beta": {"ok": True}}


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"point": "output"}, "runtime_error:intervention_point_unknown"),
        ({"mode": "shadow"}, "host_error:context_invalid"),
        ({"snap": {"input": {1, 2}}}, "host_error:context_invalid"),
        ({"snap": {"other": 1}}, "runtime_error:path_missing"),
        ({"snap": {"input": "str"}}, "runtime_error:path_type_mismatch"),
        ({"limits": Limits(max_snapshot_bytes=8)}, "runtime_error:resource_limit_exceeded"),
        ({"limits": Limits(max_policy_input_depth=3)}, "runtime_error:resource_limit_exceeded"),
        ({"limits": Limits(max_annotators_per_point=1)}, "runtime_error:resource_limit_exceeded"),
        ({"annotator": Annotator(error=ValueError())}, "runtime_error:annotation_failed"),
        ({"annotator": Annotator(error=TimeoutError())}, "runtime_error:annotation_timeout"),
        ({"annotator": Annotator({"alpha": {"reason": "runtime_error:x"}})}, "runtime_error:annotation_failed"),
        ({"annotator": Annotator({"alpha": float("nan")})}, "runtime_error:annotation_failed"),
        ({"annotator": Annotator({"alpha": "x" * 99}), "limits": Limits(max_annotator_output_bytes=50)}, "runtime_error:annotation_failed"),
        ({"point": "pre_tool_call", "snap": {"tool_call": {"name": "rm", "args": {}}}}, "runtime_error:tool_unknown"),
        ({"point": "pre_tool_call", "snap": {"tool_call": {"name": 3, "args": {}}}}, "runtime_error:path_type_mismatch"),
        ({"point": "pre_tool_call", "snap": {"tool_call": {"name": "search", "args": {}}}, "tool": "x"}, "host_error:context_invalid"),
        ({"tool": "search"}, "host_error:context_invalid"),
        ({"point": "agent_startup", "snap": {"a": 1}, "dispatcher": {"rego": lambda i: {}}}, "host_error:adapter_unsupported"),
    ],
)  # fmt: skip
def test_runtime_errors_fail_closed(kwargs, reason):
    result, _ = _run({"decision": "allow"}, **kwargs)
    assert (result.verdict.decision, result.verdict.reason) == (Decision.DENY, reason)


def test_tool_projection_and_explicit_tool_name():
    result, _ = _run(
        {"decision": "allow"},
        point="pre_tool_call",
        snap={"tool_call": {"name": "search", "args": {"q": 1}}},
    )
    assert result.policy_input["tool"] == {"clearance": "internal", "name": "search"}
    assert result.policy_input["policy_target"]["kind"] == "tool_args"
    explicit, _ = _run(
        {"decision": "allow"}, point="post_tool_call", snap={"result": 1}, tool="search"
    )
    assert explicit.policy_input["tool"]["name"] == "search"


def test_dispatcher_mapping_by_adapter_and_point_transform_rules():
    deny = {"decision": "transform", "transform": {"path": "$target.a", "value": 2}}
    result, _ = _run(
        None,
        point="agent_startup",
        snap={"a": 1},
        dispatcher={"fixture": lambda i: deny},
    )
    assert result.verdict.reason == "host_error:transform_target_forbidden"
    big = {
        "decision": "transform",
        "transform": {"path": "$target.text", "value": "x" * 64},
    }
    result, _ = _run(big, limits=Limits(max_snapshot_bytes=60))
    assert result.verdict.reason == "runtime_error:resource_limit_exceeded"


def test_manifest_yaml_parity_with_mapping_source():
    rt = AcsRuntime(
        yaml.safe_load(MANIFEST),
        dispatcher=lambda i: {"decision": "allow"},
        annotator=Annotator(),
    )
    assert (
        asyncio.run(rt.evaluate_intervention_point("input", SNAP)).verdict.decision
        is Decision.ALLOW
    )
