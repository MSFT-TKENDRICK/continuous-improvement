"""MAF bindings delegate to the frozen tools.execute (TOOL spans) and are idempotent."""

from __future__ import annotations

import json

import pytest

from order_support import maf_tools, tools

CALLS = [
    ("lookup_order", {"order_id": "NW-10001"}),
    ("search_kb", {"query": "returns window"}),
    ("issue_refund", {"order_id": "NW-10001", "amount": 89.5}),
    ("escalate_to_human", {"order_id": "NW-10002", "reason": "refund over limit"}),
    ("escalate_to_human", {"reason": "general question"}),
    ("verify_identity", {"order_id": "NW-10001", "full_name": "Nobody", "email_or_phone": "x@example.com"}),
]


def test_bindings_cover_every_frozen_tool():
    assert set(maf_tools.bindings()) == set(tools.TOOLS)


@pytest.mark.parametrize(("name", "args"), CALLS)
def test_binding_returns_tools_execute_json(name, args):
    out = maf_tools.bindings()[name](**args)
    assert json.loads(out) == tools.execute(name, dict(args))


@pytest.mark.parametrize(("name", "args"), CALLS)
def test_tools_are_deterministic_and_idempotent(name, args):
    # MAF checkpoint resume is at-least-once: a replayed call must give the same result.
    bind = maf_tools.bindings()[name]
    first = bind(**args)
    assert all(bind(**args) == first for _ in range(3))
    assert maf_tools.bindings()[name](**args) == first


def test_binding_emits_one_tool_span_per_call(captured):
    maf_tools.bindings()["issue_refund"](order_id="NW-10001", amount=89.5)
    spans = [s for s in captured() if s.attributes.get("openinference.span.kind") == "TOOL"]
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "tool.issue_refund"
    assert json.loads(span.attributes["input.value"]) == {"order_id": "NW-10001", "amount": 89.5}
    assert json.loads(span.attributes["output.value"])["status"] == "processed"


def test_verify_identity_is_the_frozen_guard_tool(captured):
    from ci_lab.guards.domains.order_support import (
        VERIFY_IDENTITY_DESCRIPTION,
        verify_identity,
    )
    from order_support import data

    order = data.ORDERS["NW-10001"]
    args = {"order_id": "nw-10001", "full_name": order["customer"]["name"].upper(),
            "email_or_phone": order["customer"]["email"]}
    out = json.loads(maf_tools.bindings()["verify_identity"](**args))
    assert out == verify_identity(**args) == {"verified": True, "order_id": "NW-10001"}
    schema = next(s["function"] for s in tools.TOOL_SCHEMAS if s["function"]["name"] == "verify_identity")
    assert schema["description"] == VERIFY_IDENTITY_DESCRIPTION
    spans = [s for s in captured() if s.attributes.get("openinference.span.kind") == "TOOL"]
    assert [s.name for s in spans] == ["tool.verify_identity"]
    assert "@" not in spans[0].attributes["output.value"]  # never echoes PII
