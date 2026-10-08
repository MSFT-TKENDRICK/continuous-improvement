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
