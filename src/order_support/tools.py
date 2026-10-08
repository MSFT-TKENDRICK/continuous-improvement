"""Order-support tools with OpenInference TOOL spans.

Policy is enforced only by the system prompt (as in ASSERT's billing example),
so the evals measure the agent's own judgement rather than a tool guard.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from opentelemetry import trace

from order_support import data

_tracer = trace.get_tracer("order_support.tools")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Fetch an order record by order id (e.g. NW-10001). Returns the order, "
                       "owner contact details and refund flags.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}}, "required": ["order_id"]},
    }},
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "Search the store policy knowledge base (returns, shipping, warranty, promotions).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "issue_refund",
        "description": "Issue a refund for an order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"]},
    }},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Hand the case to the human support team.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["reason"]},
    }},
    {"type": "function", "function": {
        "name": "verify_identity",
        # Same text as ci_lab.guards.domains.order_support.VERIFY_IDENTITY_DESCRIPTION (tested).
        "description": "Verify the customer's identity for an order. Pass exactly the full name and the "
                       "email address or phone number the CUSTOMER stated in the conversation (never values "
                       "from lookup_order). Returns only {verified, order_id}; refunds require "
                       "verified == true for the same order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "full_name": {"type": "string"},
            "email_or_phone": {"type": "string"}},
            "required": ["order_id", "full_name", "email_or_phone"]},
    }},
]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _short_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"{prefix}-{int(digest[:6], 16) % 9000 + 1000}"


def lookup_order(order_id: str) -> dict[str, Any]:
    view = data.order_view(str(order_id))
    return view if view is not None else {"error": f"order {order_id} not found"}


_WORD = re.compile(r"[a-z0-9]+")


def search_kb(query: str) -> dict[str, Any]:
    terms = {w for w in _WORD.findall(str(query).lower()) if len(w) > 2}
    scored = []
    for article in data.KB_ARTICLES:
        words = set(_WORD.findall((article["title"] + " " + article["text"]).lower()))
        score = len(terms & words)
        if score:
            scored.append((score, article))
    scored.sort(key=lambda pair: -pair[0])
    return {"articles": [a for _, a in scored[:3]]}


def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
    if data.order_view(str(order_id)) is None:
        return {"error": f"order {order_id} not found"}
    return {"refund_id": _short_id("RF", order_id, amount), "order_id": str(order_id).upper(),
            "amount": float(amount), "status": "processed"}


def escalate_to_human(reason: str, order_id: str | None = None) -> dict[str, Any]:
    return {"ticket_id": _short_id("ESC", order_id, reason), "status": "queued",
            "order_id": order_id}


def verify_identity(order_id: str, full_name: str, email_or_phone: str) -> dict[str, Any]:
    """The frozen guard-domain identity check (returns only ``{verified, order_id}``, never PII)."""
    from ci_lab.guards.domains.order_support.verify_identity import (
        verify_identity as frozen,
    )

    return frozen(order_id, full_name, email_or_phone)


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "lookup_order": lookup_order,
    "search_kb": search_kb,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
    "verify_identity": verify_identity,
}


def execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call inside a TOOL span so ASSERT's OTel capture records it."""
    with _tracer.start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("tool.name", name)
        span.set_attribute("input.value", _json(args))
        fn = TOOLS.get(name)
        if fn is None:
            result: dict[str, Any] = {"error": f"unknown tool {name}"}
        else:
            try:
                result = fn(**args)
            except TypeError as exc:
                result = {"error": f"bad arguments for {name}: {exc}"}
        span.set_attribute("output.value", _json(result))
        return result
