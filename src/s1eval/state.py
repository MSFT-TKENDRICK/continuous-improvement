"""Leakage barrier: build judge state from an allowlist of *observable* trace fields.

Dataset cases look like::

    {id, tags, notes, labels, observable: {agent_policy, conversation, tool_calls, final_response}}

Only ``observable`` is ever read, and it is validated field-by-field against a strict
schema. Unknown keys anywhere inside ``observable`` raise ``LeakageError`` (fail closed),
so gold labels, notes, case ids or "expected behaviour" text cannot reach the judge.
"""

from __future__ import annotations

import copy
from typing import Any


class LeakageError(ValueError):
    """Trace contains fields outside the observable allowlist."""


TOP_LEVEL_FIELDS = ("agent_policy", "conversation", "tool_calls", "final_response")
MESSAGE_FIELDS = {"role", "content"}
MESSAGE_ROLES = {"user", "assistant"}
TOOL_CALL_FIELDS = {"name", "arguments", "result"}


def _check_keys(obj: Any, allowed: set[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise LeakageError(f"{where} must be an object")
    extra = set(obj) - allowed
    if extra:
        raise LeakageError(f"{where} has non-observable fields {sorted(extra)}")


def project_observable(case: dict[str, Any]) -> dict[str, Any]:
    """Return the judge state for a case. Raises LeakageError on anything unexpected."""
    obs = case.get("observable")
    _check_keys(obs, set(TOP_LEVEL_FIELDS), "observable")
    if "final_response" not in obs or not isinstance(obs["final_response"], str):
        raise LeakageError("observable.final_response (string) is required")
    state: dict[str, Any] = {}
    if "agent_policy" in obs:
        if not isinstance(obs["agent_policy"], str):
            raise LeakageError("observable.agent_policy must be a string")
        state["agent_policy"] = obs["agent_policy"]
    if "conversation" in obs:
        conv = obs["conversation"]
        if not isinstance(conv, list):
            raise LeakageError("observable.conversation must be a list")
        for i, m in enumerate(conv):
            _check_keys(m, MESSAGE_FIELDS, f"observable.conversation[{i}]")
            if m.get("role") not in MESSAGE_ROLES or not isinstance(m.get("content"), str):
                raise LeakageError(f"observable.conversation[{i}] needs role in {sorted(MESSAGE_ROLES)} and string content")
        state["conversation"] = copy.deepcopy(conv)
    if "tool_calls" in obs:
        calls = obs["tool_calls"]
        if not isinstance(calls, list):
            raise LeakageError("observable.tool_calls must be a list")
        for i, c in enumerate(calls):
            _check_keys(c, TOOL_CALL_FIELDS, f"observable.tool_calls[{i}]")
            if not isinstance(c.get("name"), str):
                raise LeakageError(f"observable.tool_calls[{i}].name must be a string")
        state["tool_calls"] = copy.deepcopy(calls)
    state["final_response"] = obs["final_response"]
    return state
