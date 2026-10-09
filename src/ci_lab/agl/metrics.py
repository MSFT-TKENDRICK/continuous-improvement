"""Deterministic runtime metrics for Agent Lightning rollout events."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

METRIC = "ci.metric"
RUNTIME_FIELDS = ("wall_ms", "llm_calls", "tool_calls", "tokens_in", "tokens_out")
_IGNORED_EVENTS = frozenset({"ci.credit", "ci.error"})


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else 0.0


def _summary_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _usage(data: Mapping[str, Any]) -> tuple[int, int]:
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return 0, 0
    tokens_in = _number(usage.get("prompt_tokens", usage.get("input_tokens")))
    tokens_out = _number(usage.get("completion_tokens", usage.get("output_tokens")))
    return int(tokens_in), int(tokens_out)


def rollout_metrics(events: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Aggregate runtime data from ``model_request`` and ``ci.*`` events.

    Model requests and runtime delta events are summed. ``ci.score`` and ``ci.metric`` are
    typed summaries, so their valid fields overwrite earlier values instead of being added;
    replaying a completed journal therefore cannot recursively inflate its metrics.
    ``ci.tool_call`` and ``ci.llm_call`` count as one call when an explicit count is absent.
    """
    wall_ms = 0.0
    llm_calls = tool_calls = tokens_in = tokens_out = 0
    for event in events:
        event_type = str(event.get("event_type") or "")
        data = event.get("data")
        if not isinstance(data, Mapping):
            data = {}
        if event_type == "model_request":
            llm_calls += 1
            wall_ms += _number(data.get("latency_ms"))
            tin, tout = _usage(data)
            tokens_in += tin
            tokens_out += tout
            continue
        if not event_type.startswith("ci.") or event_type in _IGNORED_EVENTS:
            continue
        if event_type in (METRIC, "ci.score"):
            named = data.get("name")
            if named in RUNTIME_FIELDS:
                value = _summary_number(data.get("value"))
                if value is None:
                    continue
                if named == "wall_ms":
                    wall_ms = value
                elif named == "llm_calls":
                    llm_calls = int(value)
                elif named == "tool_calls":
                    tool_calls = int(value)
                elif named == "tokens_in":
                    tokens_in = int(value)
                elif named == "tokens_out":
                    tokens_out = int(value)
            for field in RUNTIME_FIELDS:
                if field not in data:
                    continue
                value = _summary_number(data[field])
                if value is None:
                    continue
                if field == "wall_ms":
                    wall_ms = value
                elif field == "llm_calls":
                    llm_calls = int(value)
                elif field == "tool_calls":
                    tool_calls = int(value)
                elif field == "tokens_in":
                    tokens_in = int(value)
                else:
                    tokens_out = int(value)
            continue
        values = data.get("metrics")
        values = values if isinstance(values, Mapping) else data
        wall_ms += _number(values.get("wall_ms"))
        llm = _number(values.get("llm_calls"))
        tools = _number(values.get("tool_calls"))
        llm_calls += int(llm) if llm else int(event_type == "ci.llm_call")
        tool_calls += int(tools) if tools else int(event_type == "ci.tool_call")
        tokens_in += int(_number(values.get("tokens_in")))
        tokens_out += int(_number(values.get("tokens_out")))
        tin, tout = _usage(values)
        tokens_in += tin
        tokens_out += tout
    return {
        "wall_ms": wall_ms,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
