"""Deterministic metric checks: ``{kind: metric, metric, op: le|ge, target, worst}`` scored linearly to [0, 1]."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = ["METRIC_CHECK_KEYS", "METRIC_NAMES", "METRIC_OPS", "score_metric", "validate_metric_check"]

METRIC_NAMES = ("wall_ms", "tokens", "tokens_in", "tokens_out", "llm_calls", "tool_calls", "output_chars",
                "output_lines", "complexity_delta", "edit_lines")
METRIC_OPS = ("le", "ge")
METRIC_CHECK_KEYS = frozenset({"kind", "metric", "op", "target", "worst"})


def _number(check: Mapping[str, Any], key: str) -> float:
    v = check.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"metric check {key!r} must be a finite number, got {v!r}")
    return float(v)


def validate_metric_check(check: Mapping[str, Any]) -> tuple[str, str, float, float]:
    """``(metric, op, target, worst)`` of a well-formed check; ``ValueError`` otherwise.

    ``le`` (lower is better) needs ``target < worst``; ``ge`` (higher is better) needs ``target > worst``."""
    if not isinstance(check, Mapping):
        raise ValueError(f"metric check must be a mapping, got {type(check).__name__}")  # noqa: TRY004 - spec: ValueError
    if unknown := sorted(set(check) - METRIC_CHECK_KEYS):
        raise ValueError(f"metric check has unknown keys {unknown}")
    if check.get("kind") != "metric":
        raise ValueError(f"metric check needs kind: metric, got {check.get('kind')!r}")
    metric = check.get("metric")
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric check metric {metric!r} not in {list(METRIC_NAMES)}")
    op = check.get("op")
    if op not in METRIC_OPS:
        raise ValueError(f"metric check op {op!r} not in {list(METRIC_OPS)}")
    target, worst = _number(check, "target"), _number(check, "worst")
    if (op == "le" and not target < worst) or (op == "ge" and not target > worst):
        raise ValueError(f"metric check op {op} needs target {'<' if op == 'le' else '>'} worst "
                         f"(target={target}, worst={worst})")
    return str(metric), str(op), target, worst


def score_metric(check: Mapping[str, Any], value: float) -> float:
    """1.0 at or beyond ``target``, 0.0 at or beyond ``worst``, linear in between (direction from ``op``)."""
    _, _, target, worst = validate_metric_check(check)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"metric value must be a finite number, got {value!r}")
    v = float(value)
    frac = (worst - v) / (worst - target)
    return min(max(frac, 0.0), 1.0)
