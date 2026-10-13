"""Deterministic feature reduction for untrusted traces (B3, C12/C22).

Untrusted sources (usage, PR comments, customer text) and injection-suspect trajectories are
reduced to typed features before clustering, persistence or any LLM call:

* args/results → typed shapes: bools; numbers clamped to ±``NUM_BOUND`` and rounded; strings kept
  only when they belong to a closed enum vocabulary; id-like strings → ``#id:<keyed hash>`` of the
  normalized id (joins on the same subject still work); any other text → ``#txt:<keyed hash>``;
* response text → ``text_digest`` only (``text`` dropped); user text dropped entirely;
* rule / rubric ids and error classes must be well-formed identifiers; tool names likewise.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from ci_lab.lessons.common import clean_ident, clean_rule_ids, digest, keyed_hash
from ci_lab.rulespec import Outcome, Trajectory, TrajectoryStep, normalize_subject

NUM_BOUND = 1_000_000.0
MAX_ITEMS = 20
MAX_KEYS = 32
MAX_DEPTH = 4
ID_PREFIX = "#id:"
TXT_PREFIX = "#txt:"

#: Closed vocabulary of enum-like tokens that may survive reduction verbatim.
DEFAULT_VOCAB: frozenset[str] = frozenset({
    "true", "false", "yes", "no", "none", "null", "ok", "error", "blocked",
    "delivered", "shipped", "in_transit", "processing", "pending", "processed", "cancelled", "canceled",
    "returned", "refunded", "failed", "succeeded", "open", "closed", "escalated", "denied", "approved",
    "full", "partial", "damaged", "defective", "late", "lost", "wrong_item", "not_received", "other",
    "email", "phone", "name", "address", "card", "low", "medium", "high", "urgent",
})

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_TOOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_ID_RES = (
    re.compile(r"^[A-Za-z]{1,8}[-_]?\d{2,}[A-Za-z0-9-]*$"),           # NW-10003, RF5531, ORD_123
    re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    re.compile(r"^\d{4,}$"),
    re.compile(r"^[0-9a-fA-F]{16,64}$"),
)


def is_reduced(value: Any) -> bool:
    return isinstance(value, str) and value.startswith((ID_PREFIX, TXT_PREFIX))


def reduce_value(value: Any, *, vocab: frozenset[str] = DEFAULT_VOCAB, depth: int = 0) -> Any:
    """Typed shape of one value (see module doc). Pure and deterministic for a fixed salt."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        f = max(-NUM_BOUND, min(NUM_BOUND, f))
        return int(f) if isinstance(value, int) else round(f, 2)
    if depth >= MAX_DEPTH:
        return TXT_PREFIX + "deep"
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in vocab:
            return s.lower()
        if is_reduced(s):
            return s
        if any(rx.match(s) for rx in _ID_RES):
            return ID_PREFIX + keyed_hash(normalize_subject(s))
        return TXT_PREFIX + keyed_hash(s)
    if isinstance(value, (list, tuple)):
        return [reduce_value(v, vocab=vocab, depth=depth + 1) for v in list(value)[:MAX_ITEMS]]
    if isinstance(value, dict):
        keys = sorted(k for k in value if isinstance(k, str) and _KEY_RE.match(k))[:MAX_KEYS]
        return {k: reduce_value(value[k], vocab=vocab, depth=depth + 1) for k in keys}
    return TXT_PREFIX + keyed_hash(repr(type(value)))


def reduce_tool(name: str | None) -> str | None:
    if name is None:
        return None
    return name if _TOOL_RE.match(name) else TXT_PREFIX + keyed_hash(name)


def reduce_step(step: TrajectoryStep, *, vocab: frozenset[str] = DEFAULT_VOCAB) -> TrajectoryStep:
    text_digest = None
    if step.kind == "response":
        text_digest = step.text_digest or (digest(step.text) if step.text else None)
    flags = {f: [reduce_value(s, vocab=vocab) for s in subs] for f, subs in step.flags.items()
             if clean_ident(f)}
    return TrajectoryStep(
        i=step.i, kind=step.kind, tool=reduce_tool(step.tool),
        call_id=None if step.call_id is None else ID_PREFIX + keyed_hash(step.call_id),
        args=reduce_value(step.args, vocab=vocab) if step.args else {},
        result=reduce_value(step.result, vocab=vocab) if step.result is not None else None,
        status=step.status, text=None, text_digest=None if step.kind == "user" else text_digest,
        flags={k: [str(x) for x in v] for k, v in flags.items()})


def reduce_outcome(outcome: Outcome) -> Outcome:
    return Outcome(passed=outcome.passed, oracle_rules=clean_rule_ids(outcome.oracle_rules),
                   rubric_fails=clean_rule_ids(outcome.rubric_fails),
                   error_class=clean_ident(outcome.error_class), human_label=outcome.human_label,
                   injection_suspect=outcome.injection_suspect)


def reduce_trajectory(traj: Trajectory, *, vocab: frozenset[str] = DEFAULT_VOCAB) -> Trajectory:
    """Fully reduced copy: no raw text, ids hashed, enums from the closed vocabulary only."""
    return traj.model_copy(update={
        "steps": tuple(reduce_step(s, vocab=vocab) for s in traj.steps),
        "outcome": reduce_outcome(traj.outcome),
    })


def raw_strings(traj: Trajectory) -> Iterable[str]:
    """Every string a trajectory carries (test helper / leak audits)."""
    def walk(v: Any) -> Iterable[str]:
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            for k, x in v.items():
                yield str(k)
                yield from walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from walk(x)

    yield from walk(traj.model_dump(mode="json"))
