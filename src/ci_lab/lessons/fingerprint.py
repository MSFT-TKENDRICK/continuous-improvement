"""Failure fingerprints (design §13.3 step 2, N2).

``Fingerprint`` = oracle rule ids + rubric fail ids + canonicalized tool-sequence 3-grams preceding the
failure + error class, versioned by the oracle/evaluator ``pin``. Canonicalization dedupes immediate
retries (consecutive calls to the same tool) and collapses repeated lookups (a read-only lookup tool
seen again before any other tool ran).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ci_lab.lessons.common import clean_ident
from ci_lab.rulespec import Fingerprint, Trajectory, TrajectoryStep

START = "^"
END = "$"
LOOKUP_PREFIXES = ("get_", "lookup_", "search_", "find_", "list_", "read_", "fetch_", "check_")
DEFAULT_WINDOW = 1  # number of trailing 3-grams kept (the ones immediately preceding the failure)


def is_failure(t: Trajectory) -> bool:
    o = t.outcome
    return (o.passed is False or bool(o.oracle_rules) or bool(o.rubric_fails) or o.human_label == "bad"
            or (o.error_class is not None and o.passed is not True))


def is_good(t: Trajectory) -> bool:
    """Known-good negatives for FP estimation: human-good, or passed with no findings and not human-bad."""
    o = t.outcome
    if o.human_label is not None:
        return o.human_label == "good"
    return o.passed is True and not o.oracle_rules and not o.rubric_fails


def is_lookup(tool: str) -> bool:
    return tool.lower().startswith(LOOKUP_PREFIXES)


@dataclass(frozen=True)
class Call:
    """A tool_call step paired with its result."""

    index: int          # position in canonical sequence
    step: TrajectoryStep
    result: TrajectoryStep | None

    @property
    def tool(self) -> str:
        return self.step.tool or "?"

    @property
    def status(self) -> str:
        if self.step.status == "blocked":
            return "blocked"
        return (self.result.status or "ok") if self.result is not None else "ok"


def calls(t: Trajectory) -> list[Call]:
    """All tool calls (raw order) with their results."""
    results = {s.call_id: s for s in t.steps if s.kind == "tool_result" and s.call_id}
    out = []
    for s in t.steps:
        if s.kind == "tool_call":
            out.append(Call(index=len(out), step=s, result=results.get(s.call_id) if s.call_id else None))
    return out


def canonical_calls(t: Trajectory) -> list[Call]:
    """Calls with immediate retries deduped (last attempt kept) and repeated lookups collapsed."""
    out: list[Call] = []
    seen_lookups: set[str] = set()
    for c in calls(t):
        if out and out[-1].tool == c.tool:
            out[-1] = Call(index=out[-1].index, step=c.step, result=c.result)
            continue
        if is_lookup(c.tool):
            if c.tool in seen_lookups:
                continue
            seen_lookups.add(c.tool)
        else:
            seen_lookups.clear()
        out.append(Call(index=len(out), step=c.step, result=c.result))
    return out


def _rule_stems(rules: Sequence[str]) -> list[str]:
    stems = []
    for r in rules:
        parts = r.replace("-", "_").replace(".", "_").split("_")
        stems += [p for p in parts if len(p) >= 4]
    return stems


def anchor_index(t: Trajectory, seq: Sequence[Call] | None = None) -> int | None:
    """Index (in the canonical sequence) of the failing call, or None if the failure is at the end."""
    seq = canonical_calls(t) if seq is None else seq
    for c in seq:
        if c.status in ("error", "blocked"):
            return c.index
    stems = _rule_stems(t.outcome.oracle_rules)
    for c in seq:
        if any(s in c.tool.lower() for s in stems):
            return c.index
    return None


def anchor_call(t: Trajectory) -> Call | None:
    seq = canonical_calls(t)
    i = anchor_index(t, seq)
    return seq[i] if i is not None else None


def error_class(t: Trajectory) -> str | None:
    if t.outcome.error_class:
        return t.outcome.error_class
    for c in canonical_calls(t):
        if c.status == "blocked":
            return "guard_blocked"
        if c.status == "error" and c.result is not None and c.result.result:
            r = c.result.result
            for k in ("error_code", "code", "error_type", "error"):
                if (v := clean_ident(r.get(k))) and len(v) <= 40 and " " not in v:
                    return v
            return "tool_error"
    return None


def tool_ngrams(t: Trajectory, *, window: int = DEFAULT_WINDOW) -> tuple[tuple[str, ...], ...]:
    seq = canonical_calls(t)
    i = anchor_index(t, seq)
    names = [c.tool for c in seq]
    tokens = [START, START] + (names[: i + 1] if i is not None else [*names, END])
    grams = [tuple(tokens[k:k + 3]) for k in range(len(tokens) - 2)]
    return tuple(grams[-window:]) if window > 0 else tuple(grams)


def fingerprint(t: Trajectory, *, window: int = DEFAULT_WINDOW) -> Fingerprint:
    return Fingerprint(pin=t.pin, oracle_rules=tuple(sorted(t.outcome.oracle_rules)),
                       rubric_ids=tuple(sorted(t.outcome.rubric_fails)), tool_ngrams=tool_ngrams(t, window=window),
                       error_class=error_class(t))


def cluster_id(fp: Fingerprint) -> str:
    return "lc-" + fp.digest()
