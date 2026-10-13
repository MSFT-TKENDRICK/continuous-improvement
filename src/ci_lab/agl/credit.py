"""Typed LLM credit assignment over bounded rollout digests."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ci_lab.contracts import COMPONENTS, RolloutJournal, RolloutKey, op_id

MAX_DIGESTS = 64
MAX_RULES = 32
MAX_COMPONENTS = 16
MAX_TEXT = 128
_REASON = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
RULE_PREFIX_COMPONENT: Mapping[str, str] = MappingProxyType({
    "budget.": "loop",
    "loop.": "loop",
    "agent.": "agent",
    "workflow.": "workflow",
    "mcp.": "mcp",
    "client_tool.": "client_tool",
    "tool.": "client_tool",
    "config.": "config",
    "context.": "context_mgmt",
    "memory.": "memory",
    "prompt.": "prompt",
    "skill.": "skill",
    "guard.": "guard",
    "injection.": "agent",
    "model.": "config",
    "metric.": "loop",
})
METRIC_NAMES = frozenset({"wall_ms", "llm_calls", "tool_calls", "tokens_in", "tokens_out"})


def _bounded_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT:
        raise ValueError(f"{name} must be a non-empty string of <= {MAX_TEXT} chars")
    return value


@dataclass(frozen=True)
class RolloutDigest:
    """No-payload rollout summary safe to send to the optimizer."""

    case: str
    suite: str
    score: float | None
    violation_rule_ids: tuple[str, ...] = ()
    component_touches: Mapping[str, int] = MappingProxyType({})
    metrics: Mapping[str, float] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "case", _bounded_text(self.case, "case"))
        object.__setattr__(self, "suite", _bounded_text(self.suite, "suite"))
        if self.score is not None and (isinstance(self.score, bool) or not math.isfinite(self.score)
                                       or not 0 <= self.score <= 1):
            raise ValueError("score must be None or finite in [0, 1]")
        rules = tuple(dict.fromkeys(_bounded_text(r, "rule id") for r in self.violation_rule_ids))
        if len(rules) > MAX_RULES:
            raise ValueError(f"at most {MAX_RULES} violation rule ids")
        touches: dict[str, int] = {}
        for component, count in self.component_touches.items():
            if component not in COMPONENTS or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"bad component touch {component!r}: {count!r}")
            touches[component] = min(count, 10_000)
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if name not in METRIC_NAMES or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"bad metric {name!r}: {value!r}")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"bad metric {name!r}: {value!r}")
            metrics[name] = value
        object.__setattr__(self, "violation_rule_ids", rules)
        object.__setattr__(self, "component_touches", MappingProxyType(touches))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    def as_dict(self) -> dict[str, Any]:
        return {"case": self.case, "suite": self.suite, "score": self.score,
                "violation_rule_ids": list(self.violation_rule_ids),
                "component_touches": dict(self.component_touches), "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class Credit:
    component: str
    weight: float
    reason_code: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.component not in COMPONENTS:
            raise ValueError(f"unknown component {self.component!r}")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)) \
                or not math.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("credit weight must be finite in [0, 1]")
        if not isinstance(self.reason_code, str) or not _REASON.fullmatch(self.reason_code):
            raise ValueError(f"bad reason code {self.reason_code!r}")
        evidence = tuple(dict.fromkeys(_bounded_text(e, "evidence id") for e in self.evidence_ids))
        if len(evidence) > MAX_RULES:
            raise ValueError(f"at most {MAX_RULES} evidence ids")
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "evidence_ids", evidence)


def _schema(components: Sequence[str]) -> dict[str, Any]:
    item = {"type": "object", "additionalProperties": False,
            "required": ["component", "weight", "reason_code", "evidence_ids"],
            "properties": {
                "component": {"type": "string", "enum": list(components)},
                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                "reason_code": {"type": "string", "pattern": _REASON.pattern},
                "evidence_ids": {"type": "array", "maxItems": MAX_RULES,
                                 "items": {"type": "string", "maxLength": MAX_TEXT}},
            }}
    return {"type": "object", "additionalProperties": False, "required": ["credits"],
            "properties": {"credits": {"type": "array", "maxItems": len(components), "items": item}}}


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _parse(text: str, components: Sequence[str]) -> list[Credit]:
    data = json.loads(text, object_pairs_hook=_no_dupes)
    if not isinstance(data, dict) or set(data) != {"credits"} or not isinstance(data["credits"], list):
        raise ValueError("response must be exactly {'credits': [...]}")
    if len(data["credits"]) > len(components):
        raise ValueError("too many credits")
    out: list[Credit] = []
    seen: set[str] = set()
    for row in data["credits"]:
        if not isinstance(row, dict) or set(row) != {"component", "weight", "reason_code", "evidence_ids"}:
            raise ValueError("credit has unknown or missing fields")
        if not isinstance(row["component"], str) or not isinstance(row["reason_code"], str) \
                or not isinstance(row["evidence_ids"], list) \
                or not all(isinstance(item, str) for item in row["evidence_ids"]):
            raise ValueError("credit fields have the wrong type")
        credit = Credit(row["component"], row["weight"], row["reason_code"], tuple(row["evidence_ids"]))
        if credit.component not in components or credit.component in seen:
            raise ValueError("credit component is unavailable or duplicated")
        seen.add(credit.component)
        out.append(credit)
    return out


def _outlier(value: float, values: Sequence[float]) -> bool:
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return False
    median = statistics.median(positive)
    return value > max(median * 1.5, median + 1)


def heuristic_credit(digests: Sequence[RolloutDigest], components: Sequence[str]) -> list[Credit]:
    """Frozen deterministic fallback: rule prefixes, then resource outliers and touched failures."""
    allowed = tuple(dict.fromkeys(components))
    found: dict[tuple[str, str], tuple[float, set[str]]] = {}

    def add(component: str, weight: float, reason: str, evidence: str) -> None:
        if component not in allowed:
            return
        current, ids = found.setdefault((component, reason), (weight, set()))
        ids.add(evidence)
        found[(component, reason)] = (max(current, weight), ids)

    for digest in digests:
        for rule in digest.violation_rule_ids:
            component = next((c for prefix, c in RULE_PREFIX_COMPONENT.items() if rule.startswith(prefix)), "agent")
            add(component, 1.0, "violation.rule", f"{digest.case}:{rule}")
        calls = digest.metrics.get("llm_calls", 0) + digest.metrics.get("tool_calls", 0)
        all_calls = [d.metrics.get("llm_calls", 0) + d.metrics.get("tool_calls", 0) for d in digests]
        if _outlier(calls, all_calls):
            add("loop", 0.9, "cost.calls", f"{digest.case}:calls")
        tokens = digest.metrics.get("tokens_in", 0) + digest.metrics.get("tokens_out", 0)
        all_tokens = [d.metrics.get("tokens_in", 0) + d.metrics.get("tokens_out", 0) for d in digests]
        if _outlier(tokens, all_tokens):
            add("agent", 0.8, "cost.tokens", f"{digest.case}:tokens")
        wall = digest.metrics.get("wall_ms", 0)
        if _outlier(wall, [d.metrics.get("wall_ms", 0) for d in digests]):
            add("loop", 0.7, "cost.wall", f"{digest.case}:wall")
        if digest.score is not None and digest.score < 0.5 and not digest.violation_rule_ids:
            touched = sorted(digest.component_touches.items(), key=lambda item: (-item[1], item[0]))
            component = next((c for c, count in touched if count and c in allowed), "agent")
            add(component, 0.6, "score.failure", digest.case)
    if not found and allowed:
        add(allowed[0], 0.0, "no.signal", "batch")
    credits = [Credit(component, weight, reason, tuple(sorted(ids)))
               for (component, reason), (weight, ids) in found.items()]
    return sorted(credits, key=lambda c: (-c.weight, c.component, c.reason_code))


async def assign_credit(digests: Sequence[RolloutDigest], *, client: Any,
                        components: Sequence[str]) -> list[Credit]:
    """Make one strict optimizer call for the batch, retry once, then use frozen heuristics."""
    batch = tuple(digests)
    allowed = tuple(dict.fromkeys(components))
    if len(batch) > MAX_DIGESTS:
        raise ValueError(f"at most {MAX_DIGESTS} rollout digests per batch")
    if not allowed or len(allowed) > MAX_COMPONENTS or any(c not in COMPONENTS for c in allowed):
        raise ValueError("components must be a non-empty bounded subset of contracts.COMPONENTS")
    schema = _schema(allowed)
    prompt = ("Assign failure/resource credit to harness components. Use only the typed digests; "
              "never infer prompt or tool payload text. Return strict JSON matching the schema.\n"
              + json.dumps({"components": allowed, "digests": [d.as_dict() for d in batch]},
                           sort_keys=True, separators=(",", ":")))
    options = {"response_format": {"type": "json_schema",
                                   "json_schema": {"name": "agl_credit", "strict": True, "schema": schema}}}
    retry_note = ""
    for attempt in range(2):
        try:
            response = await client.get_response(prompt + retry_note, options=options)
            return _parse(str(getattr(response, "text", response)), allowed)
        except Exception as exc:  # noqa: BLE001 - invalid/model/client failures use deterministic fallback
            retry_note = f"\nPrevious response failed validation ({type(exc).__name__}); return only valid JSON."
    return heuristic_credit(batch, allowed)


def journal_credits(journal: RolloutJournal, key: RolloutKey, credits: Sequence[Credit]) -> None:
    """Append typed ``ci.credit`` events; deterministic ids make replay a no-op."""
    for credit in credits:
        name = "|".join((credit.component, credit.reason_code, *sorted(credit.evidence_ids)))
        journal.event(key, "ci.credit", {
            "component": credit.component, "weight": credit.weight,
            "reason_code": credit.reason_code, "evidence_ids": list(credit.evidence_ids),
        }, event_id=op_id(key.rollout_id, key.attempt_id, "ci.credit", name))
