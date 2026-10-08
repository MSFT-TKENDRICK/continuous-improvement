"""Shared guard runtime: conversations, decisions (B1), modes, telemetry, templates.

One :class:`GuardRuntime` backs the middleware returned by ``install_guards``. It is the only
place that maps rule matches to decisions: every match is recorded (sink + span event) *before*
the middleware acts on it (B1, all-match telemetry N6).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from opentelemetry import trace

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_GUARD_ACTION,
    ATTR_GUARD_BUNDLE,
    ATTR_GUARD_DEGRADED,
    ATTR_GUARD_ENFORCED,
    ATTR_GUARD_MODE,
    ATTR_GUARD_RULE,
    ATTR_GUARD_VERSION,
    SPAN_GUARD,
)
from ci_lab.guards.engine import (
    GuardEngine,
    GuardUnavailable,
    On,
    bundle_digest,
    bundle_templates,
)
from ci_lab.guards.recorder import DecisionSink, TrajectoryRecorder
from ci_lab.rulespec import (
    MAX_GUARD_BLOCKS_PER_TURN,
    TOOL_SIDE_EFFECT_KEY,
    GuardDecision,
    GuardView,
    TemplateSpec,
    TrajectoryStep,
    attempt_digest,
)

Mode = Literal["off", "shadow", "enforce"]
MODES: tuple[str, ...] = ("off", "shadow", "enforce")
MAX_GUARD_BLOCKS_PER_CONVERSATION = 6  # B4 per-conversation retry ceiling
DEGRADED_RULE_ID = "guards.degraded"
TERMINAL_TEMPLATE = "guard.terminal"
RESPONSE_BLOCKED_TEMPLATE = "response.blocked"

# Frozen fallbacks (trusted code, B3) used only when the bundle's catalog lacks the template.
_FALLBACK_TEMPLATES = {
    TERMINAL_TEMPLATE: TemplateSpec(
        id=TERMINAL_TEMPLATE,
        message="Guard limit reached for this turn; no further tool calls will run.",
        fix="Stop calling tools. Tell the user the request cannot be completed right now and offer a human handoff."),
    RESPONSE_BLOCKED_TEMPLATE: TemplateSpec(
        id=RESPONSE_BLOCKED_TEMPLATE,
        message="I'm sorry, I can't share that. I can help with your order once your identity is verified.",
        fix="Rephrase without the blocked content."),
}

class _Slots(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


def render(template: TemplateSpec, slots: Mapping[str, Any] | None = None) -> tuple[str, str]:
    s = _Slots(slots or {})
    return template.message.format_map(s), template.fix.format_map(s)


def side_effect_policies(tool_policies: Mapping[str, Any]) -> dict[str, bool]:
    """``{tool: bool}`` or tool_specs-style ``{tool: {"side_effect": bool, ...}}`` → ``{tool: bool}``."""
    out: dict[str, bool] = {}
    for name, spec in tool_policies.items():
        if isinstance(spec, Mapping):
            out[str(name)] = bool(spec.get(TOOL_SIDE_EFFECT_KEY, False))
        else:
            out[str(name)] = bool(spec)
    return out


@dataclass
class Verdict:
    """Batch-preflight outcome for one side-effecting call (B5)."""

    pending: TrajectoryStep
    block: Any | None  # the enforced block Match, if any


@dataclass
class TurnState:
    blocks: int = 0
    terminal: bool = False
    terminal_rule: str = ""
    preflight: dict[str, Verdict] = field(default_factory=dict)


@dataclass
class ConvState:
    key: str
    recorder: TrajectoryRecorder
    turn: TurnState = field(default_factory=TurnState)
    decisions: list[GuardDecision] = field(default_factory=list)
    _lock: asyncio.Lock | None = None

    @property
    def side_effect_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


class GuardRuntime:
    """State + policy shared by the guard middleware of one install."""

    def __init__(self, *, bundle: Any | None, engine: GuardEngine | None, tool_policies: Mapping[str, Any],
                 mode_override: Mode | None, sink: DecisionSink | None, degraded: bool,
                 max_blocks_per_turn: int = MAX_GUARD_BLOCKS_PER_TURN,
                 max_blocks_per_conversation: int = MAX_GUARD_BLOCKS_PER_CONVERSATION,
                 default_side_effect: bool = True, buffer_streams: bool = True) -> None:
        self.bundle = bundle
        self.engine = engine
        self.policies = side_effect_policies(tool_policies)
        self.mode_override = mode_override
        self.sink = sink
        self.degraded = degraded or bundle is None or engine is None
        self.max_blocks_per_turn = max_blocks_per_turn
        self.max_blocks_per_conversation = max_blocks_per_conversation
        self.default_side_effect = default_side_effect
        self.buffer_streams = buffer_streams
        self.digest = bundle_digest(bundle)
        self.decisions: list[GuardDecision] = []
        self._convs: dict[str, ConvState] = {}
        self._current: ContextVar[ConvState | None] = ContextVar(f"ci_guards_conv_{id(self)}", default=None)

    # ------------------------------------------------------------ policy
    def is_side_effect(self, tool: str) -> bool:
        return self.policies.get(tool, self.default_side_effect)

    def effective_mode(self, rule: Any) -> Mode:
        # "off" still records every match (paired guard-off arm, B1) but never enforces.
        if self.mode_override in ("off", "shadow"):
            return self.mode_override
        if self.mode_override == "enforce":
            return "enforce"
        return "enforce" if getattr(rule, "mode", "shadow") == "enforce" else "shadow"

    def template(self, template_id: str) -> TemplateSpec:
        found = bundle_templates(self.bundle).get(template_id)
        return found if isinstance(found, TemplateSpec) else _FALLBACK_TEMPLATES[template_id]

    # ------------------------------------------------------------ conversations
    def conversation(self, session: Any | None = None) -> ConvState:
        """The active conversation: the one bound by the agent middleware for this run, else the
        session's, else a shared unscoped fallback (chat-client use without an agent)."""
        bound = self._current.get()
        if bound is not None:
            return bound
        sid = getattr(session, "session_id", None)
        return self.open(str(sid) if sid else "_unscoped", getattr(session, "state", None))

    def open(self, key: str, state: Any | None = None) -> ConvState:
        conv = self._convs.get(key)
        if conv is None:
            conv = self._convs[key] = ConvState(key=key, recorder=TrajectoryRecorder(key))
        if isinstance(state, dict):
            conv.recorder.bind_state(state)
        return conv

    def start_run(self, session: Any | None) -> ConvState:
        sid = getattr(session, "session_id", None)
        conv = self.open(str(sid) if sid else f"run-{uuid.uuid4().hex}", getattr(session, "state", None))
        conv.turn = TurnState()
        return conv

    def bind(self, conv: ConvState | None) -> Any:
        return self._current.set(conv)

    def unbind(self, token: Any) -> None:
        try:
            self._current.reset(token)
        except ValueError:  # reset from a different context (stream consumed elsewhere)
            self._current.set(None)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, view: GuardView, *, on: On) -> list[Any]:
        if self.bundle is None or self.engine is None:
            raise GuardUnavailable("guard bundle or engine unavailable")
        return list(self.engine.evaluate(self.bundle, view, on=on))  # engine order: (rung, id)

    def redact(self, text: str, matches: Sequence[Any]) -> str:
        if self.engine is None:
            raise GuardUnavailable("guard engine unavailable")
        return self.engine.redact(text, matches, self.bundle)

    def decide(self, conv: ConvState, matches: Sequence[Any], pending: TrajectoryStep) -> list[GuardDecision]:
        """Record one decision per match BEFORE acting (B1); return them in precedence order."""
        digest = attempt_digest(pending)
        out = []
        for m in matches:
            rule = m.rule
            mode = self.effective_mode(rule)
            enforced = mode == "enforce" and rule.action in ("block", "redact")
            out.append(self._record(conv, GuardDecision(
                rule_id=rule.id, rule_version=rule.version, mode=mode, action=rule.action, enforced=enforced,
                step_index=pending.i, target=rule.target, attempt_digest=digest, degraded=self.degraded)))
        return out

    def record_degraded(self, conv: ConvState, pending: TrajectoryStep) -> GuardDecision:
        obs.annotate({ATTR_GUARD_DEGRADED: True})
        return self._record(conv, GuardDecision(
            rule_id=DEGRADED_RULE_ID, rule_version=0, mode="shadow", action="warn", enforced=False,
            step_index=pending.i, target=pending.tool or "*", attempt_digest=attempt_digest(pending),
            degraded=True))

    def _record(self, conv: ConvState, decision: GuardDecision) -> GuardDecision:
        conv.decisions.append(decision)
        self.decisions.append(decision)
        if self.sink is not None:
            self.sink(decision)
        trace.get_current_span().add_event(SPAN_GUARD, attributes={
            ATTR_GUARD_RULE: decision.rule_id,
            ATTR_GUARD_VERSION: decision.rule_version,
            ATTR_GUARD_MODE: decision.mode,
            ATTR_GUARD_ACTION: decision.action,
            ATTR_GUARD_ENFORCED: decision.enforced,
            ATTR_GUARD_BUNDLE: self.digest,
            ATTR_GUARD_DEGRADED: decision.degraded,
        })
        return decision

    @staticmethod
    def first_enforced(matches: Sequence[Any], decisions: Sequence[GuardDecision], action: str) -> Any | None:
        for m, d in zip(matches, decisions, strict=True):
            if d.enforced and d.action == action:
                return m
        return None
