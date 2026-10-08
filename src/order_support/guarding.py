"""Guard wiring for the order-support agent (HOOK(M3), docs/guards.md "Order-support installation").

* :func:`install` builds the frozen guard middleware (``ci_lab.guards.install_guards``) once per
  process for a given guards dir, ``$CI_GUARDS`` mode, decision sink and tool policies.
* :func:`conversation` gives each conversation one persistent ``AgentSession`` so guard state
  (``identity_verified``, prior lookups) survives across turns; :func:`transcript` loads the
  text-seeded prior turns through a history provider, so the model input is unchanged.
* With ``$CI_GUARD_DECISIONS`` set (``ci_lab.lessons_arm.paired``), every decision is appended to
  ``$CI_GUARD_DECISIONS/<case_id>/<trial>.jsonl`` with ``case_id``/``trial`` filled in, and
  :func:`flush` adds the ``call``/``opportunity`` records ``paired.read_run`` reads.
* :func:`seed` derives the sampling seed from ``(case, trial)`` only, never the variant, so the
  guard-off and guard-on arms of a paired run see the same randomness.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_framework import AgentSession, HistoryProvider, Message

DECISIONS_ENV = "CI_GUARD_DECISIONS"  # == ci_lab.lessons_arm.paired.DECISIONS_ENV
GUARDS_ENV = "CI_GUARDS"              # read by ci_lab.guards.install.resolve_mode only
RUN_DIR_ENV = "CI_RUN_DIR"
CASE_ENV = "CI_CASE_ID"
TRIAL_ENV = "CI_TRIAL"
SEED_ENV = "ORDER_AGENT_SEED"
GUARDS_SUBDIR = "guards"
TRANSCRIPT_SOURCE = "order_support.transcript"
MAX_CONVERSATIONS = 512
UNATTRIBUTED = "_unattributed"

_SAFE = re.compile(r"[^A-Za-z0-9._-]")
# (ASSERT test_case_id, per-case-run token), bound by order_support.assert_wrapper around each case.
_case: ContextVar[tuple[str, str] | None] = ContextVar("order_support_case", default=None)


# ------------------------------------------------------------------ case identity

def bind_case(test_case_id: str) -> Token[tuple[str, str] | None]:
    """Mark the current context as one run of ASSERT case ``test_case_id`` (fresh conversation scope)."""
    return _case.set((str(test_case_id), uuid.uuid4().hex))


def reset_case(token: Token[tuple[str, str] | None]) -> None:
    _case.reset(token)


def case_identity(env: Mapping[str, str] | None = None) -> tuple[str | None, int]:
    """``(case_id, trial)``: ``$CI_CASE_ID`` (set by the domain per subprocess), else the bound ASSERT
    case; ``$CI_TRIAL`` (default 0)."""
    env = os.environ if env is None else env
    bound = _case.get()
    case = env.get(CASE_ENV) or (bound[0] if bound else None)
    return case, int(env.get(TRIAL_ENV) or 0)


def seed(env: Mapping[str, str] | None = None) -> int | None:
    """``$ORDER_AGENT_SEED``, else a hash of ``(case, trial)`` when ``$CI_CASE_ID`` is set, else None.

    Deliberately independent of ``$CI_VARIANT`` / ``$CI_GUARDS`` (paired guard arms, B1).
    """
    env = os.environ if env is None else env
    if raw := (env.get(SEED_ENV) or "").strip():
        return int(raw)
    case = env.get(CASE_ENV)
    if not case:
        return None
    digest = hashlib.sha256(f"{case}|{int(env.get(TRIAL_ENV) or 0)}".encode()).hexdigest()
    return int(digest[:8], 16) % 2**31


# ------------------------------------------------------------------ decision sink

class DecisionWriter:
    """Callable ``GuardDecision`` sink with case attribution, plus raw ``call``/``opportunity`` records.

    ``per_case`` (``$CI_GUARD_DECISIONS``): ``<root>/<case_id>/<trial>.jsonl``; otherwise one file
    ``<root>/decisions.jsonl`` (``$CI_RUN_DIR/guards``). Unknown cases never get a ``case_id`` key.
    """

    def __init__(self, root: Path | str, *, per_case: bool) -> None:
        self.root = Path(root)
        self.per_case = per_case
        self._lock = threading.Lock()

    def path(self, case_id: str | None, trial: int) -> Path:
        if not self.per_case:
            return self.root / "decisions.jsonl"
        if case_id is None:
            return self.root / f"{UNATTRIBUTED}.jsonl"
        return self.root / (_SAFE.sub("_", case_id) or "_") / f"{trial}.jsonl"

    def write(self, records: Sequence[Mapping[str, Any]]) -> None:
        from ci_lab.rulespec import canonical_json

        if not records:
            return
        case_id, trial = case_identity()
        lines = []
        for record in records:
            obj = {k: v for k, v in record.items() if v is not None}
            if case_id is not None:
                obj.update(case_id=case_id, trial=trial)
            lines.append(canonical_json(obj) + "\n")
        path = self.path(case_id, trial)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.writelines(lines)

    def __call__(self, decision: Any) -> None:
        self.write([decision.model_dump(mode="json", exclude_none=True)])


def decision_writer(env: Mapping[str, str] | None = None) -> DecisionWriter | None:
    env = os.environ if env is None else env
    if root := env.get(DECISIONS_ENV):
        return DecisionWriter(root, per_case=True)
    if run_dir := env.get(RUN_DIR_ENV):
        return DecisionWriter(Path(run_dir) / GUARDS_SUBDIR, per_case=False)
    return None


# ------------------------------------------------------------------ install (once per process)

@dataclass
class Installed:
    guards: Any            # ci_lab.guards.install.GuardMiddleware ([agent, preflight, function], .runtime)
    writer: DecisionWriter | None
    guards_dir: Path


_installs: dict[tuple[Any, ...], Installed] = {}
_install_lock = threading.Lock()
install_count = 0  # install_guards() calls in this process (tests)


def guards_dir(harness: Path | str) -> Path:
    """``<harness>/guards`` when it holds rules, else the packaged seed rules (never fail closed on a
    harness copy without guards)."""
    from ci_lab.guards.domains.order_support import GUARDS_DIR

    candidate = Path(harness) / GUARDS_SUBDIR
    if candidate.is_dir() and any(candidate.glob("*.yaml")):
        return candidate.resolve()
    return Path(GUARDS_DIR).resolve()


def install(harness: Path | str, tool_policies: Mapping[str, bool]) -> Installed:
    """The process-wide guard install for ``harness`` (cached by guards dir, mode, sink and policies)."""
    global install_count
    gdir = guards_dir(harness)
    key = (str(gdir), (os.environ.get(GUARDS_ENV) or "").strip().lower(), os.environ.get(DECISIONS_ENV) or "",
           os.environ.get(RUN_DIR_ENV) or "", tuple(sorted(tool_policies.items())))
    with _install_lock:
        found = _installs.get(key)
        if found is not None:
            return found
        from ci_lab.guards import install_guards
        from ci_lab.guards.domains.order_support import load_order_support_bundle

        bundle, degraded = load_order_support_bundle(gdir)
        writer = decision_writer()
        guards = install_guards(bundle=bundle, degraded=degraded, tool_policies=dict(tool_policies), sink=writer)
        install_count += 1
        found = _installs[key] = Installed(guards=guards, writer=writer, guards_dir=gdir)
        return found


def reset() -> None:
    """Forget installs and conversations (tests)."""
    global install_count
    with _install_lock:
        _installs.clear()
        install_count = 0
    with _conv_lock:
        _conversations.clear()


# ------------------------------------------------------------------ conversations

@dataclass
class Conversation:
    session: Any           # agent_framework.AgentSession
    flushed: int = 0       # recorder steps already turned into call/opportunity records


_conversations: OrderedDict[str, Conversation] = OrderedDict()
_conv_lock = threading.Lock()


def _key(message: str, history: Sequence[Mapping[str, Any]] | None) -> tuple[str, int]:
    users = [str(t.get("content") or "") for t in history or () if t.get("role") == "user"]
    first = users[0] if users else str(message)
    bound = _case.get()
    scope = f"case:{bound[0]}:{bound[1]}" if bound else "proc"
    return f"{scope}:{hashlib.sha256(first.encode()).hexdigest()}", len(users)


def conversation(message: str, history: Sequence[Mapping[str, Any]] | None) -> Conversation:
    """The persistent session for this conversation; a first turn always starts a new one.

    Keyed by the bound ASSERT case run (if any) and the conversation's first user message.
    """
    key, user_turns = _key(message, history)
    with _conv_lock:
        conv = _conversations.get(key) if user_turns > 1 else None
        if conv is None:
            conv = _conversations[key] = Conversation(AgentSession(session_id=f"os-{uuid.uuid4().hex}"))
            while len(_conversations) > MAX_CONVERSATIONS:
                _conversations.popitem(last=False)
        _conversations.move_to_end(key)
        return conv


class TranscriptProvider(HistoryProvider):
    """Loads the text-seeded prior turns before the current input; stores nothing (ASSERT owns history)."""

    def __init__(self, messages: Sequence[Message]) -> None:
        super().__init__(TRANSCRIPT_SOURCE, load_messages=True, store_inputs=False, store_outputs=False)
        self._messages = list(messages)

    async def get_messages(self, session_id: str | None, *, state: dict[str, Any] | None = None,
                           **kwargs: Any) -> list[Message]:
        return list(self._messages)

    async def save_messages(self, session_id: str | None, messages: Sequence[Message], *,
                            state: dict[str, Any] | None = None, **kwargs: Any) -> None:
        return None


def transcript(messages: Sequence[Message]) -> TranscriptProvider:
    return TranscriptProvider(messages)


# ------------------------------------------------------------------ paired-eval records

def flush(installed: Installed, conv: Conversation) -> int:
    """Write ``call``/``opportunity`` records for this conversation's new guard steps; returns the count.

    An opportunity is one guarded attempt: each tool call (``on: tool_call``, ``target: <tool>``) and
    each final response (``on: response``, ``target: "*"``). No-op without ``$CI_GUARD_DECISIONS``.
    """
    writer = installed.writer
    runtime = installed.guards.runtime
    state = runtime.open(conv.session.session_id)
    steps = state.recorder.steps
    new, conv.flushed = steps[conv.flushed:], len(steps)
    if writer is None or not writer.per_case:
        return 0
    records: list[dict[str, Any]] = []
    for step in new:
        if step.kind == "tool_call" and step.tool:
            records.append({"kind": "call", "tool": step.tool, "step_index": step.i,
                            "side_effect": runtime.is_side_effect(step.tool),
                            "blocked": True if step.status == "blocked" else None})
            records.append({"kind": "opportunity", "n": 1, "on": "tool_call", "target": step.tool,
                            "step_index": step.i})
        elif step.kind == "response":
            records.append({"kind": "opportunity", "n": 1, "on": "response", "target": "*",
                            "step_index": step.i})
    writer.write(records)
    return len(records)
