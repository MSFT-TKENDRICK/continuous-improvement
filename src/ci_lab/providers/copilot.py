"""GitHub Copilot SDK inference provider for Microsoft Agent Framework.

``CopilotChatClient`` is a MAF ``BaseChatClient`` that uses the GitHub Copilot SDK
(``github-copilot-sdk``) as a plain model endpoint: built-in Copilot tools are
disabled, the system prompt is replaced, and MAF tools are executed by MAF.

Primary path, the *suspended tool bridge*: each MAF tool is registered as a
Copilot ``Tool`` whose handler parks on a future. When the model asks for a batch
of tool calls, ``_inner_get_response`` returns them to MAF as ``function_call``
contents. MAF's ``FunctionInvocationLayer`` runs the tools (with middleware,
approvals, and telemetry) and calls back with ``function_result`` contents. The
bridge then resolves the parked futures by call id and keeps driving the same
Copilot session.

Fallback path, *transcript replay*: a fresh session receives the full MAF
history, rendered as a transcript. This is used when no live session matches
(after a crash, resume, cache miss, or continuity mismatch) and for calls that
have no tools.

Streaming (``stream=True``, which AG-UI uses) is *buffered*: the turn runs exactly
like the non-stream path and the finished response is replayed as one
``ChatResponseUpdate`` per message, so tool calls, approval requests and usage
survive unchanged. There are no token-level deltas.

Authentication is always ambient (logged-in user / ``gh``). The client never
accepts or logs tokens.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    FunctionTool,
    Message,
    ResponseStream,
    UsageDetails,
    normalize_tools,
)
from agent_framework.exceptions import ChatClientException, ChatClientInvalidResponseException
from agent_framework.observability import ChatTelemetryLayer

from ci_lab import obs
from ci_lab.providers.streaming import buffered_response_stream

logger = logging.getLogger(__name__)

__all__ = [
    "CopilotChatClient",
    "CopilotSessionError",
    "CopilotTimeoutError",
    "IGNORED_OPTIONS",
    "copilot_scope",
    "service_env",
    "session_scope",
]

session_scope: ContextVar[str | None] = ContextVar("ci_lab.providers.copilot.session_scope", default=None)
"""Ambient run/rollout/conversation scope id used to key Copilot sessions."""


@contextlib.contextmanager
def copilot_scope(scope_id: str) -> Iterator[str]:
    """Bind ``scope_id`` as the Copilot session scope for the enclosed code (and tasks it spawns)."""
    token = session_scope.set(str(scope_id))
    try:
        yield str(scope_id)
    finally:
        session_scope.reset(token)


def service_env() -> dict[str, str]:
    """Environment for a long-lived child service: the current env minus any pinned W3C trace context."""
    return {k: v for k, v in os.environ.items() if k not in (obs.TRACEPARENT_ENV, "TRACESTATE")}


IGNORED_OPTIONS = ("temperature", "top_p", "seed", "frequency_penalty", "presence_penalty", "max_tokens",
                   "stop", "logit_bias", "top_k")
"""Chat options with no Copilot SDK equivalent. They are accepted, ignored, and reported."""

_DISABLED_TOOL_TEXT = "Tool calls are disabled for this turn; answer without calling tools."
_ABORTED_TOOL_TEXT = "The tool call was cancelled because the session was aborted."
_MISSING_RESULT_TEXT = "No result was provided for this tool call."


class CopilotTimeoutError(ChatClientException, TimeoutError):
    """Copilot did not finish a turn within ``timeout_s``. The session was aborted and destroyed."""


class CopilotSessionError(ChatClientException):
    """The Copilot session reported an error or was aborted."""


@dataclass
class _Pending:
    future: asyncio.Future[Any]
    name: str
    arguments: Any


@dataclass(eq=False)
class _Live:
    """One Copilot session plus its bridge state."""

    key: tuple[str, str, str]
    tool_sig: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: asyncio.Queue[tuple[str, Any]] = field(default_factory=asyncio.Queue)
    session: Any = None
    unsubscribe: Callable[[], None] | None = None
    pending: dict[str, _Pending] = field(default_factory=dict)
    batch: list[str] = field(default_factory=list)
    state: str = "new"  # new | running | waiting_tools | idle | dead
    expected_prefix: str = ""
    expected_len: int = 0
    last_used: float = field(default_factory=time.monotonic)
    keep: bool = True
    tools_enabled: bool = True
    turns: int = 0

    @property
    def session_id(self) -> str:
        return str(getattr(self.session, "session_id", "") or "")


@dataclass
class _Outcome:
    kind: str  # "tools" | "final"
    text: str = ""
    calls: list[Any] = field(default_factory=list)


@dataclass
class _CallCtx:
    """Per ``get_response`` bookkeeping: usage aggregation and model-request records."""

    scope: str
    path: str
    request: dict[str, Any]
    usages: list[Any] = field(default_factory=list)
    served_model: str | None = None
    finish_reason: str | None = None
    unpaired_usage: dict[str, Any] = field(default_factory=dict)
    unpaired_msgs: dict[str, Any] = field(default_factory=dict)
    turn_t0: float = field(default_factory=time.monotonic)
    turn_index: int = 0


def _approve_all(request: Any, invocation: Any = None) -> Any:
    try:
        from copilot.session import PermissionDecisionApproveOnce

        return PermissionDecisionApproveOnce()
    except Exception:  # pragma: no cover - older SDKs
        return {"kind": "approved"}


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:32]


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, Content):
        return result.text or _json(result.to_dict())
    if isinstance(result, (list, tuple)) and all(isinstance(r, Content) for r in result):
        return "\n".join(_result_text(r) for r in result)
    return _json(result)


def _norm_messages(messages: Sequence[Message]) -> list[Any]:
    """Stable, lossy view of a message list used only to validate conversation continuity."""
    out: list[Any] = []
    for m in messages:
        parts: list[Any] = []
        for c in m.contents:
            if c.type == "text" and c.text:
                parts.append(["t", c.text.strip()])
            elif c.type == "function_call":
                parts.append(["c", c.call_id])
            elif c.type == "function_result":
                parts.append(["r", c.call_id])
        if parts:
            out.append([str(m.role), parts])
    return out


def _prefix_hash(messages: Sequence[Message]) -> str:
    return _sha(_norm_messages(messages))


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```$", t, re.S)
    return m.group(1).strip() if m else t


def _schema_from_format(fmt: Any) -> tuple[bool, dict[str, Any] | None]:
    """Return ``(json_required, schema)`` for a MAF/OpenAI ``response_format`` value."""
    if fmt is None:
        return False, None
    if isinstance(fmt, type):
        from pydantic import BaseModel

        if issubclass(fmt, BaseModel):
            return True, fmt.model_json_schema()
        return False, None
    if isinstance(fmt, Mapping):
        if fmt.get("type") == "json_object":
            return True, None
        if fmt.get("type") == "json_schema" or "json_schema" in fmt:
            spec = fmt.get("json_schema") or {}
            return True, dict(spec.get("schema") or {}) or None
        if fmt.get("type") == "text":
            return False, None
        return True, dict(fmt)
    return False, None


