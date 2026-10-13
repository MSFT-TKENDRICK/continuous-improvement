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


class CopilotChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    """MAF chat client backed by the GitHub Copilot SDK (see module docstring).

    One client is bound to one asyncio event loop at a time. If it is reused from a
    new loop (for example successive ``asyncio.run`` calls), it drops its sessions and
    restarts the SDK client it owns.
    """

    OTEL_PROVIDER_NAME = "github.copilot"

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = None,
        base_directory: str | None = None,
        timeout_s: float = 300,
        session_ttl_s: float = 900,
        sdk_client: Any | None = None,
        on_model_request: Callable[[dict[str, Any]], None] | None = None,
        session_scope: Callable[[], str] | None = None,
        verify_model: bool = True,
        **kwargs: Any,
    ) -> None:
        if not model:
            raise ValueError("CopilotChatClient requires a model")
        super().__init__(**kwargs)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = float(timeout_s)
        self.session_ttl_s = float(session_ttl_s)
        self.on_model_request = on_model_request
        self._scope_fn = session_scope
        self._base_directory = base_directory
        self._owns_base_dir = False
        self._sdk = sdk_client
        self._owns_sdk = sdk_client is None
        self._sdk_started = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lives: set[_Live] = set()
        self._calls: dict[str, list[_Live]] = {}
        self.last_served_model: str | None = None
        self.verify_model = verify_model
        self._model_checked = False

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> CopilotChatClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Abort every session and stop the owned Copilot SDK client."""
        for live in list(self._lives):
            await self._kill(live, "client closed")
        if self._sdk is not None and self._owns_sdk and self._sdk_started:
            with contextlib.suppress(Exception):
                await self._sdk.stop()
            self._sdk = None
        self._sdk_started = False
        if self._owns_base_dir and self._base_directory:
            shutil.rmtree(self._base_directory, ignore_errors=True)
            self._base_directory = None
            self._owns_base_dir = False

    async def reap(self, now: float | None = None) -> int:
        """Abort and destroy idle or orphaned sessions unused for ``session_ttl_s``. Returns how many were reaped."""
        now = time.monotonic() if now is None else now
        stale = [lv for lv in self._lives if not lv.lock.locked() and now - lv.last_used > self.session_ttl_s]
        for live in stale:
            await self._kill(live, "ttl expired")
        return len(stale)

    @property
    def live_sessions(self) -> int:
        return len(self._lives)

    async def _ensure_sdk(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            # Sessions from a previous loop cannot be driven or aborted here. Drop them.
            for live in list(self._lives):
                self._forget(live)
            if self._owns_sdk:
                self._sdk = None
            self._sdk_started = False
        self._loop = loop
        if self._sdk is None:
            from copilot import CopilotClient

            if not self._base_directory:
                self._base_directory = tempfile.mkdtemp(prefix="ci-lab-copilot-")
                self._owns_base_dir = True
            # The Copilot CLI is a long-lived child: never pin a startup TRACEPARENT on it (design §12.5).
            self._sdk = CopilotClient(mode="empty", base_directory=self._base_directory, log_level="error",
                                      env=service_env())
            self._owns_sdk = True
        if not self._sdk_started:
            start = getattr(self._sdk, "start", None)
            if start is not None:
                await start()
            self._sdk_started = True
        if self.verify_model and not self._model_checked:
            await self.check_model()
        return self._sdk

    async def check_model(self) -> None:
        """Fail fast with the available ids if ``self.model`` is not in the SDK's ``list_models()``.

        Raises :class:`~ci_lab.providers.models.ModelPreflightError`. SDKs without ``list_models``
        (test fakes) and listing failures are skipped with a log line: the session would report them.
        """
        from ci_lab.providers.models import (
            ModelPreflightError,
            ModelUse,
            check_copilot_models,
        )

        self._model_checked = True
        sdk = await self._ensure_sdk()
        list_models = getattr(sdk, "list_models", None)
        if list_models is None:
            return
        try:
            await check_copilot_models([ModelUse(self.model, "CopilotChatClient")], list_models=list_models)
        except ModelPreflightError as exc:
            if exc.__cause__ is None:
                self._model_checked = False
                raise
            logger.warning("copilot model check skipped: %s", exc)

    # ------------------------------------------------------------------ MAF entry point
    def _inner_get_response(self, *, messages: Sequence[Message], stream: bool, options: Mapping[str, Any],
                            **kwargs: Any) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        msgs, opts = list(messages), dict(options or {})
        if stream:
            # Buffered: the turn completes like the non-stream path, then replays as updates.
            return buffered_response_stream(self, lambda: self._get_response(msgs, opts),
                                            response_format=opts.get("response_format"))
        return self._get_response(msgs, opts)

    def _scope(self) -> str:
        if self._scope_fn is not None:
            return str(self._scope_fn())
        return session_scope.get() or "default"

    async def _get_response(self, messages: list[Message], options: dict[str, Any]) -> ChatResponse:
        await self._ensure_sdk()
        await self.reap()
        tools = [t for t in normalize_tools(options.get("tools")) if isinstance(t, FunctionTool)]
        json_required, schema = _schema_from_format(options.get("response_format"))
        system = self._system_text(messages, options, json_required, schema)
        tool_sig = _sha([system, [[t.name, t.description, t.parameters()] for t in tools]])
        scope = self._scope()
        key = (scope, self.model, self._fingerprint(messages, options))
        tools_enabled = options.get("tool_choice") != "none"
        ignored = {k: options[k] for k in IGNORED_OPTIONS if options.get(k) is not None}

        n_trailing = 0
        for m in reversed(messages):
            if m.contents and all(c.type == "function_result" for c in m.contents):
                n_trailing += 1
            else:
                break
        results = [c for m in messages[len(messages) - n_trailing:] for c in m.contents]
        base = messages[: len(messages) - n_trailing]

        live: _Live | None = None
        path = "replay"
        prompt: str | None = None
        if results and tools:
            live = self._find_waiting(key, results, base)
            if live is not None:
                path = "bridge"
        elif tools and not results:
            live, prompt = self._find_idle(key, tool_sig, messages)
            if live is not None:
                path = "followup"

        ctx = _CallCtx(scope=scope, path=path, request={
            "path": path, "message_count": len(messages), "tool_count": len(tools),
            "tool_results": len(results), "structured": json_required})

        if live is None:
            if not tools and len([m for m in messages if str(m.role) != "system"]) == 1:
                ctx.path = ctx.request["path"] = "single"
            return await self._run_fresh(messages, options, tools, system, tool_sig, key, ctx, tools_enabled,
                                         json_required, schema, ignored)

        async with live.lock:
            if live.state == "dead":  # reaped/killed while we waited for the lock
                return await self._run_fresh(messages, options, tools, system, tool_sig, key, ctx, tools_enabled,
                                             json_required, schema, ignored)
            live.state = "running"
            live.tools_enabled = tools_enabled
            if path == "bridge":
                self._resolve_results(live, results)
            else:
                try:
                    await live.session.send(prompt or "")
                except BaseException:
                    await asyncio.shield(self._kill(live, "send failed"))
                    raise
            return await self._drive_to_response(live, messages, options, ctx, json_required, schema, ignored)

    # ------------------------------------------------------------------ session lookup
    def _fingerprint(self, messages: Sequence[Message], options: Mapping[str, Any]) -> str:
        if options.get("conversation_id"):
            return f"conv:{options['conversation_id']}"
        first = next((m for m in messages if str(m.role) != "system"), None)
        return _sha([options.get("instructions") or "", _norm_messages([first]) if first else []])

    def _find_waiting(self, key: tuple[str, str, str], results: list[Content], base: list[Message]) -> _Live | None:
        call_ids = [c.call_id for c in results]
        candidates = [lv for lv in self._calls.get(call_ids[0], []) if lv.key == key and lv.state == "waiting_tools"]
        if not candidates:
            return None
        prefix = _prefix_hash(base)
        matches = [lv for lv in candidates if lv.expected_prefix == prefix]
        if len(matches) == 1 and set(call_ids) <= set(matches[0].batch):
            return matches[0]
        # Continuity cannot be proven. Abort the stale sessions in the background and replay.
        for lv in candidates:
            logger.info("copilot bridge continuity mismatch for session %s; falling back to replay", lv.session_id)
            asyncio.ensure_future(self._kill(lv, "continuity mismatch"))
        return None

    def _find_idle(self, key: tuple[str, str, str], tool_sig: str,
                   messages: list[Message]) -> tuple[_Live | None, str | None]:
        for lv in self._lives:
            if lv.key != key or lv.state != "idle" or lv.tool_sig != tool_sig or lv.lock.locked():
                continue
            if lv.expected_len >= len(messages) or _prefix_hash(messages[: lv.expected_len]) != lv.expected_prefix:
                continue
            tail = messages[lv.expected_len:]
            if not all(str(m.role) == "user" for m in tail):
                continue
            return lv, "\n\n".join(self._message_text(m) for m in tail)
        return None, None

    # ------------------------------------------------------------------ fresh / replay sessions
    async def _run_fresh(self, messages: list[Message], options: dict[str, Any], tools: list[FunctionTool],
                         system: str, tool_sig: str, key: tuple[str, str, str], ctx: _CallCtx,
                         tools_enabled: bool, json_required: bool, schema: dict[str, Any] | None,
                         ignored: dict[str, Any]) -> ChatResponse:
        live = _Live(key=key, tool_sig=tool_sig, keep=bool(tools))
        live.tools_enabled = tools_enabled
        await live.lock.acquire()
        try:
            try:
                await self._open_session(live, tools, system)
                live.state = "running"
                await live.session.send(self._render_prompt(messages))
            except BaseException:
                await asyncio.shield(self._kill(live, "send failed"))
                raise
            return await self._drive_to_response(live, messages, options, ctx, json_required, schema, ignored)
        finally:
            live.lock.release()

    async def _open_session(self, live: _Live, tools: list[FunctionTool], system: str) -> None:
        from copilot import ToolSet
        from copilot.tools import Tool

        loop = asyncio.get_running_loop()

        def on_event(event: Any) -> None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(live.queue.put_nowait, ("event", event))

        def make_handler(name: str) -> Callable[[Any], Awaitable[Any]]:
            async def handler(invocation: Any) -> Any:
                fut: asyncio.Future[Any] = loop.create_future()
                live.pending[invocation.tool_call_id] = _Pending(fut, name, invocation.arguments)
                live.queue.put_nowait(("tool", invocation.tool_call_id))
                return await fut

            return handler

        sdk_tools = [Tool(name=t.name, description=t.description or t.name, handler=make_handler(t.name),
                          parameters=t.parameters(), skip_permission=True) for t in tools]
        kw: dict[str, Any] = {
            "model": self.model,
            "tools": sdk_tools,
            "available_tools": ToolSet().add_custom("*"),
            "system_message": {"mode": "replace", "content": system},
            "on_permission_request": _approve_all,
        }
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        sdk = await self._ensure_sdk()
        live.session = await sdk.create_session(**kw)
        live.unsubscribe = live.session.on(on_event)
        self._lives.add(live)

    # ------------------------------------------------------------------ driving a session
    def _resolve_results(self, live: _Live, results: list[Content]) -> None:
        from copilot.tools import ToolResult

        by_id = {c.call_id: c for c in results}
        for call_id in list(live.batch):
            pend = live.pending.pop(call_id, None)
            self._unregister(live, call_id)
            if pend is None or pend.future.done():
                continue
            c = by_id.get(call_id)
            if c is None:
                pend.future.set_result(ToolResult(text_result_for_llm=_MISSING_RESULT_TEXT, result_type="failure",
                                                  error="missing result"))
            elif c.exception is not None:
                text = _result_text(c.result) or f"Error: {c.exception}"
                pend.future.set_result(ToolResult(text_result_for_llm=text, result_type="failure",
                                                  error=str(c.exception)))
            else:
                pend.future.set_result(ToolResult(text_result_for_llm=_result_text(c.result), result_type="success"))
        live.batch = []

    async def _drive_to_response(self, live: _Live, messages: list[Message], options: dict[str, Any],
                                 ctx: _CallCtx, json_required: bool, schema: dict[str, Any] | None,
                                 ignored: dict[str, Any]) -> ChatResponse:
        deadline = time.monotonic() + self.timeout_s
        ctx.turn_t0 = time.monotonic()
        try:
            outcome = await self._drive(live, ctx, deadline)
            value_text = outcome.text
            if outcome.kind == "final" and json_required:
                value_text, err = self._check_json(outcome.text, schema)
                if err is not None:
                    ctx.request["repair"] = True
                    await live.session.send(
                        f"Your previous reply was not valid: {err}. Reply again with only the JSON value, "
                        "no prose and no code fences.")
                    outcome = await self._drive(live, ctx, deadline)
                    if outcome.kind == "final":
                        value_text, err = self._check_json(outcome.text, schema)
                        if err is not None:
                            raise ChatClientInvalidResponseException(
                                f"Copilot reply did not match the requested JSON schema: {err}")
        except asyncio.CancelledError:
            await asyncio.shield(self._kill(live, "cancelled"))
            raise
        except TimeoutError as exc:
            await self._kill(live, "timeout")
            self._flush_turns(ctx, live, status="timeout")
            raise CopilotTimeoutError(
                f"Copilot did not respond within {self.timeout_s:g}s (session {live.session_id or '?'} aborted)"
            ) from exc
        except BaseException:
            await self._kill(live, "error")
            self._flush_turns(ctx, live, status="error")
            raise
        self._flush_turns(ctx, live)
        live.last_used = time.monotonic()

        if outcome.kind == "tools":
            contents: list[Content] = []
            if outcome.text:
                contents.append(Content.from_text(outcome.text))
            for req in outcome.calls:
                args = req.arguments
                contents.append(Content.from_function_call(
                    call_id=req.tool_call_id, name=req.name,
                    arguments=args if isinstance(args, (str, Mapping)) or args is None else _json(args)))
            reply = [Message(role="assistant", contents=contents)]
            live.batch = [req.tool_call_id for req in outcome.calls]
            for call_id in live.batch:
                self._calls.setdefault(call_id, []).append(live)
            live.expected_prefix = _prefix_hash(messages + reply)
            live.expected_len = len(messages) + 1
            live.state = "waiting_tools"
            finish = "tool_calls"
        else:
            reply = [Message(role="assistant", contents=[Content.from_text(value_text)])]
            if live.keep:
                live.expected_prefix = _prefix_hash(messages + reply)
                live.expected_len = len(messages) + 1
                live.state = "idle"
            else:
                await self._kill(live, "done")
            finish = {"tool_calls": "stop", None: "stop"}.get(ctx.finish_reason, ctx.finish_reason)
            if finish not in ("stop", "length", "content_filter", "tool_calls"):
                finish = "stop"

        served = ctx.served_model or self.model
        self.last_served_model = served
        props: dict[str, Any] = {"copilot_path": ctx.path, "copilot_session_id": live.session_id,
                                 "requested_model": self.model}
        costs = [u.cost for u in ctx.usages if getattr(u, "cost", None) is not None]
        if costs:
            props["copilot_cost"] = float(sum(costs))
        if ignored:
            props["ignored_options"] = ignored
        fmt = options.get("response_format")
        return ChatResponse(
            messages=reply,
            response_id=f"copilot-{uuid.uuid4().hex[:16]}",
            model=served,
            finish_reason=finish,
            usage_details=self._usage(ctx),
            response_format=fmt if outcome.kind == "final" and json_required else None,
            additional_properties=props,
        )

    async def _drive(self, live: _Live, ctx: _CallCtx, deadline: float) -> _Outcome:
        from copilot.session_events import (AssistantMessageData, AssistantUsageData, SessionErrorData,
                                            SessionIdleData)
        from copilot.tools import ToolResult

        batch: list[Any] | None = None
        batch_text = ""
        last_text = ""
        while True:
            if batch is not None and all(
                    r.tool_call_id in live.pending and not live.pending[r.tool_call_id].future.done() for r in batch):
                return _Outcome("tools", text=batch_text, calls=batch)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            kind, item = await asyncio.wait_for(live.queue.get(), remaining)
            if kind == "tool":
                pend = live.pending.get(item)
                if pend is not None and not live.tools_enabled and not pend.future.done():
                    live.pending.pop(item, None)
                    pend.future.set_result(ToolResult(text_result_for_llm=_DISABLED_TOOL_TEXT, result_type="failure",
                                                      error="tools disabled"))
                continue
            data = item.data
            if isinstance(data, AssistantUsageData):
                if data.parent_tool_call_id:
                    continue
                ctx.usages.append(data)
                ctx.served_model = data.model or ctx.served_model
                ctx.finish_reason = data.finish_reason or ctx.finish_reason
                self._pair(ctx, live, usage=data)
            elif isinstance(data, AssistantMessageData):
                if data.parent_tool_call_id:
                    continue
                live.turns += 1
                if data.model:
                    ctx.served_model = ctx.served_model or data.model
                self._pair(ctx, live, message=data)
                if data.tool_requests and live.tools_enabled:
                    batch = list(data.tool_requests)
                    batch_text = (data.content or "").strip()
                    last_text = ""
                elif data.content and data.content.strip():
                    last_text = data.content.strip()
            elif isinstance(data, SessionErrorData):
                raise CopilotSessionError(f"Copilot session error ({data.error_type}): {data.message}")
            elif isinstance(data, SessionIdleData):
                if data.aborted:
                    raise CopilotSessionError("Copilot session was aborted")
                return _Outcome("final", text=last_text)

    # ------------------------------------------------------------------ usage / telemetry
    def _pair(self, ctx: _CallCtx, live: _Live, *, usage: Any = None, message: Any = None) -> None:
        if usage is not None:
            api = usage.api_call_id or f"anon-{id(usage)}"
            msg = ctx.unpaired_msgs.pop(api, None) if usage.api_call_id else None
            if msg is None and not usage.api_call_id and ctx.unpaired_msgs:
                msg = ctx.unpaired_msgs.pop(next(iter(ctx.unpaired_msgs)))
            if msg is None:
                ctx.unpaired_usage[api] = usage
                return
            self._emit(ctx, live, msg, usage)
        elif message is not None:
            api = message.api_call_id or f"anon-{id(message)}"
            use = ctx.unpaired_usage.pop(api, None) if message.api_call_id else None
            if use is None and not message.api_call_id and ctx.unpaired_usage:
                use = ctx.unpaired_usage.pop(next(iter(ctx.unpaired_usage)))
            if use is None:
                ctx.unpaired_msgs[api] = message
                return
            self._emit(ctx, live, message, use)

    def _flush_turns(self, ctx: _CallCtx, live: _Live, status: str = "ok") -> None:
        for msg in list(ctx.unpaired_msgs.values()):
            self._emit(ctx, live, msg, None, status=status)
        for use in list(ctx.unpaired_usage.values()):
            self._emit(ctx, live, None, use, status=status)
        ctx.unpaired_msgs.clear()
        ctx.unpaired_usage.clear()

    def _emit(self, ctx: _CallCtx, live: _Live, msg: Any, use: Any, status: str = "ok") -> None:
        now = time.monotonic()
        latency_ms = (use.duration.total_seconds() * 1000 if use is not None and use.duration is not None
                      else (now - ctx.turn_t0) * 1000)
        ctx.turn_t0 = now
        ctx.turn_index += 1
        if self.on_model_request is None:
            return
        tool_calls = [{"name": r.name, "call_id": r.tool_call_id} for r in (getattr(msg, "tool_requests", None) or [])]
        text = (getattr(msg, "content", "") or "") if msg is not None else ""
        record = {
            "model": (getattr(use, "model", None) or getattr(msg, "model", None) or self.model),
            "requested_model": self.model,
            "server": "copilot",
            "scope": ctx.scope,
            "session_id": live.session_id,
            "api_call_id": getattr(use, "api_call_id", None) or getattr(msg, "api_call_id", None),
            "request": {**ctx.request, "turn": ctx.turn_index},
            "response": {"text": text[:500], "tool_calls": tool_calls},
            "latency_ms": round(latency_ms, 3),
            "usage": {
                "input_tokens": getattr(use, "input_tokens", None) or 0,
                "output_tokens": getattr(use, "output_tokens", None) or 0,
                "cache_read_tokens": getattr(use, "cache_read_tokens", None) or 0,
                "cache_write_tokens": getattr(use, "cache_write_tokens", None) or 0,
                "reasoning_tokens": getattr(use, "reasoning_tokens", None) or 0,
                "total_nano_aiu": (use.copilot_usage.total_nano_aiu
                                   if use is not None and use.copilot_usage is not None else 0),
                "cost": getattr(use, "cost", None) or 0.0,
            },
            "finish_reason": getattr(use, "finish_reason", None) or ("tool_calls" if tool_calls else "stop"),
            "status": "error" if getattr(use, "content_filter_triggered", False) else status,
        }
        try:
            self.on_model_request(record)
        except Exception:
            logger.warning("on_model_request callback failed", exc_info=True)

    @staticmethod
    def _usage(ctx: _CallCtx) -> UsageDetails | None:
        if not ctx.usages:
            return None

        def total(attr: str) -> int:
            return int(sum((getattr(u, attr, None) or 0) for u in ctx.usages))

        inp, out = total("input_tokens"), total("output_tokens")
        usage = UsageDetails(input_token_count=inp, output_token_count=out, total_token_count=inp + out,
                             cache_read_input_token_count=total("cache_read_tokens"),
                             cache_creation_input_token_count=total("cache_write_tokens"),
                             reasoning_output_token_count=total("reasoning_tokens"))
        aiu = sum(u.copilot_usage.total_nano_aiu for u in ctx.usages if u.copilot_usage is not None)
        usage["copilot_total_nano_aiu"] = int(aiu)  # type: ignore[typeddict-unknown-key]
        return usage

    # ------------------------------------------------------------------ teardown
    def _unregister(self, live: _Live, call_id: str) -> None:
        lst = self._calls.get(call_id)
        if lst and live in lst:
            lst.remove(live)
            if not lst:
                del self._calls[call_id]

    def _forget(self, live: _Live) -> None:
        live.state = "dead"
        for call_id in list(live.batch):
            self._unregister(live, call_id)
        live.batch = []
        self._lives.discard(live)
        if live.unsubscribe is not None:
            with contextlib.suppress(Exception):
                live.unsubscribe()
            live.unsubscribe = None

    async def _kill(self, live: _Live, reason: str) -> None:
        """Resolve every parked tool future, then abort and destroy the session."""
        if live.state == "dead" and live.session is None:
            return
        from copilot.tools import ToolResult

        self._forget(live)
        for pend in live.pending.values():
            if not pend.future.done():
                pend.future.set_result(ToolResult(text_result_for_llm=_ABORTED_TOOL_TEXT, result_type="failure",
                                                  error=f"aborted: {reason}"))
        live.pending.clear()
        session, live.session = live.session, None
        if session is None:
            return
        logger.debug("destroying copilot session %s (%s)", getattr(session, "session_id", "?"), reason)
        if reason != "done":
            with contextlib.suppress(Exception):
                await asyncio.wait_for(session.abort(), 10)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(session.disconnect(), 10)

    # ------------------------------------------------------------------ prompt rendering
    @staticmethod
    def _message_text(m: Message) -> str:
        return "\n".join(c.text for c in m.contents if c.type == "text" and c.text)

    def _system_text(self, messages: Sequence[Message], options: Mapping[str, Any], json_required: bool,
                     schema: dict[str, Any] | None) -> str:
        parts = [str(options["instructions"]).strip()] if options.get("instructions") else []
        parts += [t for m in messages if str(m.role) == "system" and (t := self._message_text(m).strip())]
        if json_required:
            if schema:
                parts.append("Respond with only a single JSON value (no prose, no Markdown code fences) that "
                             "validates against this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False))
            else:
                parts.append("Respond with only a single valid JSON object (no prose, no Markdown code fences).")
        return "\n\n".join(parts) or "You are a helpful assistant."

    def _render_prompt(self, messages: Sequence[Message]) -> str:
        convo = [m for m in messages if str(m.role) != "system"]
        if len(convo) == 1 and str(convo[0].role) == "user" and all(c.type == "text" for c in convo[0].contents):
            return self._message_text(convo[0])
        lines: list[str] = []
        for m in convo:
            role = str(m.role)
            for c in m.contents:
                if c.type == "text" and c.text:
                    lines.append(f"[{role}]\n{c.text}")
                elif c.type == "function_call":
                    lines.append(f"[assistant tool call {c.call_id}] {c.name}({_json(c.arguments)})")
                elif c.type == "function_result":
                    status = f" error: {c.exception}" if c.exception is not None else ""
                    lines.append(f"[tool result {c.call_id}{status}]\n{_result_text(c.result)}")
                elif c.type not in ("usage", "text_reasoning"):
                    lines.append(f"[{role} {c.type} content omitted]")
        return ("Here is the conversation so far. Tool calls shown in it have already been executed. "
                "Continue the conversation by writing the next assistant turn; call tools if you still need "
                "them.\n\n<transcript>\n" + "\n\n".join(lines) + "\n</transcript>")

    @staticmethod
    def _check_json(text: str, schema: dict[str, Any] | None) -> tuple[str, str | None]:
        cleaned = _strip_fences(text)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return cleaned, f"not valid JSON ({exc})"
        if schema:
            import jsonschema

            try:
                jsonschema.validate(value, schema)
            except jsonschema.ValidationError as exc:
                return cleaned, f"JSON does not match the schema ({exc.message})"
            except jsonschema.SchemaError:
                pass
        return cleaned, None
