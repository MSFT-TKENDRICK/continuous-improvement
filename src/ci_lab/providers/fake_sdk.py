"""In-process fake of the GitHub Copilot SDK client/session for offline tests.

It emits real ``copilot.session_events`` objects (``assistant.usage``,
``assistant.message`` with ``tool_requests``, ``session.idle``, ``session.error``)
and invokes registered ``Tool`` handlers the way the SDK does when it sees
``external_tool.requested``. Each session plays a script. Every ``send`` consumes
steps up to and including the next final answer. A step can be:

* ``str``: a final assistant message, followed by ``session.idle``.
* a list of :class:`FakeCall`: one assistant turn requesting tools in parallel.
  The handlers' results are recorded in ``session.tool_results``.
* :class:`Hang`: never answers (use it to exercise timeouts).
* :class:`Fail`: emits ``session.error``.
* a callable ``(session) -> step``, evaluated lazily so it can see earlier tool results.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageToolRequest,
    AssistantUsageCopilotUsage,
    AssistantUsageData,
    SessionErrorData,
    SessionEvent,
    SessionEventType,
    SessionIdleData,
)
from copilot.tools import ToolInvocation

__all__ = ["FakeCall", "FakeCopilotClient", "FakeSession", "Fail", "Hang"]


@dataclass
class FakeCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass
class Hang:
    """Step that never completes."""


@dataclass
class Fail:
    message: str = "boom"
    error_type: str = "model_error"


_EVENT_TYPES = {
    AssistantMessageData: SessionEventType.ASSISTANT_MESSAGE,
    AssistantUsageData: SessionEventType.ASSISTANT_USAGE,
    SessionIdleData: SessionEventType.SESSION_IDLE,
    SessionErrorData: SessionEventType.SESSION_ERROR,
}


class FakeSession:
    _ids = itertools.count(1)

    def __init__(self, client: FakeCopilotClient, kwargs: dict[str, Any], script: list[Any]) -> None:
        self.client = client
        self.kwargs = kwargs
        self.session_id = f"fake-session-{next(self._ids)}"
        self.tools = {t.name: t for t in kwargs.get("tools") or []}
        self.script = list(script)
        self.prompts: list[str] = []
        self.tool_results: list[tuple[str, str, Any]] = []
        self.aborted = False
        self.disconnected = False
        self._handlers: list[Callable[[SessionEvent], None]] = []
        self._task: asyncio.Task[None] | None = None
        self._calls = itertools.count(1)
        self._api = itertools.count(1)

    # -- SDK surface
    def on(self, handler: Callable[[SessionEvent], None]) -> Callable[[], None]:
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler) if handler in self._handlers else None

    async def send(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        self._task = asyncio.get_running_loop().create_task(self._play())
        return f"msg-{uuid.uuid4().hex[:8]}"

    async def abort(self) -> None:
        self.aborted = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def disconnect(self) -> None:
        self.disconnected = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # -- helpers
    def emit(self, data: Any) -> None:
        event = SessionEvent(data=data, id=uuid.uuid4(), timestamp=datetime.now(UTC), type=_EVENT_TYPES[type(data)])
        for handler in list(self._handlers):
            handler(event)

    def _usage(self, api: str, finish: str) -> AssistantUsageData:
        u = dict(self.client.usage)
        return AssistantUsageData(
            model=self.client.served_model, api_call_id=api, input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0), cache_read_tokens=u.get("cache_read_tokens", 0),
            cache_write_tokens=u.get("cache_write_tokens", 0), reasoning_tokens=u.get("reasoning_tokens", 0),
            cost=u.get("cost", 0.0), copilot_usage=AssistantUsageCopilotUsage(total_nano_aiu=u.get("nano_aiu", 0)),
            duration=timedelta(milliseconds=u.get("duration_ms", 5)), finish_reason=finish)

    async def _play(self) -> None:
        while self.script:
            step = self.script.pop(0)
            if callable(step) and not isinstance(step, (Hang, Fail)):
                step = step(self)
            api = f"api-{self.session_id}-{next(self._api)}"
            if isinstance(step, Hang):
                await asyncio.Event().wait()
            if isinstance(step, Fail):
                self.emit(SessionErrorData(error_type=step.error_type, message=step.message))
                return
            if isinstance(step, str):
                self.emit(self._usage(api, "stop"))
                self.emit(AssistantMessageData(content=step, message_id=uuid.uuid4().hex, api_call_id=api,
                                               model=self.client.served_model))
                self.emit(SessionIdleData(aborted=False))
                return
            calls = [FakeCall(c.name, dict(c.arguments), c.call_id or f"call_{next(self._calls)}") for c in step]
            self.emit(self._usage(api, "tool_calls"))
            self.emit(AssistantMessageData(
                content="", message_id=uuid.uuid4().hex, api_call_id=api, model=self.client.served_model,
                tool_requests=[AssistantMessageToolRequest(name=c.name, tool_call_id=c.call_id,
                                                           arguments=dict(c.arguments)) for c in calls]))
            await asyncio.sleep(0)
            # Like the SDK, handlers run as independent tasks: aborting the turn does not cancel them.
            tasks = [asyncio.ensure_future(self._invoke(c)) for c in calls]
            await asyncio.wait(tasks)
        self.emit(SessionIdleData(aborted=False))

    async def _invoke(self, call: FakeCall) -> Any:
        tool = self.tools[call.name]
        result = tool.handler(ToolInvocation(session_id=self.session_id, tool_call_id=call.call_id,
                                             tool_name=call.name, arguments=dict(call.arguments)))
        if inspect.isawaitable(result):
            result = await result
        self.tool_results.append((call.call_id, call.name, result))
        return result


class FakeCopilotClient:
    """Drop-in for ``copilot.CopilotClient``. ``script`` may be a list (copied per session)
    or a callable ``(create_session_kwargs) -> list`` for per-session scripts."""

    def __init__(self, script: Sequence[Any] | Callable[[dict[str, Any]], Sequence[Any]] = ("ok",), *,
                 served_model: str = "gpt-fake-served", usage: Mapping[str, Any] | None = None) -> None:
        self.script = script
        self.served_model = served_model
        self.usage = dict(usage or {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 2,
                                    "nano_aiu": 1000, "cost": 0.0, "duration_ms": 7})
        self.sessions: list[FakeSession] = []
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def create_session(self, **kwargs: Any) -> FakeSession:
        script = self.script(kwargs) if callable(self.script) else self.script
        session = FakeSession(self, kwargs, list(script))
        self.sessions.append(session)
        return session
