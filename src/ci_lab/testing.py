"""Test doubles shared by every ci_lab module (no network, no Copilot, no .NET)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)
from agent_framework.observability import ChatTelemetryLayer

from ci_lab.contracts import RolloutKey


@dataclass
class Call:
    """A scripted assistant tool call."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


Step = str | Sequence[Call] | Callable[[Sequence[Message], Mapping[str, Any]], "str | Sequence[Call]"]


class FakeChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    """Scriptable MAF chat client. Each ``get_response`` consumes the next step:
    a str (final text), a list of :class:`Call` (tool calls), or a callable
    ``(messages, options) -> str | list[Call]``. When the script is exhausted it
    returns ``default``. Records every request in ``requests``."""

    OTEL_PROVIDER_NAME = "fake"

    def __init__(self, script: Sequence[Step] = (), *, model: str = "fake-model",
                 default: str = "ok", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.script = list(script)
        self.default = default
        self.requests: list[tuple[list[Message], dict[str, Any]]] = []

    async def _respond(self, messages: Sequence[Message], options: Mapping[str, Any]) -> ChatResponse:
        self.requests.append((list(messages), dict(options)))
        step: Any = self.script.pop(0) if self.script else self.default
        if callable(step):
            step = step(messages, options)
        if isinstance(step, str):
            contents: list[Any] = [step]
        else:
            n = len(self.requests)
            contents = [Content.from_function_call(call_id=c.call_id or f"call-{n}-{i}", name=c.name,
                                                   arguments=dict(c.arguments)) for i, c in enumerate(step)]
        return ChatResponse(messages=[Message(role="assistant", contents=contents)], model=self.model)

    def _inner_get_response(self, *, messages: Sequence[Message], stream: bool,
                            options: Mapping[str, Any], **kwargs: Any) -> Awaitable[ChatResponse]:
        if stream:
            raise NotImplementedError("FakeChatClient does not stream")
        return self._respond(messages, options)


class MemoryOutbox:
    """In-memory :class:`ci_lab.contracts.Outbox` (durable version lives in ci_lab.ledger.outbox)."""

    def __init__(self) -> None:
        self.done: dict[str, Any] = {}
        self.calls: list[str] = []

    def run_once(self, op: str, fn: Callable[[], Any], *, reconcile: Callable[[], Any | None] | None = None) -> Any:
        if op in self.done:
            return self.done[op]
        if reconcile is not None and (found := reconcile()) is not None:
            self.done[op] = found
            return found
        self.calls.append(op)
        self.done[op] = result = fn()
        return result

    async def arun_once(self, op: str, fn: Callable[[], Awaitable[Any]], *,
                        reconcile: Callable[[], Awaitable[Any | None]] | None = None) -> Any:
        if op in self.done:
            return self.done[op]
        if reconcile is not None and (found := await reconcile()) is not None:
            self.done[op] = found
            return found
        self.calls.append(op)
        self.done[op] = result = await fn()
        return result


class MemoryJournal:
    """In-memory :class:`ci_lab.contracts.RolloutJournal` with event-id dedupe."""

    def __init__(self) -> None:
        self.rollouts: dict[str, dict[str, Any]] = {}

    def start(self, key: RolloutKey, input: Mapping[str, Any]) -> None:
        self.rollouts.setdefault(key.rollout_id, {"input": dict(input), "events": {}, "status": "running"})

    def event(self, key: RolloutKey, event_type: str, data: Mapping[str, Any], *, event_id: str) -> None:
        self.rollouts[key.rollout_id]["events"].setdefault(
            event_id, {"event_id": event_id, "event_type": event_type, "data": dict(data)})

    def finish(self, key: RolloutKey, status: Literal["succeeded", "failed"]) -> None:
        self.rollouts[key.rollout_id]["status"] = status

    def events(self, key: RolloutKey) -> list[dict[str, Any]]:
        return list(self.rollouts.get(key.rollout_id, {}).get("events", {}).values())
