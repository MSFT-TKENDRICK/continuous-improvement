"""GuardedStream: R3 output lint for streaming runs (B6).

The agent's ``ResponseStream`` is consumed completely (tool loop included) before the first update
is released; the finalized response is linted/redacted, then re-emitted as updates. Nothing
reaches the caller before the verdict.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    MiddlewareFailure,
    ResponseStream,
)


class GuardStreamingError(MiddlewareFailure):
    """Streaming was requested but guard stream buffering is disabled (R3 cannot be applied)."""


def rederive(response: AgentResponse) -> list[AgentResponseUpdate]:
    return [AgentResponseUpdate(contents=list(m.contents), role=m.role, author_name=m.author_name,
                                message_id=m.message_id, response_id=response.response_id,
                                agent_id=response.agent_id)
            for m in response.messages if m.contents]


def guarded_stream(inner: ResponseStream[Any, Any], *, bind: Callable[[], Any], unbind: Callable[[Any], None],
                   finish: Callable[[AgentResponse], AgentResponse]) -> ResponseStream[AgentResponseUpdate, AgentResponse]:
    """Fully buffer ``inner``; ``bind``/``unbind`` scope the guard conversation over the consumption
    (tool calls run lazily while ``inner`` is pulled); ``finish`` lints the final response."""
    holder: dict[str, AgentResponse] = {}

    async def updates() -> AsyncIterator[AgentResponseUpdate]:
        token = bind()
        try:
            async for _ in inner:
                pass
            final = await inner.get_final_response()
        finally:
            unbind(token)
        holder["final"] = finish(final)
        for update in rederive(holder["final"]):
            yield update

    def finalizer(_updates: Any) -> AgentResponse:
        return holder["final"]

    return ResponseStream(updates(), finalizer=finalizer)
