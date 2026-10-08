"""Buffered streaming for chat clients that only produce whole responses.

MAF callers such as the AG-UI endpoint always request ``stream=True``. A client whose
backend has no token stream can satisfy them with :func:`buffered_response_stream`:
the full :class:`ChatResponse` is computed lazily on first iteration and replayed as
one :class:`ChatResponseUpdate` per message. Function calls, approval requests,
usage and the finish reason are preserved, so ``ChatResponse.from_updates`` rebuilds
an equivalent response.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from agent_framework import ChatResponse, ChatResponseUpdate, Content, ResponseStream

__all__ = ["buffered_response_stream", "response_updates"]


def response_updates(response: ChatResponse) -> list[ChatResponseUpdate]:
    """Split ``response`` into updates (one per message; usage rides on the last one)."""
    response_id = response.response_id or f"resp-{uuid.uuid4().hex[:12]}"
    updates: list[ChatResponseUpdate] = []
    for msg in response.messages:
        updates.append(ChatResponseUpdate(
            contents=list(msg.contents), role=msg.role, author_name=msg.author_name,
            message_id=msg.message_id or f"msg-{uuid.uuid4().hex[:12]}", response_id=response_id,
            conversation_id=response.conversation_id, model=response.model, created_at=response.created_at))
    if not updates:
        updates.append(ChatResponseUpdate(contents=[], role="assistant", response_id=response_id,
                                          conversation_id=response.conversation_id, model=response.model))
    last = updates[-1]
    if response.usage_details:
        last.contents.append(Content.from_usage(usage_details=response.usage_details))
    last.finish_reason = response.finish_reason
    if response.additional_properties:
        last.additional_properties = {**(last.additional_properties or {}), **response.additional_properties}
    return updates


def buffered_response_stream(client: Any, make_response: Callable[[], Awaitable[ChatResponse]], *,
                             response_format: Any | None = None) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
    """A MAF ``ResponseStream`` that awaits ``make_response()`` on first iteration and replays it.

    ``client`` is the ``BaseChatClient`` whose standard finalizer builds the final response."""

    async def _updates() -> AsyncIterator[ChatResponseUpdate]:
        response = await make_response()
        for update in response_updates(response):
            yield update

    return client._build_response_stream(_updates(), response_format=response_format)
