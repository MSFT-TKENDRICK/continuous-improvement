"""agent_framework middleware that reports LLM calls/tokens and tool calls into a :class:`RunMeter`.

Imported lazily (only by code that already runs MAF agents) so ``ci_lab.metrics`` stays dependency-free."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agent_framework import (
    ChatContext,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
)

from ci_lab.metrics.runtime import RunMeter

__all__ = ["MeterChatMiddleware", "MeterFunctionMiddleware", "metering_middleware", "usage_tokens"]


def usage_tokens(usage: Any) -> tuple[int, int]:
    """``(input, output)`` token counts of a MAF ``usage_details`` (mapping or attribute style)."""
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, Mapping) else (lambda k: getattr(usage, k, None))
    return int(get("input_token_count") or 0), int(get("output_token_count") or 0)


class MeterChatMiddleware(ChatMiddleware):
    """One ``llm_call`` per chat-client request, with tokens from the response usage."""

    def __init__(self, meter: RunMeter) -> None:
        self.meter = meter

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        try:
            await call_next()
        finally:
            result = getattr(context, "result", None)
            tin, tout = usage_tokens(getattr(result, "usage_details", None))
            self.meter.llm_call(tin, tout)


class MeterFunctionMiddleware(FunctionMiddleware):
    """One ``tool_call`` per function invocation that reaches this middleware."""

    def __init__(self, meter: RunMeter) -> None:
        self.meter = meter

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        self.meter.tool_call(getattr(getattr(context, "function", None), "name", "?"))
        await call_next()


def metering_middleware(meter: RunMeter) -> list[Any]:
    """``[chat, function]`` metering middleware for ``Agent(middleware=...)``."""
    return [MeterChatMiddleware(meter), MeterFunctionMiddleware(meter)]
