"""Deterministic scripted chat client for ``ci-lab chat serve --profile fake``.

No inference and no network. It reads the latest message and replies like a tool-using model:

* the last message carries a tool result -> a short text summary of that result;
* the last user message contains "launch" -> a ``launch_campaign`` call (this triggers the
  AG-UI approval interrupt);
* it contains "draft" -> a ``draft_campaign`` call (cid, arms and rounds parsed from the text;
  defaults ``chat-demo``, 2 arms, 1 round, local target);
* anything else -> a canned text reply.

Text is streamed in several chunks so clients see multiple ``TEXT_MESSAGE_CONTENT`` events.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
)
from agent_framework.observability import ChatTelemetryLayer

from ci_lab.providers.streaming import response_updates

__all__ = ["DEFAULT_CID", "FakeDesignerClient", "parse_request"]

DEFAULT_CID = "chat-demo"
GREETING = ("I design RRSI campaigns. Tell me what to improve, how many arms and rounds you can afford, "
            "and whether to run locally or via the workflow. Say \"draft\" to save a design and "
            "\"launch\" to start it after you approve.")
_CID_RE = re.compile(r"\b[a-z0-9][a-z0-9-]{2,40}\b")
_COUNT_RE = re.compile(r"\b(\d+)[- ]?(?P<unit>arms?|rounds?)\b")


def _role(message: Message) -> str:
    return str(getattr(message.role, "value", message.role))


def parse_request(text: str) -> dict[str, Any]:
    """``{cid, arms, rounds, target}`` parsed from a user message (deterministic, forgiving)."""
    lowered = text.lower()
    cid = next((t for t in _CID_RE.findall(lowered)
                if "-" in t and any(c.isalpha() for c in t) and not _COUNT_RE.fullmatch(t)), DEFAULT_CID)
    counts = {m.group("unit").rstrip("s"): int(m.group(1)) for m in _COUNT_RE.finditer(lowered)}
    return {"cid": cid, "arms": counts.get("arm", 2), "rounds": counts.get("round", 1),
            "target": "workflow" if "workflow" in lowered else "local"}


def _summarize(name: str | None, result: Any) -> str:
    raw = result if isinstance(result, str) else json.dumps(result, default=str)
    if "rejected by user" in raw:
        return "The launch was not approved, so nothing was launched."
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return f"{name or 'The tool'} returned: {raw[:300]}"
    if not isinstance(data, dict):
        return f"{name or 'The tool'} returned: {raw[:300]}"
    if data.get("ok") is False:
        return f"{name or 'The tool'} failed: " + "; ".join(map(str, data.get("errors") or ["unknown error"]))
    if "draft" in data and isinstance(data["draft"], dict):
        draft = data["draft"]
        est = draft.get("estimate") or {}
        return (f"Drafted {draft.get('cid')}: {draft.get('rounds')} round(s), target {draft.get('target')}, "
                f"about {est.get('evaluations', '?')} evaluations. Say \"launch {draft.get('cid')}\" to start it.")
    if "launched" in data:
        if data.get("launched"):
            return f"Launched {data.get('cid')} ({data.get('status')})."
        if data.get("dry_run"):
            return f"Dry run: recorded the launch command for {data.get('cid')}; nothing was executed."
        return f"{data.get('cid')} was not launched (status {data.get('status')})."
    return f"{name or 'The tool'} returned {len(data)} field(s)."


class FakeDesignerClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    """Scripted MAF chat client for the ``fake`` chat profile (see module docstring)."""

    OTEL_PROVIDER_NAME = "fake"

    def __init__(self, *, model: str = "fake-designer", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.requests: list[list[Message]] = []

    def _reply(self, messages: Sequence[Message]) -> list[Content] | str:
        self.requests.append(list(messages))
        names: dict[str, str] = {}
        for m in messages:
            for c in m.contents:
                if c.type == "function_call" and c.call_id:
                    names[c.call_id] = c.name or ""
        last = messages[-1] if messages else None
        if last is not None:
            results = [c for c in last.contents if c.type == "function_result"]
            if results:
                return " ".join(_summarize(names.get(c.call_id or ""), c.result) for c in results)
        text = next((m.text for m in reversed(messages) if _role(m) == "user" and m.text), "")
        lowered = text.lower()
        req = parse_request(text)
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        if "launch" in lowered:
            return [Content.from_function_call(call_id=call_id, name="launch_campaign",
                                               arguments={"cid": req["cid"]})]
        if "draft" in lowered:
            return [Content.from_function_call(
                call_id=call_id, name="draft_campaign",
                arguments={"cid": req["cid"], "hyper": {"arms": req["arms"]}, "rounds": req["rounds"],
                           "target": req["target"], "rationale": "scripted fake-profile draft"})]
        return GREETING

    async def _respond(self, messages: Sequence[Message]) -> ChatResponse:
        reply = self._reply(messages)
        contents: list[Any] = [Content.from_text(reply)] if isinstance(reply, str) else reply
        return ChatResponse(messages=[Message(role="assistant", contents=contents)], model=self.model,
                            response_id=f"resp_{uuid.uuid4().hex[:12]}", finish_reason="stop")

    async def _stream(self, messages: Sequence[Message]) -> AsyncIterator[ChatResponseUpdate]:
        reply = self._reply(messages)
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        if not isinstance(reply, str):
            response = ChatResponse(messages=[Message(role="assistant", contents=reply)], model=self.model,
                                    response_id=response_id)
            for update in response_updates(response):
                yield update
            return
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        words = reply.split(" ")
        chunks = [" ".join(words[i:i + 6]) + (" " if i + 6 < len(words) else "") for i in range(0, len(words), 6)]
        for i, chunk in enumerate(chunks):
            yield ChatResponseUpdate(contents=[Content.from_text(chunk)], role="assistant", message_id=message_id,
                                     response_id=response_id, model=self.model,
                                     finish_reason="stop" if i == len(chunks) - 1 else None)

    def _inner_get_response(self, *, messages: Sequence[Message], stream: bool,
                            options: Mapping[str, Any], **kwargs: Any) -> Awaitable[ChatResponse] | Any:
        if stream:
            return self._build_response_stream(self._stream(messages),
                                               response_format=options.get("response_format"))
        return self._respond(messages)
