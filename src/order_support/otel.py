"""OpenInference LLM spans for the MAF order-support agent.

Before M3 the LLM spans under ``agent.chat`` came from LiteLLM's OpenInference
auto-instrumentation. :class:`OpenInferenceChatMiddleware` reproduces that span
per model call so ASSERT's ``assert_ai.core.otel._spans_to_events`` sees the
same shape: ``openinference.span.kind=LLM``, ``llm.model_name``,
``llm.input_messages.*`` / ``llm.output_messages.*`` (with tool calls),
``llm.token_count.*`` and ``input.value`` / ``output.value``.

MAF also emits its own gen_ai.* spans (chat, invoke_agent, execute_tool) on the
global tracer provider. ASSERT captures *every* span during a turn, so those
would duplicate tool calls and LLM counts. :func:`suppress_maf_telemetry`
turns MAF's instrumentation off for the current context only (a contextvar),
leaving other agents in the process untouched.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Any

from agent_framework import ChatContext, ChatMiddleware, Message
from agent_framework import observability as _maf_obs
from opentelemetry import trace

_tracer = trace.get_tracer("order_support.otel")

LLM_SPAN_NAME = "completion"  # the span name LiteLLM's instrumentor used
_SUPPRESS = contextvars.ContextVar("order_support_suppress_maf_telemetry", default=False)


def _install_scoped_settings() -> None:
    """Make MAF's global ``OBSERVABILITY_SETTINGS.ENABLED`` honour :data:`_SUPPRESS`.

    Every MAF telemetry layer checks that one singleton, so swapping its class for a
    subclass whose ``ENABLED`` consults a contextvar scopes the switch to our calls.
    """
    settings = _maf_obs.OBSERVABILITY_SETTINGS
    base = type(settings)
    if getattr(base, "_order_support_scoped", False):
        return

    class _Scoped(base):  # type: ignore[misc, valid-type]
        _order_support_scoped = True

        @property
        def ENABLED(self) -> bool:  # noqa: N802 - MAF's name
            return False if _SUPPRESS.get() else base.ENABLED.fget(self)  # type: ignore[attr-defined]

        @property
        def SENSITIVE_DATA_ENABLED(self) -> bool:  # noqa: N802 - MAF's name
            return False if _SUPPRESS.get() else base.SENSITIVE_DATA_ENABLED.fget(self)  # type: ignore[attr-defined]

    _Scoped.__name__ = base.__name__
    settings.__class__ = _Scoped


@contextlib.contextmanager
def suppress_maf_telemetry() -> Iterator[None]:
    """Disable MAF's own gen_ai spans within this context (and tasks/threads it spawns)."""
    _install_scoped_settings()
    token = _SUPPRESS.set(True)
    try:
        yield
    finally:
        _SUPPRESS.reset(token)


def maf_telemetry_suppressed() -> bool:
    return _SUPPRESS.get() and not _maf_obs.OBSERVABILITY_SETTINGS.ENABLED


# --------------------------------------------------------------- message shaping

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _arguments_json(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments or "{}"
    return _json(dict(arguments) if isinstance(arguments, Mapping) else arguments)


def _result_text(content: Any) -> str:
    result = getattr(content, "result", None)
    if result is None:
        items = getattr(content, "items", None) or []
        return "".join(str(getattr(i, "text", "") or "") for i in items)
    return result if isinstance(result, str) else _json(result)


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """MAF messages as the OpenAI chat-completions dicts the old LiteLLM loop sent."""
    out: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for message in messages:
        role = str(getattr(message.role, "value", message.role))
        calls = [c for c in message.contents if c.type == "function_call"]
        results = [c for c in message.contents if c.type == "function_result"]
        text = "".join(str(c.text or "") for c in message.contents if c.type == "text")
        if results:
            for r in results:
                entry = {"role": "tool", "tool_call_id": r.call_id, "content": _result_text(r)}
                if r.call_id in names:
                    entry["name"] = names[r.call_id]
                out.append(entry)
            continue
        entry: dict[str, Any] = {"role": role, "content": text}
        if calls:
            entry["tool_calls"] = []
            for c in calls:
                names[c.call_id] = c.name
                entry["tool_calls"].append({"id": c.call_id, "type": "function",
                                            "function": {"name": c.name,
                                                         "arguments": _arguments_json(c.arguments)}})
        out.append(entry)
    return out


def _message_attributes(prefix: str, message: Mapping[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {f"{prefix}.message.role": message["role"]}
    if message.get("content"):
        attrs[f"{prefix}.message.content"] = message["content"]
    if message.get("tool_call_id"):
        attrs[f"{prefix}.message.tool_call_id"] = message["tool_call_id"]
    if message.get("name"):
        attrs[f"{prefix}.message.name"] = message["name"]
    for j, call in enumerate(message.get("tool_calls") or []):
        tc = f"{prefix}.message.tool_calls.{j}.tool_call"
        attrs[f"{tc}.id"] = call["id"]
        attrs[f"{tc}.function.name"] = call["function"]["name"]
        attrs[f"{tc}.function.arguments"] = call["function"]["arguments"]
    return attrs


def _tool_schema(tool: Any) -> dict[str, Any] | None:
    name = getattr(tool, "name", None)
    parameters = getattr(tool, "parameters", None)
    if not name or not callable(parameters):
        return None
    return {"type": "function", "function": {"name": name, "description": getattr(tool, "description", ""),
                                             "parameters": parameters()}}


def llm_request_attributes(model: str, messages: Sequence[Message], options: Mapping[str, Any],
                           provider: str | None = None) -> dict[str, Any]:
    from agent_framework._types import prepend_instructions_to_messages

    sent = list(messages)
    if instructions := options.get("instructions"):
        sent = prepend_instructions_to_messages(sent, instructions, role="system")
    as_dicts = to_openai_messages(sent)
    attrs: dict[str, Any] = {"openinference.span.kind": "LLM", "llm.model_name": model,
                             "input.value": _json({"messages": as_dicts}),
                             "input.mime_type": "application/json"}
    if provider:
        attrs["llm.provider"] = provider
    for i, message in enumerate(as_dicts):
        attrs.update(_message_attributes(f"llm.input_messages.{i}", message))
    params = {k: v for k, v in options.items()
              if k not in ("tools", "instructions") and isinstance(v, (str, int, float, bool))}
    attrs["llm.invocation_parameters"] = _json(params)
    for i, tool in enumerate(options.get("tools") or []):
        if (schema := _tool_schema(tool)) is not None:
            attrs[f"llm.tools.{i}.tool.json_schema"] = _json(schema)
    return attrs


def llm_response_attributes(response: Any) -> dict[str, Any]:
    out_messages = to_openai_messages(list(getattr(response, "messages", None) or []))
    attrs: dict[str, Any] = {}
    if getattr(response, "model", None):
        attrs["llm.model_name"] = str(response.model)
    if out_messages:
        merged: dict[str, Any] = {"role": "assistant",
                                  "content": "".join(m.get("content") or "" for m in out_messages
                                                     if m["role"] == "assistant")}
        calls = [c for m in out_messages for c in (m.get("tool_calls") or [])]
        if calls:
            merged["tool_calls"] = calls
        attrs.update(_message_attributes("llm.output_messages.0", merged))
        if merged["content"]:
            attrs["output.value"] = merged["content"]
        else:
            attrs["output.value"] = _json(merged)
            attrs["output.mime_type"] = "application/json"
    if (finish := getattr(response, "finish_reason", None)) is not None:
        attrs["llm.finish_reason"] = str(getattr(finish, "value", finish))
    usage = getattr(response, "usage_details", None) or {}
    for key, attr in (("input_token_count", "llm.token_count.prompt"),
                      ("output_token_count", "llm.token_count.completion"),
                      ("total_token_count", "llm.token_count.total")):
        if isinstance(usage.get(key), int):
            attrs[attr] = usage[key]
    return attrs


class OpenInferenceChatMiddleware(ChatMiddleware):
    """Emit one OpenInference LLM span per model call, as a child of the current span."""

    def __init__(self, model: str, provider: str | None = None) -> None:
        self.model = model
        self.provider = provider

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        if context.stream:
            await call_next()
            return
        options = dict(context.options or {})
        attrs = llm_request_attributes(self.model, context.messages, options, self.provider)
        with _tracer.start_as_current_span(LLM_SPAN_NAME, attributes=attrs) as span:
            await call_next()
            if context.result is not None:
                span.set_attributes(llm_response_attributes(context.result))
