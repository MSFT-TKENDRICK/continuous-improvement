"""Northwind Outdoor order-support agent: the ASSERT callable eval target.

ASSERT invokes :func:`chat` once per turn (``inference.target.callable``) and
captures the OpenTelemetry spans emitted here (AGENT root, LiteLLM LLM spans,
TOOL spans from :mod:`order_support.tools`) to build the judged transcript.

Model routing is LiteLLM's: ``ORDER_AGENT_MODEL`` (default ``openai/local``)
plus the usual provider env vars, e.g. ``OPENAI_API_BASE`` for a local
OpenAI-compatible server such as llama-server.
"""

from __future__ import annotations

import json
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

try:
    from assert_ai import auto_trace

    auto_trace.enable(project_name=os.environ.get("PHOENIX_PROJECT_NAME", "order-support-agent"),
                      auto_instrument=True)
except Exception:  # pragma: no cover - tracing is best-effort outside ASSERT
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())

import litellm

from order_support import data, tools

_tracer = trace.get_tracer("order_support.agent")

MAX_TOOL_LOOP_ITERATIONS = 8
SYSTEM_PROMPT = data.load_policy() + f"\nToday's date is {data.TODAY.isoformat()}."


def agent_model() -> str:
    return os.environ.get("ORDER_AGENT_MODEL", "openai/local")


def _seed_messages(message: str, history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """System prompt + prior user/assistant turns (current turn is ``history[-1]``)."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns = [{"role": t["role"], "content": str(t.get("content") or "")}
             for t in (history or []) if t.get("role") in ("user", "assistant")]
    messages.extend(turns or [{"role": "user", "content": message}])
    return messages


def _tool_call_parts(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
    fn = tool_call.function
    try:
        args = json.loads(fn.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return tool_call.id, fn.name, args if isinstance(args, dict) else {}


def chat(message: str, history: list[dict[str, Any]] | None = None) -> str:
    """Run one support turn and return the assistant's reply."""
    model = agent_model()
    messages = _seed_messages(message, history)
    kwargs: dict[str, Any] = {}
    if (temp := os.environ.get("ORDER_AGENT_TEMPERATURE")) is not None:
        kwargs["temperature"] = float(temp)

    with _tracer.start_as_current_span("agent.chat") as root:
        root.set_attribute("openinference.span.kind", "AGENT")
        root.set_attribute("input.value", message)
        root.set_attribute("llm.model_name", model)
        final_text = "[agent: tool loop exceeded]"
        for _ in range(MAX_TOOL_LOOP_ITERATIONS):
            response = litellm.completion(model=model, messages=messages, tools=tools.TOOL_SCHEMAS,
                                          tool_choice="auto", **kwargs)
            msg = response.choices[0].message
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                final_text = str(getattr(msg, "content", "") or "")
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name,
                                                          "arguments": c.function.arguments or "{}"}}
                                            for c in calls]})
            for call in calls:
                call_id, name, args = _tool_call_parts(call)
                result = tools.execute(name, args)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                                 "content": json.dumps(result, ensure_ascii=False)})
        root.set_attribute("output.value", final_text)
        return final_text


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    print(chat(" ".join(sys.argv[1:]) or "Where is order NW-10007? Email ivy.chen@example.com"))
