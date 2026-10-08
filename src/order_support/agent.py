"""Northwind Outdoor order-support agent: the ASSERT callable eval target.

ASSERT invokes :func:`chat` once per turn (``inference.target.callable``) and
captures the OpenTelemetry spans emitted here to build the judged transcript:
the ``agent.chat`` AGENT root, one OpenInference LLM span per model call
(:mod:`order_support.otel`) and one TOOL span per tool call
(:mod:`order_support.tools`).

The agent is a Microsoft Agent Framework *declarative* Prompt agent. Its
evolvable surface is data under ``harness/`` (or ``$ORDER_SUPPORT_HARNESS_DIR``):
``agent.yaml`` plus the instruction files listed under its ``x-ci`` key.

The chat client comes from ``ORDER_AGENT_PROFILE``:

* ``offline`` (default): an OpenAI-compatible Chat Completions endpoint at
  ``OPENAI_API_BASE`` (e.g. llama-server), model ``ORDER_AGENT_MODEL`` without
  its LiteLLM-style ``openai/`` prefix;
* ``copilot``: ``ci_lab.providers.copilot.CopilotChatClient`` with the
  ``model.id`` alias from ``agent.yaml``;
* ``fake``: ``ci_lab.testing.FakeChatClient`` (tests only).

:func:`set_client_override` pins a client regardless of profile. Each model call
is bounded by :func:`agent_timeout` and a turn makes at most
:data:`MAX_TOOL_LOOP_ITERATIONS` model calls.
"""

from __future__ import annotations

import asyncio
import copy
import os
import threading
import warnings
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

try:
    from assert_ai import auto_trace

    auto_trace.enable(project_name=os.environ.get("PHOENIX_PROJECT_NAME", "order-support-agent"),
                      auto_instrument=True)
except Exception:  # pragma: no cover - tracing is best-effort outside ASSERT
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())

from agent_framework import ChatContext, ChatMiddleware, ChatResponse, Message

from ci_lab import obs
from order_support import data, maf_tools, otel

_tracer = trace.get_tracer("order_support.agent")

MAX_TOOL_LOOP_ITERATIONS = 8
LOOP_EXCEEDED_TEXT = "[agent: tool loop exceeded]"
DATE_LINE = f"Today's date is {data.TODAY.isoformat()}."
# What ASSERT records as the target system prompt (evals/assert/*/eval_config.yaml) and the
# judge sees: the frozen policy plus the date. The model's instructions are the harness
# composition (see instructions()), which starts from the same policy.
SYSTEM_PROMPT = data.load_policy() + "\n" + DATE_LINE
TIMEOUT_ENV = "ORDER_AGENT_TIMEOUT_S"
# Set by ``order-support-evals run --model-timeout`` so the agent follows the eval timeout.
EVALS_TIMEOUT_ENV = "ORDER_EVALS_MODEL_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 600.0

HARNESS_DIR = Path(__file__).resolve().with_name("harness")
HARNESS_ENV = "ORDER_SUPPORT_HARNESS_DIR"
PROFILE_ENV = "ORDER_AGENT_PROFILE"
PROFILES = ("offline", "copilot", "fake")
X_CI_KEY = "x-ci"

_client_override: Any = None


def agent_model() -> str:
    """The offline profile's model, in LiteLLM form (``openai/<served name>``)."""
    return os.environ.get("ORDER_AGENT_MODEL", "openai/local")


def agent_timeout() -> float:
    """Per-model-call timeout: ``ORDER_AGENT_TIMEOUT_S``, else the eval model timeout, else 600 s."""
    for name in (TIMEOUT_ENV, EVALS_TIMEOUT_ENV):
        if raw := os.environ.get(name, "").strip():
            value = float(raw)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {raw!r}")
            return value
    return DEFAULT_TIMEOUT_S


def agent_profile() -> str:
    profile = os.environ.get(PROFILE_ENV, "").strip().lower() or "offline"
    if profile not in PROFILES:
        raise ValueError(f"{PROFILE_ENV} must be one of {', '.join(PROFILES)}, got {profile!r}")
    return profile


def set_client_override(client: Any) -> None:
    """Use ``client`` (any MAF chat client) for every :func:`chat` call; ``None`` restores profiles."""
    global _client_override
    _client_override = client


# ------------------------------------------------------------------ harness spec

def harness_dir() -> Path:
    return Path(os.environ.get(HARNESS_ENV) or HARNESS_DIR).resolve()


def _strip_front_matter(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "".join(lines[i + 1:])
    return text


def _read_contained(root: Path, relative: str) -> str:
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError(f"harness instructions file {relative!r} escapes {root}")
    return _strip_front_matter(resolved.read_text(encoding="utf-8")).strip()


def _load_spec(root: Path | str | None = None) -> dict[str, Any]:
    """agent.yaml with ``x-ci`` stripped and its instruction files composed into ``instructions``.

    Files are read in order from inside the harness dir (front matter dropped) and
    joined by blank lines; today's date is appended last.
    """
    root = Path(root).resolve() if root is not None else harness_dir()
    spec = yaml.safe_load((root / "agent.yaml").read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"{root / 'agent.yaml'} is not a mapping")
    x_ci = spec.pop(X_CI_KEY, None) or {}
    parts = [str(spec["instructions"]).strip()] if spec.get("instructions") else []
    parts += [_read_contained(root, str(f)) for f in x_ci.get("instructions_files") or []]
    spec["instructions"] = "\n\n".join([*(p for p in parts if p), DATE_LINE])
    return spec


def instructions(root: Path | str | None = None) -> str:
    """The instructions the model sees (harness composition plus date)."""
    return _load_spec(root)["instructions"]


def spec_model_alias(spec: Mapping[str, Any]) -> str | None:
    model = spec.get("model")
    return str(model["id"]) if isinstance(model, Mapping) and model.get("id") else None


def _factory_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The spec handed to AgentFactory: model routing removed so it uses our client.

    AgentFactory builds its own client whenever ``model.id`` is set (and rejects the
    unmapped ``GitHubCopilot`` provider), so routing stays with :func:`_resolve_client`.
    """
    out = copy.deepcopy(dict(spec))
    model = out.get("model")
    if isinstance(model, dict):
        for key in ("id", "provider", "apiType", "connection"):
            model.pop(key, None)
        if not model:
            out.pop("model")
    return out


def _tidy_tool_schemas(agent: Any) -> None:
    """Drop the declarative loader's empty ``examples``/``strict`` so tool schemas match TOOL_SCHEMAS."""
    for tool in (agent.default_options or {}).get("tools") or []:
        params = tool.parameters()
        if params.get("examples") == []:
            params.pop("examples")
        if params.get("strict") is False:
            params.pop("strict")


def build_agent(client: Any, spec: Mapping[str, Any] | None = None) -> Any:
    """The declarative MAF agent for ``spec`` (default: the current harness) on ``client``."""
    from agent_framework_declarative import AgentFactory

    spec = spec if spec is not None else _load_spec()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*experimental.*")
        factory = AgentFactory(client=client, bindings=maf_tools.bindings(), safe_mode=True)
        agent = factory.create_agent_from_dict(_factory_spec(spec))
    _tidy_tool_schemas(agent)
    return agent


# ------------------------------------------------------------------ clients

def _offline_client() -> Any:
    from agent_framework_openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or None
    api_key = os.environ.get("OPENAI_API_KEY") or "sk-local"
    return OpenAIChatCompletionClient(
        model=agent_model().removeprefix("openai/"),
        async_client=AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=agent_timeout()))


def _copilot_client(model: str | None) -> Any:
    try:
        from ci_lab.providers.copilot import CopilotChatClient
    except ImportError as exc:
        raise RuntimeError(f"{PROFILE_ENV}=copilot needs ci_lab.providers.copilot.CopilotChatClient, "
                           "which is not available in this checkout") from exc
    return CopilotChatClient(model=model) if model else CopilotChatClient()


def _resolve_client(spec: Mapping[str, Any]) -> tuple[Any, str, str | None]:
    """(client, model name for spans, OpenInference provider) for this call."""
    alias = spec_model_alias(spec)
    if _client_override is not None:
        return _client_override, str(getattr(_client_override, "model", None) or alias or "unknown"), None
    profile = agent_profile()
    if profile == "offline":
        return _offline_client(), agent_model(), "openai"
    if profile == "copilot":
        client = _copilot_client(alias)
        return client, str(getattr(client, "model", None) or alias), None
    from ci_lab.testing import FakeChatClient

    client = FakeChatClient()
    return client, client.model, None


# ------------------------------------------------------------------ run loop

class _TurnGuard(ChatMiddleware):
    """Bounds a turn like the pre-MAF loop: per-call timeout, at most N model calls.

    MAF asks for one extra tool-free response once ``max_iterations`` is spent; that
    call is answered with :data:`LOOP_EXCEEDED_TEXT` instead of reaching the model.
    """

    def __init__(self, max_calls: int, timeout_s: float) -> None:
        self.max_calls = max_calls
        self.timeout_s = timeout_s
        self.calls = 0

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        if self.calls >= self.max_calls:
            context.result = ChatResponse(messages=[Message(role="assistant", contents=[LOOP_EXCEEDED_TEXT])],
                                          finish_reason="stop")
            return
        self.calls += 1
        await asyncio.wait_for(call_next(), timeout=self.timeout_s)


def _seed_messages(message: str, history: list[dict[str, Any]] | None) -> list[Message]:
    """Prior user/assistant turns (current turn is ``history[-1]``); system turns are ignored."""
    turns = [Message(role=t["role"], contents=[str(t.get("content") or "")])
             for t in (history or []) if t.get("role") in ("user", "assistant")]
    return turns or [Message(role="user", contents=[message])]


def _configure_client(client: Any) -> None:
    config = getattr(client, "function_invocation_configuration", None)
    if isinstance(config, dict):
        config["max_iterations"] = MAX_TOOL_LOOP_ITERATIONS
        # The old loop fed tool errors back until the iteration bound; keep that.
        config["max_consecutive_errors_per_request"] = MAX_TOOL_LOOP_ITERATIONS
        # One TOOL span at a time, in the order the model asked for them.
        config["allow_concurrent_invocation"] = False


def _final_text(response: Any) -> str:
    for message in reversed(list(response.messages or [])):
        if str(getattr(message.role, "value", message.role)) == "assistant":
            return str(message.text or "")
    return str(response.text or "")


async def _run_turn(spec: Mapping[str, Any], client: Any, model_name: str, provider: str | None,
                    message: str, history: list[dict[str, Any]] | None, timeout_s: float) -> str:
    _configure_client(client)
    agent = build_agent(client, spec)
    options: dict[str, Any] = {}
    if (temp := os.environ.get("ORDER_AGENT_TEMPERATURE")) is not None:
        options["temperature"] = float(temp)
    middleware = [_TurnGuard(MAX_TOOL_LOOP_ITERATIONS, timeout_s),
                  otel.OpenInferenceChatMiddleware(model_name, provider)]
    try:
        response = await agent.run(_seed_messages(message, history), middleware=middleware,
                                   options=options or None)
    finally:
        if client is not _client_override and (inner := getattr(client, "client", None)) is not None:
            close = getattr(inner, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
    return _final_text(response)


def _run_sync(make: Callable[[], Awaitable[str]]) -> str:
    """Run a coroutine from sync code, also when this thread already runs an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())  # type: ignore[arg-type]
    run = obs.wrap_ctx(lambda: asyncio.run(make()))  # keeps the AGENT span and telemetry suppression
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = run()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=worker, name="order-support-chat", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def chat(message: str, history: list[dict[str, Any]] | None = None) -> str:
    """Run one support turn and return the assistant's reply."""
    spec = _load_spec()
    timeout_s = agent_timeout()
    client, model_name, provider = _resolve_client(spec)
    with _tracer.start_as_current_span("agent.chat") as root:
        root.set_attribute("openinference.span.kind", "AGENT")
        root.set_attribute("input.value", message)
        root.set_attribute("llm.model_name", model_name)
        with otel.suppress_maf_telemetry():
            final_text = _run_sync(lambda: _run_turn(spec, client, model_name, provider, message,
                                                     history, timeout_s))
        root.set_attribute("output.value", final_text)
        return final_text


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    print(chat(" ".join(sys.argv[1:]) or "Where is order NW-10007? Email ivy.chen@example.com"))
