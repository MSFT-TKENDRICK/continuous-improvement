"""Chat client factory implementing :class:`ci_lab.contracts.ChatClientFactory`."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from ci_lab.contracts import ChatClientFactory, Profile, Purpose, RolloutKey
from ci_lab.providers.offline import check_loopback, check_offline_endpoints

__all__ = ["make_chat_client"]

_REQUEST_CARRIER: ContextVar[dict[str, str] | None] = ContextVar(
    "ci_lab_request_carrier", default=None
)


def make_chat_client(*, profile: Profile | str, model: str, purpose: Purpose,
                     rollout: RolloutKey | None = None, **kw: Any) -> Any:
    """Build a MAF chat client for ``profile``.

    * ``COPILOT``: :class:`ci_lab.providers.copilot.CopilotChatClient`. When ``rollout`` is
      given and no ``session_scope`` is passed, Copilot sessions are scoped to
      ``rollout.rollout_id``.
    * ``OFFLINE``: ``agent_framework_openai.OpenAIChatCompletionClient`` pointed at
      ``base_url=`` (for example an AGL proxy URL), or at ``$OPENAI_API_BASE`` /
      ``$OPENAI_BASE_URL``. The API key is ``api_key=``, then ``$OPENAI_API_KEY``, then ``"local"``.
      ``offline`` is network-free: the base URL, the endpoint variables and the s1 judge backend
      URLs (``$CI_S1_LLAMA_URL``, ``$CI_S1_SYSTEMONE_URL``) must all be loopback, otherwise
      :class:`ci_lab.providers.offline.NetworkPolicyError` is raised (fail closed).
    * ``FAKE``: :class:`ci_lab.testing.FakeChatClient` (``script=`` and ``default=`` pass through).

    Extra keyword arguments go to the client constructor.
    """
    profile = Profile(profile)
    if profile is Profile.COPILOT:
        from ci_lab.providers.copilot import CopilotChatClient

        if rollout is not None and kw.get("session_scope") is None:
            rollout_id = rollout.rollout_id
            kw["session_scope"] = lambda: rollout_id
        return CopilotChatClient(model=model, **kw)
    if profile is Profile.OFFLINE:
        from agent_framework_openai import OpenAIChatCompletionClient
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient

        base_url = kw.pop("base_url", None) or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("OFFLINE profile needs base_url= or OPENAI_API_BASE")
        check_offline_endpoints()
        check_loopback("base_url", base_url)
        api_key = kw.pop("api_key", None) or os.environ.get("OPENAI_API_KEY") or "local"
        headers = {"x-ci-purpose": str(purpose), **(kw.pop("default_headers", None) or {})}
        if rollout is not None:
            headers.setdefault("x-ci-rollout-id", rollout.rollout_id)

        class _TraceContextOpenAIChatCompletionClient(OpenAIChatCompletionClient):
            def get_response(self, *args: Any, **kwargs: Any) -> Any:
                if kwargs.get("stream"):
                    return super().get_response(*args, **kwargs)
                carrier = _current_carrier()

                async def invoke() -> Any:
                    token = _REQUEST_CARRIER.set(carrier)
                    try:
                        response = super(
                            _TraceContextOpenAIChatCompletionClient, self
                        ).get_response(*args, **kwargs)
                        return await response
                    finally:
                        _REQUEST_CARRIER.reset(token)

                return invoke()

        if kw.get("async_client") is None:
            # Per-request W3C trace context so long-lived servers (AGL proxy, copilot-serve) join our trace.
            class _TraceContextHttpClient(DefaultAsyncHttpxClient):
                async def send(self, request: Any, *args: Any, **kwargs: Any) -> Any:
                    from opentelemetry.instrumentation.utils import (
                        suppress_http_instrumentation,
                    )

                    _inject_trace_context(request)
                    with suppress_http_instrumentation():
                        return await super().send(request, *args, **kwargs)

            kw["async_client"] = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers,
                                             http_client=_TraceContextHttpClient())
        return _TraceContextOpenAIChatCompletionClient(
            model=model, api_key=api_key, base_url=base_url, default_headers=headers, **kw
        )
    if profile is Profile.FAKE:
        from ci_lab.testing import FakeChatClient

        return FakeChatClient(kw.pop("script", ()), model=model, **kw)
    raise ValueError(f"unknown profile: {profile!r}")  # pragma: no cover


_: ChatClientFactory = make_chat_client


def _inject_trace_context(request: Any) -> None:
    carrier = _REQUEST_CARRIER.get()
    if carrier is None:
        carrier = _current_carrier()
    for name in ("traceparent", "tracestate"):
        if name not in carrier:
            request.headers.pop(name, None)
    for k, v in carrier.items():
        request.headers[k] = v


def _current_carrier() -> dict[str, str]:
    from ci_lab import obs

    return obs.carrier()
