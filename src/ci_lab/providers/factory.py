"""Chat client factory implementing :class:`ci_lab.contracts.ChatClientFactory`."""

from __future__ import annotations

import os
from typing import Any

from ci_lab.contracts import ChatClientFactory, Profile, Purpose, RolloutKey

__all__ = ["make_chat_client"]


def make_chat_client(*, profile: Profile | str, model: str, purpose: Purpose,
                     rollout: RolloutKey | None = None, **kw: Any) -> Any:
    """Build a MAF chat client for ``profile``.

    * ``COPILOT``: :class:`ci_lab.providers.copilot.CopilotChatClient`. When ``rollout`` is
      given and no ``session_scope`` is passed, Copilot sessions are scoped to
      ``rollout.rollout_id``.
    * ``OFFLINE``: ``agent_framework_openai.OpenAIChatCompletionClient`` pointed at
      ``base_url=`` (for example an AGL proxy URL), or at ``$OPENAI_API_BASE`` /
      ``$OPENAI_BASE_URL``. The API key is ``api_key=``, then ``$OPENAI_API_KEY``, then ``"local"``.
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
        api_key = kw.pop("api_key", None) or os.environ.get("OPENAI_API_KEY") or "local"
        headers = {"x-ci-purpose": str(purpose), **(kw.pop("default_headers", None) or {})}
        if rollout is not None:
            headers.setdefault("x-ci-rollout-id", rollout.rollout_id)
        if kw.get("async_client") is None:
            # Per-request W3C trace context so long-lived servers (AGL proxy, copilot-serve) join our trace.
            kw["async_client"] = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers,
                                             http_client=DefaultAsyncHttpxClient(
                                                 event_hooks={"request": [_inject_trace_context]}))
        return OpenAIChatCompletionClient(model=model, api_key=api_key, base_url=base_url, default_headers=headers,
                                          **kw)
    if profile is Profile.FAKE:
        from ci_lab.testing import FakeChatClient

        return FakeChatClient(kw.pop("script", ()), model=model, **kw)
    raise ValueError(f"unknown profile: {profile!r}")  # pragma: no cover


_: ChatClientFactory = make_chat_client


async def _inject_trace_context(request: Any) -> None:
    from ci_lab import obs

    for k, v in obs.carrier().items():
        request.headers[k] = v
