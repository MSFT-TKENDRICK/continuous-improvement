import asyncio

import pytest

from ci_lab.contracts import Profile, RolloutKey
from ci_lab.providers.copilot import CopilotChatClient
from ci_lab.providers.factory import make_chat_client
from ci_lab.providers.fake_sdk import FakeCopilotClient
from ci_lab.testing import FakeChatClient


def test_copilot_profile_scopes_sessions_to_rollout():
    key = RolloutKey("exp-r01", "v1", "c01", 2)
    client = make_chat_client(profile=Profile.COPILOT, model="gpt-5-mini", purpose="target", rollout=key,
                              sdk_client=FakeCopilotClient(["hi"]), timeout_s=12)
    assert isinstance(client, CopilotChatClient)
    assert client.model == "gpt-5-mini" and client.timeout_s == 12
    assert client._scope() == key.rollout_id
    assert asyncio.run(client.get_response("q")).text == "hi"
    assert [lv.key[0] for lv in client._lives] == []  # tool-less: one-shot session


def test_copilot_profile_accepts_string_profile():
    assert isinstance(make_chat_client(profile="copilot", model="m", purpose="judge"), CopilotChatClient)


def test_offline_profile_uses_openai_chat_completions(monkeypatch):
    from agent_framework_openai import OpenAIChatCompletionClient

    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:9999/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    key = RolloutKey("e", "v", "c")
    client = make_chat_client(profile=Profile.OFFLINE, model="qwen", purpose="target", rollout=key)
    assert isinstance(client, OpenAIChatCompletionClient)
    assert client.model == "qwen"
    assert str(client.client.base_url).startswith("http://127.0.0.1:9999/v1")
    assert client.client.api_key == "local"
    assert client.client.default_headers["x-ci-rollout-id"] == key.rollout_id

    proxy = make_chat_client(profile=Profile.OFFLINE, model="qwen", purpose="judge",
                             base_url="http://127.0.0.1:1234/rollout/x/v1", api_key="k")
    assert str(proxy.client.base_url).startswith("http://127.0.0.1:1234/rollout/x/v1")
    assert proxy.client.api_key == "k"


def test_offline_profile_requires_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        make_chat_client(profile=Profile.OFFLINE, model="m", purpose="target")


def test_fake_profile():
    client = make_chat_client(profile=Profile.FAKE, model="fm", purpose="critic", script=["scripted"])
    assert isinstance(client, FakeChatClient) and client.model == "fm"
    assert asyncio.run(client.get_response("q")).text == "scripted"
