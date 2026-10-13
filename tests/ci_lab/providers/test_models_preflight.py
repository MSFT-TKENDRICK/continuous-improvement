"""Model availability preflight primitives, the CopilotChatClient model check and copilot-serve
startup check. No network: listings are injected or come from FakeCopilotClient(models=...)."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from fastapi.testclient import TestClient

from ci_lab.providers.copilot import CopilotChatClient
from ci_lab.providers.fake_sdk import FakeCopilotClient
from ci_lab.providers.models import (
    ModelPreflightError,
    ModelUse,
    check_copilot_models,
    check_served_models,
    model_ids,
)
from ci_lab.providers.serve import create_app

AVAILABLE = ["claude-sonnet-5", "gpt-5-mini", "gpt-5.5"]


def _lister(ids=AVAILABLE, calls=None):
    async def list_models():
        if calls is not None:
            calls.append(1)
        return [SimpleNamespace(id=m) for m in ids]

    return list_models


def test_model_ids_accepts_objects_dicts_and_strings():
    assert model_ids([SimpleNamespace(id="b"), {"id": "a"}, "c", {"id": None}, "a"]) == ("a", "b", "c")


def test_check_copilot_models_passes_and_returns_available():
    out = asyncio.run(check_copilot_models([ModelUse("gpt-5-mini", "x")], list_models=_lister()))
    assert out == tuple(sorted(AVAILABLE))


def test_check_copilot_models_without_requirements_does_not_list():
    calls: list[int] = []
    assert asyncio.run(check_copilot_models([], list_models=_lister(calls=calls))) == ()
    assert calls == []


def test_check_copilot_models_missing_names_users_override_and_available():
    uses = [ModelUse("claude-sonnet-4.5", "meta agent analyst", "CI_META_MODEL=<id>"),
            ModelUse("claude-sonnet-4.5", "meta agent critic", "CI_META_MODEL=<id>"),
            ModelUse("gpt-5-mini", "order agent")]
    with pytest.raises(ModelPreflightError) as ei:
        asyncio.run(check_copilot_models(uses, list_models=_lister()))
    msg = str(ei.value)
    assert ei.value.__cause__ is None
    assert "1 model(s)" in msg and "'claude-sonnet-4.5'" in msg
    assert "meta agent analyst, meta agent critic" in msg
    assert msg.count("override: CI_META_MODEL=<id>") == 1
    assert "Available: claude-sonnet-5, gpt-5-mini, gpt-5.5" in msg
    assert "'gpt-5-mini'" not in msg


def test_check_copilot_models_wraps_listing_failures():
    async def boom():
        raise ConnectionError("no cli")

    with pytest.raises(ModelPreflightError, match="could not list models") as ei:
        asyncio.run(check_copilot_models([ModelUse("gpt-5-mini", "x")], list_models=boom))
    assert isinstance(ei.value.__cause__, ConnectionError)


def test_check_served_models_strips_openai_prefix():
    seen = []

    def fetch(base, key):
        seen.append((base, key))
        return [{"id": "gpt-5-mini"}]

    out = check_served_models("http://h/v1", "k", [ModelUse("openai/gpt-5-mini", "tester")], fetch=fetch)
    assert out == ("gpt-5-mini",) and seen == [("http://h/v1", "k")]


def test_check_served_models_missing_and_unreachable():
    with pytest.raises(ModelPreflightError, match=r"(?s)'local' used by tester.*Available: gpt-5-mini"):
        check_served_models("http://h/v1", "k", [ModelUse("openai/local", "tester", "CI_ASSERT_MODEL=x")],
                            fetch=lambda b, k: ["gpt-5-mini"])

    def down(base, key):
        raise OSError("refused")

    with pytest.raises(ModelPreflightError, match="could not list served models") as ei:
        check_served_models("http://h/v1", "k", [ModelUse("x", "y")], fetch=down)
    assert isinstance(ei.value.__cause__, OSError)


# ---------------------------------------------------------------- CopilotChatClient


def test_client_model_check_passes_once():
    sdk = FakeCopilotClient(models=AVAILABLE)
    calls: list[int] = []
    sdk.list_models = _lister(calls=calls)
    client = CopilotChatClient(model="gpt-5-mini", sdk_client=sdk)

    async def go():
        await client._ensure_sdk()
        await client._ensure_sdk()

    asyncio.run(go())
    assert calls == [1]


def test_client_model_check_raises_for_unavailable_model():
    client = CopilotChatClient(model="claude-sonnet-4.5", sdk_client=FakeCopilotClient(models=AVAILABLE))
    with pytest.raises(ModelPreflightError, match=r"(?s)'claude-sonnet-4.5' used by CopilotChatClient.*Available"):
        asyncio.run(client._ensure_sdk())
    assert client._model_checked is False  # checked again on the next attempt


def test_client_model_check_skipped_without_list_models_or_when_disabled():
    asyncio.run(CopilotChatClient(model="anything", sdk_client=FakeCopilotClient())._ensure_sdk())
    sdk = FakeCopilotClient(models=AVAILABLE)
    asyncio.run(CopilotChatClient(model="missing", sdk_client=sdk, verify_model=False)._ensure_sdk())


def test_client_model_check_tolerates_listing_failure():
    sdk = FakeCopilotClient(models=AVAILABLE)

    async def boom():
        raise RuntimeError("listing broke")

    sdk.list_models = boom
    asyncio.run(CopilotChatClient(model="gpt-5-mini", sdk_client=sdk)._ensure_sdk())


# ---------------------------------------------------------------- copilot-serve startup


def test_serve_startup_fails_for_unavailable_model(capsys):
    client = CopilotChatClient(model="claude-sonnet-4.5", sdk_client=FakeCopilotClient(models=AVAILABLE))
    app = create_app(client, api_key="k", close_client=True)
    with pytest.raises(ModelPreflightError), TestClient(app):
        pass
    err = capsys.readouterr().err
    assert "claude-sonnet-4.5" in err and "ci-lab copilot-serve --model <id>" in err


def test_serve_startup_passes_or_skips():
    client = CopilotChatClient(model="gpt-5-mini", sdk_client=FakeCopilotClient(models=AVAILABLE))
    with TestClient(create_app(client, api_key="k", close_client=True)) as tc:
        assert tc.get("/v1/models", headers={"Authorization": "Bearer k"}).status_code == 200
    client = CopilotChatClient(model="missing", sdk_client=FakeCopilotClient(models=AVAILABLE), verify_model=False)
    with TestClient(create_app(client, api_key="k", check_model=False)):
        pass
