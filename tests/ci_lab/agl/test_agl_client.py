from __future__ import annotations

import json

import pytest

from ci_lab.agl.client import AglClient, AglConflict, AglError, proxy_base_url
from ci_lab.contracts import RolloutKey

BASE = "http://agl.test"


def _client(fake, key: str | None = None) -> AglClient:  # type: ignore[no-untyped-def]
    return AglClient(BASE, fake.key if key is None else key, transport=fake.transport())


def test_api_key_header_and_auth_error(fake_agl) -> None:  # type: ignore[no-untyped-def]
    with _client(fake_agl) as c:
        assert c.healthz()
        c.register_model("gpt-x", "http://up/v1")
    assert fake_agl.requests[-1].headers["x-api-key"] == fake_agl.key
    with _client(fake_agl, key="wrong") as bad, pytest.raises(AglError) as ei:
        bad.register_model("gpt-x", "http://up/v1")
    assert ei.value.status_code == 401
    assert fake_agl.key not in repr(_client(fake_agl))


def test_register_models_body_shape(fake_agl) -> None:  # type: ignore[no-untyped-def]
    with _client(fake_agl) as c:
        out = c.register_models([{"model": "m", "endpoint": "http://a/v1"},
                                 {"model": "m", "endpoint": "http://b/v1", "version": 2}])
    assert json.loads(fake_agl.requests[-1].content) == [
        {"model": "m", "endpoint": "http://a/v1", "version": 0},
        {"model": "m", "endpoint": "http://b/v1", "version": 2}]
    assert [m["endpoint"] for m in out] == ["http://a/v1", "http://b/v1"]
    assert set(fake_agl.models["m"]) == {"http://a/v1", "http://b/v1"}


def test_create_rollout_is_idempotent(fake_agl) -> None:  # type: ignore[no-untyped-def]
    with _client(fake_agl) as c:
        r1 = c.create_rollout("ro-1", {"data_id": "case-1"}, metadata={"variant": "base"})
        c.patch_state("ro-1", "running")
        r2 = c.create_rollout("ro-1", {"data_id": "other"})
        assert r1["rollout_id"] == r2["rollout_id"] == "ro-1"
        assert r2["input"] == {"data_id": "case-1"} and r2["status"]["state"] == "running"
        body = json.loads(fake_agl.requests[0].content)
        assert body == [{"rollout_id": "ro-1", "input": {"data_id": "case-1"}, "is_train": False,
                         "metadata": {"variant": "base"}}]
        fake_agl.create_conflict = True
        r3 = c.create_rollout("ro-1", {})
        assert r3["status"]["state"] == "running"
        assert c.get_rollout("ro-missing") is None


def test_post_and_get_events(fake_agl) -> None:  # type: ignore[no-untyped-def]
    with _client(fake_agl) as c:
        c.create_rollout("ro-1", {})
        ev = c.post_event("ro-1", "0", "reward", {"value": 1.0})
        assert fake_agl.requests[-1].url.path == "/api/rollouts/ro-1/attempt/0/events"
        assert set(ev) == {"event_type", "rollout_id", "attempt_id", "timestamp", "data"}
        c.post_event("ro-1", "0", "ci.score", {"name": "s"})
        assert [e["event_type"] for e in c.get_events("ro-1")] == ["reward", "ci.score"]
        assert [e["data"] for e in c.get_events("ro-1", event_type="reward")] == [{"value": 1.0}]
        assert fake_agl.requests[-1].url.params["event_type"] == "reward"
        c.post_event("ro-1", "1", "reward", {"value": 0.0})
        assert len(c.get_events("ro-1")) == 2  # server reads status.last_attempt_id ("0")
        c.patch_status("ro-1", last_attempt_id="1")
        assert [e["data"]["value"] for e in c.get_events("ro-1")] == [0.0]


def test_patch_state_lifecycle_and_reconcile(fake_agl) -> None:  # type: ignore[no-untyped-def]
    with _client(fake_agl) as c:
        c.create_rollout("ro-1", {})
        assert c.patch_state("ro-1", "running", attempt_id="0")["status"]["state"] == "running"
        assert json.loads(fake_agl.requests[-1].content) == {"status": {"state": "running", "last_attempt_id": "0"}}
        assert c.patch_state("ro-1", "succeeded")["status"]["state"] == "succeeded"
        # 409 -> re-read: already terminal is accepted (first terminal wins)
        assert c.patch_state("ro-1", "failed")["status"]["state"] == "succeeded"
        assert c.patch_state("ro-1", "succeeded")["status"]["state"] == "succeeded"
        # queuing -> succeeded is advanced through running
        c.create_rollout("ro-2", {})
        assert c.patch_state("ro-2", "succeeded")["status"]["state"] == "succeeded"
        # running -> queuing is not reconcilable
        c.create_rollout("ro-3", {})
        c.patch_state("ro-3", "running")
        with pytest.raises(AglConflict):
            c.patch_state("ro-3", "queuing")
        with pytest.raises(ValueError):
            c.patch_state("ro-3", "bogus")


def test_proxy_base_url() -> None:
    key = RolloutKey("e", "v", "c", attempt=2)
    url = proxy_base_url("http://127.0.0.1:4747/", key, "train")
    assert url == f"http://127.0.0.1:4747/proxy/rollout/{key.rollout_id}/attempt/2/mode/train/openai/v1"
    assert proxy_base_url("http://h", "ro-1").endswith("/proxy/rollout/ro-1/attempt/0/mode/val/openai/v1")
    assert AglClient("http://h/", "k").proxy_base_url("ro-1", attempt_id="3").endswith("/attempt/3/mode/val/openai/v1")
    with pytest.raises(ValueError):
        proxy_base_url("http://h", "ro-1", "test")  # type: ignore[arg-type]
