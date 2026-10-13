"""One end-to-end test against a real loopback ``agl-server`` (offline; skipped if it cannot start)."""

from __future__ import annotations

import pytest

from ci_lab.agl.client import AglClient
from ci_lab.agl.server import AglServer, AglServerError, _hydra_value
from ci_lab.contracts import RolloutKey


def test_command_never_contains_key() -> None:
    srv = AglServer("gpt-x", key="super-secret-key", port=1234)
    cmd = " ".join(srv.command())
    assert "super-secret-key" not in cmd and "super-secret-key" not in repr(srv)
    assert "port=1234" in cmd and "default_proxy.model_name=gpt-x" in cmd and "key=${oc.env:CI_LAB_AGL_KEY}" in cmd
    assert _hydra_value("a b'c") == "'a b\\'c'" and _hydra_value(0.0) == "0.0"


def test_real_server_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("agentlightning")
    server = AglServer("ci-model", startup_timeout=30.0, log_path=tmp_path / "agl-server.log", cwd=tmp_path)
    try:
        server.start()
    except AglServerError as exc:
        pytest.skip(f"agl-server could not start within 30 s on loopback: {exc}")
    try:
        c = server.client
        assert c.healthz()
        models = c.register_models([{"model": "ci-model", "endpoint": "http://127.0.0.1:9/v1"}])
        assert models[0]["model"] == "ci-model"
        key = RolloutKey("camp-int", "base", "case-1")
        created = c.create_rollout(key.rollout_id, {"data_id": "case-1"}, metadata={"variant": "base"})
        assert created["rollout_id"] == key.rollout_id and created["status"]["state"] == "queuing"
        assert c.create_rollout(key.rollout_id, {})["input"] == {"data_id": "case-1"}  # idempotent
        c.patch_state(key.rollout_id, "running", attempt_id="0")
        ev = c.post_event(key.rollout_id, "0", "reward", {"value": 1.0, "message": None, "source": "test",
                                                          "reason": None})
        assert ev["rollout_id"] == key.rollout_id and ev["attempt_id"] == "0"
        assert c.patch_state(key.rollout_id, "succeeded")["status"]["state"] == "succeeded"
        assert c.patch_state(key.rollout_id, "failed")["status"]["state"] == "succeeded"  # 409 reconciled
        events = c.get_events(key.rollout_id)
        assert [(e["event_type"], e["data"]["value"]) for e in events] == [("reward", 1.0)]
        with AglClient(server.base_url or "", "wrong-key") as bad:
            with pytest.raises(Exception):
                bad.get_events(key.rollout_id)
        assert server.proxy_base_url(key, "train").endswith(
            f"/proxy/rollout/{key.rollout_id}/attempt/0/mode/train/openai/v1")
    finally:
        server.stop()
    assert not server.running
    assert server.key not in (tmp_path / "agl-server.log").read_text(encoding="utf-8", errors="replace")
    assert not AglClient(server.base_url or "", None, timeout=1.0).healthz()
