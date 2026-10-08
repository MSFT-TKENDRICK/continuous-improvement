from __future__ import annotations

import json

import pytest

from ci_lab.governance import identity as gid

SPONSOR = "maintainer@contoso.com"


@pytest.fixture
def store(tmp_path):
    return gid.IdentityStore(tmp_path / "keys", sponsor=SPONSOR)


def test_sponsor_env_then_git_then_fail_closed():
    assert gid.resolve_sponsor({gid.SPONSOR_ENV: SPONSOR}, git=lambda: "x@y.z") == SPONSOR
    assert gid.resolve_sponsor({}, git=lambda: "dev@contoso.com") == "dev@contoso.com"
    with pytest.raises(gid.IdentityError):
        gid.resolve_sponsor({}, git=lambda: None)


def test_key_dir_env(tmp_path):
    assert gid.key_dir({gid.KEY_DIR_ENV: str(tmp_path)}) == tmp_path
    assert gid.key_dir({}) == gid.DEFAULT_KEY_DIR


def test_role_identity_persists_and_signs(store, tmp_path):
    a = store.role("proposer", ["repo:read"])
    assert str(a.did).startswith("did:mesh:") and a.sponsor_email == SPONSOR
    payload = {"b": 1, "a": [1, 2]}
    sig = gid.sign(a, payload)
    again = gid.IdentityStore(tmp_path / "keys", sponsor=SPONSOR).role("proposer", ["repo:read"])
    assert again.did == a.did and gid.verify(again, payload, gid.sign(again, payload))
    assert gid.verify(a.to_jwk(), {"a": [1, 2], "b": 1}, sig)  # canonical: key order irrelevant
    assert not gid.verify(a, {"a": [1, 2], "b": 2}, sig)
    assert not gid.verify({"kty": "nope"}, payload, sig)
    a.revoke("test")
    assert not gid.verify(a, payload, sig)
    with pytest.raises(gid.IdentityError):
        gid.sign(a, payload)


def test_capability_drift_and_bad_names_rejected(store):
    store.role("critic", ["repo:read"])
    with pytest.raises(gid.IdentityError, match="capabilities differ"):
        store.role("critic", ["repo:read", "repo:write"])
    with pytest.raises(gid.IdentityError):
        store.agent("../escape", [])


def test_delegation_narrows_and_chain_verifies(store):
    orch = store.orchestrator()
    d = store.workflow_run("r1", ["eval:run", "repo:read"])
    assert d.identity.parent_did == str(orch.did) and d.identity.delegation_depth == 1
    assert d.verify() == (True, None)
    assert d.chain.get_effective_capabilities() == ["eval:run", "repo:read"]
    assert store.workflow_run("r1", ["eval:run", "repo:read"]).identity.did == d.identity.did


def test_delegation_widening_rejected(store):
    orch = store.orchestrator(["eval:run"])
    with pytest.raises(gid.IdentityError, match="not in parent"):
        store.workflow_run("r2", ["eval:run", "pr:publish"], parent=orch)
    with pytest.raises(gid.IdentityError):
        store.workflow_run("r3", ["*"], parent=orch)


def test_tampered_child_key_file_cannot_widen(store, tmp_path):
    orch = store.orchestrator(["eval:run", "repo:read"])
    store.workflow_run("r4", ["eval:run"], parent=orch)
    p = tmp_path / "keys" / "run-r4.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["identity"]["capabilities"] = ["eval:run", "pr:publish"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(gid.IdentityError, match="scope chain"):
        store.workflow_run("r4", ["eval:run", "pr:publish"], parent=orch)
