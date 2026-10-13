"""Agent identities on AGT AgentMesh: Ed25519 DIDs, a human sponsor, narrowing delegation.

One persisted ``agentmesh.identity.AgentIdentity`` per agent role and per workflow run.
Run identities are delegated from the campaign-orchestrator identity; AGT enforces that
capabilities only narrow (``AgentIdentity.delegate``) and the resulting ``ScopeChain`` is
hash-linked and signature-checked. Keys live under ``$CI_GOVERNANCE_KEY_DIR`` or
``artifacts/governance/identity/`` (gitignored); everything works offline.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

with warnings.catch_warnings():  # AGT 5.x compat shims warn on import; we pin the core package
    warnings.simplefilter("ignore", DeprecationWarning)
    from agentmesh.identity import (
        AgentIdentity,
        DelegationLink,
        HumanSponsor,
        ScopeChain,
    )

__all__ = [
    "KEY_DIR_ENV", "ORCHESTRATOR", "ORCHESTRATOR_CAPABILITIES", "SPONSOR_ENV",
    "AgentIdentity", "Delegation", "IdentityError", "IdentityStore",
    "canonical_bytes", "key_dir", "resolve_sponsor", "sign", "verify",
]

SPONSOR_ENV = "CI_GOVERNANCE_SPONSOR"
KEY_DIR_ENV = "CI_GOVERNANCE_KEY_DIR"
DEFAULT_KEY_DIR = Path("artifacts") / "governance" / "identity"
ORCHESTRATOR = "campaign-orchestrator"
ORCHESTRATOR_CAPABILITIES: tuple[str, ...] = (
    "campaign:launch", "arm:run", "workflow:run", "eval:run", "judge:run", "repo:read",
    "repo:write", "pr:publish",
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class IdentityError(RuntimeError):
    """Identity could not be established or verified (callers must fail closed)."""


def _git_email() -> str | None:
    try:
        out = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True,
                             timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def resolve_sponsor(env: Mapping[str, str] | None = None, *,
                    git: Callable[[], str | None] = _git_email) -> str:
    """Accountable human: ``$CI_GOVERNANCE_SPONSOR`` else git ``user.email``. No orphan agents."""
    email = (os.environ if env is None else env).get(SPONSOR_ENV, "").strip() or git()
    if not email:
        raise IdentityError(f"no human sponsor: set {SPONSOR_ENV} or git user.email")
    return email


def key_dir(env: Mapping[str, str] | None = None) -> Path:
    return Path((os.environ if env is None else env).get(KEY_DIR_ENV) or DEFAULT_KEY_DIR)


def canonical_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sign(identity: AgentIdentity, payload: Any) -> str:
    """Base64 Ed25519 signature over the canonical JSON of ``payload``."""
    if not identity.is_active():
        raise IdentityError(f"identity {identity.did} is {identity.status}")
    return identity.sign(canonical_bytes(payload))


def verify(signer: AgentIdentity | Mapping[str, Any], payload: Any, signature: str) -> bool:
    """Verify with an identity or its public JWK; inactive or malformed signers fail closed."""
    try:
        ident = signer if isinstance(signer, AgentIdentity) else AgentIdentity.from_jwk(dict(signer))
    except Exception:  # noqa: BLE001 - any malformed key is a verification failure
        return False
    return ident.is_active() and ident.verify_signature(canonical_bytes(payload), signature)


@dataclass(frozen=True)
class Delegation:
    identity: AgentIdentity
    chain: ScopeChain

    def verify(self) -> tuple[bool, str | None]:
        return self.chain.verify()


class IdentityStore:
    """Persisted identities (JSON metadata + private JWK per agent) under one key directory."""

    def __init__(self, root: Path | str | None = None, *, sponsor: str | None = None) -> None:
        self.root = Path(root) if root is not None else key_dir()
        self.sponsor = HumanSponsor.create(email=sponsor or resolve_sponsor())

    def _path(self, name: str) -> Path:
        if not _NAME.fullmatch(name):
            raise IdentityError(f"invalid identity name {name!r}")
        return self.root / f"{name}.json"

    def _load(self, name: str) -> AgentIdentity | None:
        p = self._path(name)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        meta = AgentIdentity.model_validate(data["identity"])
        keyed = AgentIdentity.from_jwk(data["jwk"])
        if keyed.public_key != meta.public_key or str(keyed.did) != str(meta.did):
            raise IdentityError(f"key/metadata mismatch for {name}")
        return keyed.model_copy(update=dict(meta))  # keeps the restored private key

    def _save(self, name: str, ident: AgentIdentity) -> None:
        p = self._path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {"identity": ident.model_dump(mode="json"), "jwk": ident.to_jwk(include_private=True)}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, p)

    def _check(self, name: str, ident: AgentIdentity, capabilities: Sequence[str],
               parent: AgentIdentity | None) -> AgentIdentity:
        if sorted(ident.capabilities) != sorted(capabilities):
            raise IdentityError(f"{name}: persisted capabilities differ; rotate the key to change")
        if ident.sponsor_email != self.sponsor.email:
            raise IdentityError(f"{name}: sponsored by a different human")
        if parent is not None and ident.parent_did != str(parent.did):
            raise IdentityError(f"{name}: delegated by a different parent")
        return ident

    def agent(self, name: str, capabilities: Sequence[str]) -> AgentIdentity:
        """Load or mint a root (sponsor-held) identity; the sponsor must be able to grant caps."""
        if found := self._load(name):
            return self._check(name, found, capabilities, None)
        denied = [c for c in capabilities if not self.sponsor.can_grant_capability(c)]
        if denied:
            raise IdentityError(f"sponsor cannot grant {denied}")
        ident = AgentIdentity.create(name=name, sponsor=self.sponsor.email,
                                     capabilities=list(capabilities))
        self._save(name, ident)
        return ident

    def role(self, role: str, capabilities: Sequence[str]) -> AgentIdentity:
        return self.agent(f"role-{role}", capabilities)

    def orchestrator(self, capabilities: Sequence[str] = ORCHESTRATOR_CAPABILITIES) -> AgentIdentity:
        return self.agent(ORCHESTRATOR, capabilities)

    def delegate(self, parent: AgentIdentity, name: str,
                 capabilities: Sequence[str]) -> Delegation:
        """Child identity whose capabilities are a subset of ``parent``'s (widening raises)."""
        child = self._load(name)
        if child is None:
            try:
                child = parent.delegate(name=name, capabilities=list(capabilities))
            except ValueError as exc:
                raise IdentityError(str(exc)) from exc
            self._save(name, child)
        self._check(name, child, capabilities, parent)
        chain, root = ScopeChain.create_root(self.sponsor.email, str(parent.did),
                                             list(parent.capabilities))
        chain.known_identities[str(parent.did)] = parent
        chain.add_link(root)
        caps = list(child.capabilities)
        link = DelegationLink(
            link_id=f"link_{uuid.uuid4().hex[:12]}", depth=1, parent_did=str(parent.did),
            child_did=str(child.did), parent_capabilities=list(parent.capabilities),
            delegated_capabilities=caps, previous_link_hash=root.link_hash, link_hash="",
            parent_signature=parent.sign(f"{parent.did}:{child.did}:{','.join(sorted(caps))}".encode()),
        )
        link.link_hash = link.compute_hash()
        try:
            chain.add_link(link)
        except ValueError as exc:  # e.g. a tampered key file that widens capabilities
            raise IdentityError(f"scope chain rejected: {exc}") from exc
        ok, why = chain.verify()
        if not ok:
            raise IdentityError(f"scope chain invalid: {why}")
        return Delegation(child, chain)

    def workflow_run(self, run_id: str, capabilities: Sequence[str], *,
                     parent: AgentIdentity | None = None) -> Delegation:
        """Per-run identity delegated from ``parent`` (default: the campaign orchestrator)."""
        return self.delegate(parent or self.orchestrator(), f"run-{run_id}", capabilities)
