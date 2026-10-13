"""Content-free, hash-chained audit of governance decisions on AGT ``AuditEntry``/``AuditChain``.

Each decision becomes an ``agentmesh.governance.AuditEntry`` (event_type
``ci.governance.decision``) whose AGT ``compute_hash`` covers the timestamp, agent DID,
intervention point, outcome and the fixed ``data`` fields below, linked by
``previous_hash``. AGT's ``AuditLog`` is in-memory only, so entries are persisted one JSON
line each here; ``verify`` re-checks every entry with AGT ``AuditEntry.verify_hash`` plus the
link and replays them into an AGT ``AuditChain`` to report its Merkle root. Records carry
identities and reason codes only, never prompts, messages or tool arguments (ACS §19).
"""
from __future__ import annotations

import os
import re
import threading
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace

with warnings.catch_warnings():  # AGT 5.x compat shims warn on import
    warnings.simplefilter("ignore", DeprecationWarning)
    from agentmesh.governance import AuditChain, AuditEntry

__all__ = [
    "AUDIT_PATH_ENV", "DECISIONS", "EVENT", "INTERVENTION_POINTS", "MODES",
    "AuditError", "AuditTrail", "DecisionRecord", "VerifyResult", "envelope_extension", "verify",
]

EVENT = "ci.governance.decision"
AUDIT_PATH_ENV = "CI_GOVERNANCE_AUDIT"
DEFAULT_AUDIT_PATH = Path("artifacts") / "governance" / "audit" / "decisions.jsonl"
INTERVENTION_POINTS = frozenset({
    "agent_startup", "input", "pre_model_call", "post_model_call", "pre_tool_call",
    "post_tool_call", "output", "agent_shutdown",
})
DECISIONS = frozenset({"allow", "deny", "transform"})
MODES = frozenset({"enforce", "evaluate_only"})
_DATA_KEYS = ("reason", "input_identity", "enforced_identity", "mode", "run", "eid")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REASON = re.compile(r"[A-Za-z0-9_.:/ -]{0,160}")
_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_TAIL = 65536


class AuditError(RuntimeError):
    """Invalid record or broken chain (callers must fail closed)."""


@dataclass(frozen=True)
class DecisionRecord:
    agent_did: str
    intervention_point: str
    decision: str
    reason: str = ""
    input_identity: str | None = None
    enforced_identity: str | None = None
    mode: str = "enforce"
    run: str | None = None
    eid: str | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        bad = []
        if not self.agent_did.startswith("did:") or not _ID.fullmatch(self.agent_did):
            bad.append("agent_did")
        if self.intervention_point not in INTERVENTION_POINTS:
            bad.append("intervention_point")
        if self.decision not in DECISIONS:
            bad.append("decision")
        if self.mode not in MODES:
            bad.append("mode")
        if not _REASON.fullmatch(self.reason):
            bad.append("reason (a short code, not content)")
        bad += [k for k in ("input_identity", "enforced_identity")
                if getattr(self, k) is not None and not _SHA256.fullmatch(getattr(self, k))]
        bad += [k for k in ("run", "eid") if getattr(self, k) is not None
                and not _ID.fullmatch(getattr(self, k))]
        if self.ts.tzinfo is None:
            bad.append("ts (must be tz-aware)")
        if bad:
            raise AuditError(f"invalid decision record fields: {', '.join(bad)}")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> DecisionRecord:
        """Strict: any key outside the fixed schema (e.g. prompt, args, snapshot) is rejected."""
        extra = sorted(set(m) - set(cls.__dataclass_fields__))
        if extra:
            raise AuditError(f"audit records are content-free; unexpected fields: {extra}")
        return cls(**dict(m))


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    entries: int = 0
    head: str | None = None
    merkle_root: str | None = None
    decisions: int = 0
    denies: int = 0
    error: str | None = None


def _read_entries(path: Path) -> list[AuditEntry]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AuditEntry.model_validate_json(ln) for ln in lines if ln.strip()]


def verify(path: Path | str) -> VerifyResult:
    """Recheck every entry hash (AGT) and link; never raises on a bad file."""
    try:
        entries = _read_entries(Path(path))
    except Exception as exc:  # noqa: BLE001 - unparsable line == broken chain
        return VerifyResult(False, error=f"unparsable audit file: {type(exc).__name__}")
    chain, prev, denies = AuditChain(), "", 0
    for i, e in enumerate(entries):
        if e.event_type != EVENT or set(e.data) != set(_DATA_KEYS):
            return VerifyResult(False, i, error=f"entry {i} is not a governance decision")
        if not e.verify_hash():
            return VerifyResult(False, i, error=f"entry {i} hash mismatch")
        # AGT's entry hash omits policy_decision; outcome is hashed, so they must agree.
        if (e.policy_decision == "deny") != (e.outcome == "denied"):
            return VerifyResult(False, i, error=f"entry {i} decision/outcome mismatch")
        if e.previous_hash != prev:
            return VerifyResult(False, i, error=f"entry {i} chain broken")
        chain.add_entry(e.model_copy())
        prev, denies = e.entry_hash, denies + (e.policy_decision == "deny")
    return VerifyResult(True, len(entries), prev or None, chain.get_root_hash(), len(entries),
                        denies)


class AuditTrail:
    """Append-only JSONL decision chain (one writer per file; appends are thread-safe)."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or os.environ.get(AUDIT_PATH_ENV) or DEFAULT_AUDIT_PATH)
        self._lock = threading.Lock()

    def head(self) -> str | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open("rb") as fh:
            fh.seek(max(0, self.path.stat().st_size - _TAIL))
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        return AuditEntry.model_validate_json(lines[-1]).entry_hash if lines else None

    def append(self, record: DecisionRecord | Mapping[str, Any]) -> AuditEntry:
        rec = record if isinstance(record, DecisionRecord) else DecisionRecord.from_mapping(record)
        data = {k: v for k, v in asdict(rec).items() if k in _DATA_KEYS}
        with self._lock:
            entry = AuditEntry(
                timestamp=rec.ts, event_type=EVENT, agent_did=rec.agent_did,
                action=rec.intervention_point, data=data, policy_decision=rec.decision,
                outcome="denied" if rec.decision == "deny" else "success",
                previous_hash=self.head() or "",
            )
            entry.entry_hash = entry.compute_hash()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(entry.model_dump_json() + "\n")
        _emit(rec, entry.entry_hash)
        return entry

    def verify(self) -> VerifyResult:
        return verify(self.path)

    def oes_extension(self) -> dict[str, dict[str, Any]]:
        """OES envelope extension correlating a run with this chain; broken chain raises."""
        res = self.verify()
        if not res.ok:
            raise AuditError(f"audit chain invalid: {res.error}")
        return {"x-ci-governance": {"audit_head": res.head, "decisions": res.decisions,
                                    "denies": res.denies}}


def envelope_extension(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """``x-ci-governance`` for an OES envelope; ``{}`` before any decision was audited. A broken
    chain raises :class:`AuditError` (fail closed: no envelope vouches for a tampered trail)."""
    trail = AuditTrail(path)
    return trail.oes_extension() if trail.path.exists() else {}


def _emit(rec: DecisionRecord, entry_hash: str) -> None:
    attrs = {
        "ci.governance.agent_did": rec.agent_did, "ci.governance.point": rec.intervention_point,
        "ci.governance.decision": rec.decision, "ci.governance.reason": rec.reason,
        "ci.governance.mode": rec.mode, "ci.governance.input_identity": rec.input_identity,
        "ci.governance.enforced_identity": rec.enforced_identity, "ci.governance.run": rec.run,
        "ci.governance.eid": rec.eid, "ci.governance.audit_hash": entry_hash,
    }
    trace.get_current_span().add_event(EVENT, attributes={k: v for k, v in attrs.items()
                                                          if v is not None})
