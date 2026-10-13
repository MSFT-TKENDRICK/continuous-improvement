"""File-queue approval resolver bound to the ACS enforced identity (spec §17).

A liftable deny writes ``<root>/<enforced_identity>.json`` (status ``pending``, content-free) and
suspends the action. ``ci-lab governance approve|deny <identity>`` records the decision; the
next identical attempt (same enforced identity) is allowed once, then the approval is consumed.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ci_lab.governance.acs import ApprovalResolution

__all__ = ["APPROVALS_ENV", "DEFAULT_APPROVALS_DIR", "FileApprovalQueue"]

APPROVALS_ENV = "CI_GOVERNANCE_APPROVALS"
DEFAULT_APPROVALS_DIR = Path("artifacts/governance/approvals")
_IDENTITY = re.compile(r"^[0-9a-f]{64}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FileApprovalQueue:
    """Callable ACS approval resolver ``(point, result) -> ApprovalResolution``."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or os.environ.get(APPROVALS_ENV) or DEFAULT_APPROVALS_DIR)

    @staticmethod
    def key(identity: str) -> str:
        """``sha256:<hex>`` or ``<hex>`` -> ``<hex>``; anything else is rejected (no path tricks)."""
        hexid = identity.removeprefix("sha256:") if isinstance(identity, str) else ""
        if not _IDENTITY.fullmatch(hexid):
            raise ValueError("approval identity must be a 64-hex enforced_identity")
        return hexid

    def path(self, identity: str) -> Path:
        return self.root / f"{self.key(identity)}.json"

    def read(self, identity: str) -> dict[str, Any] | None:
        identity, p = self.key(identity), self.path(identity)
        try:
            data = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and data.get("enforced_identity") == identity else None

    def _write(self, identity: str, data: dict[str, Any]) -> None:
        p = self.path(identity)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, p)

    def decide(self, identity: str, decision: str, *, reason: str | None = None,
               by: str | None = None) -> dict[str, Any]:
        """Record ``approve``/``deny`` for a pending (or not yet requested) identity."""
        if decision not in ("approve", "deny"):
            raise ValueError("decision must be 'approve' or 'deny'")
        data = {**(self.read(identity) or {"enforced_identity": self.key(identity)}),
                "status": "approved" if decision == "approve" else "denied",
                "decided_at": _now(), "by": by, "note": reason}
        self._write(identity, data)
        return data

    def pending(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        out = [self.read(p.stem) for p in sorted(self.root.glob("*.json")) if _IDENTITY.fullmatch(p.stem)]
        return [d for d in out if d and d.get("status") == "pending"]

    def __call__(self, point: str, result: Any) -> ApprovalResolution:
        identity = result.enforced_identity
        data = self.read(identity)
        status = (data or {}).get("status")
        if status == "approved":
            self._write(identity, {**data, "status": "consumed", "consumed_at": _now()})
            return ApprovalResolution.allow(identity)
        if status in ("denied", "consumed"):
            return ApprovalResolution.deny(f"approval {status}")
        if data is None:
            self._write(identity, {"enforced_identity": self.key(identity), "status": "pending", "point": point,
                                   "reason": result.verdict.reason, "approval": result.verdict.approval,
                                   "requested_at": _now()})
        return ApprovalResolution.suspend(identity, identity)
