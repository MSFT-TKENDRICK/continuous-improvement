"""Campaign launch gate: ACS ``agent_startup`` on the packaged ``campaign`` policy.

Refuses a launch when the kill switch is engaged, the SRE/token budget is exhausted or an arm's
edit scope reaches a protected glob; a non-dry-run publish is a liftable deny routed to the file
approval queue (``ci-lab governance approve <identity>``, then retry). The snapshot is
deterministic, so the retry has the same enforced identity as the held attempt.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ci_lab.governance.acs import AgentControlSuspended
from ci_lab.governance.acs.host import AgentControlInterruption

__all__ = ["LaunchRefused", "arm_scopes", "check_launch"]


class LaunchRefused(RuntimeError):
    """The launch gate denied (or held for approval) a campaign command."""

    def __init__(self, reason: str, message: str, *, identity: str | None = None, held: bool = False) -> None:
        super().__init__(message)
        self.reason, self.message, self.identity, self.held = reason, message, identity, held

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": "governance_refused", "reason": self.reason, "message": self.message,
                               "held": self.held, "identity": self.identity}
        if self.held and self.identity:
            out["approve"] = f"ci-lab governance approve {self.identity}"
        return out


def arm_scopes(component_globs: Mapping[str, Iterable[str]],
               surface_globs: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Edit scopes any arm may receive: one entry per component plus the whole writable surface."""
    out = [{"id": c, "edit_scope": sorted(set(g))} for c, g in sorted(component_globs.items())]
    if surface_globs:
        out.append({"id": "surface", "edit_scope": sorted(set(surface_globs))})
    return out


async def check_launch(cid: str, *, arms: Sequence[Mapping[str, Any]], publish: bool, dry_run: bool,
                       budget_exhausted: bool = False, governance: Any = None) -> dict[str, Any]:
    """Evaluate the launch; returns ``{decision, reason, mode}`` or raises :class:`LaunchRefused`."""
    if governance is None:
        from ci_lab.governance.maf import Governance  # lazy: imports agent_framework

        governance = Governance("campaign", agent_name=f"campaign-{cid}", role="orchestrator")
    snap = {"campaign": {"id": cid, "publish": bool(publish), "dry_run": bool(dry_run),
                         "budget_exhausted": bool(budget_exhausted), "arms": [dict(a) for a in arms]}}
    try:
        result = await governance.check("agent_startup", snap)
    except AgentControlInterruption as exc:
        v, identity = exc.result.verdict, exc.result.enforced_identity
        raise LaunchRefused(v.reason or "denied", v.message or "campaign launch refused by governance",
                            identity=identity.removeprefix("sha256:") if identity else None,
                            held=isinstance(exc, AgentControlSuspended)) from None
    v = result.verdict if result is not None else None
    verdict = str(v.decision) if v else "allow"
    # Enforce: returning means allowed (a liftable deny only gets here once approved); evaluate_only
    # reports the would-be decision without refusing.
    effective = "allow" if governance.mode == "enforce" else verdict
    return {"decision": effective, "verdict": verdict, "approved": effective == "allow" and verdict == "deny",
            "reason": v.reason if v else None, "mode": governance.mode}
