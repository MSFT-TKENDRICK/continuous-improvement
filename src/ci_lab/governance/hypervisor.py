"""Execution rings, kill switch and per-ring rate limits for harness actions (AGT hypervisor).

Ring 3 = read-only, ring 2 = reversible, ring 1 = irreversible (requires human approval),
ring 0 = admin (never granted to harness actions). Unknown actions fail closed to ring 1.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from hypervisor import (
        ActionClassifier,
        AgentRateLimiter,
        ExecutionRing,
        KillSwitch,
        ReversibilityLevel,
    )
    from hypervisor.models import ActionDescriptor
    from hypervisor.security.kill_switch import KillReason, KillResult

__all__ = [
    "DEFAULT_KILL_FILE", "IRREVERSIBLE_ACTIONS", "KILL_ENV", "READ_ONLY_ACTIONS",
    "REVERSIBLE_ACTIONS", "ActionGuard", "KillSwitchAdapter", "RingDecision", "check_action",
]

KILL_ENV = "CI_KILL_SWITCH"
DEFAULT_KILL_FILE = Path("artifacts/governance/KILL")
HARNESS_DID = "did:mesh:ci-harness"

READ_ONLY_ACTIONS = frozenset({
    "lookup_order", "verify_identity", "search_kb", "read_file", "list_files", "git_status",
    "git_diff", "git_log", "fetch_trace", "judge", "eval",
})
REVERSIBLE_ACTIONS = frozenset({
    "escalate_to_human", "write_file", "edit_file", "run_tests", "git_commit", "create_branch",
    "draft_pr", "write_artifact",
})
IRREVERSIBLE_ACTIONS = frozenset({
    "issue_refund", "publish_pr", "push", "git_push", "merge_pr", "delete_branch", "send_email",
    "publish_lesson",
})


@dataclass(frozen=True)
class RingDecision:
    ring: int
    allowed: bool
    reason: str
    requires_approval: bool


def _profile(action: str, reversible: bool | None, read_only: bool | None
             ) -> tuple[bool, ReversibilityLevel]:
    """Known actions use the table; hints may only tighten them. Unknown actions use hints."""
    if action in READ_ONLY_ACTIONS:
        ro, rev = True, ReversibilityLevel.FULL
    elif action in REVERSIBLE_ACTIONS:
        ro, rev = False, ReversibilityLevel.FULL
    elif action in IRREVERSIBLE_ACTIONS:
        ro, rev = False, ReversibilityLevel.NONE
    else:
        ro = read_only is True and reversible is not False
        rev = ReversibilityLevel.FULL if (reversible is True or (ro and reversible is None)) \
            else ReversibilityLevel.NONE
        return ro, rev
    if read_only is False:
        ro = False
    if reversible is False:
        ro, rev = False, ReversibilityLevel.NONE
    return ro, rev


class KillSwitchAdapter:
    """File/env-engaged kill switch; engagements are recorded through ``hypervisor.KillSwitch``."""

    def __init__(self, path: str | Path | None = None, *, env: Mapping[str, str] | None = None):
        self.path = Path(path) if path is not None else DEFAULT_KILL_FILE
        self._env = env
        self.switch = KillSwitch()

    def engaged(self) -> bool:
        env = os.environ if self._env is None else self._env
        flag = env.get(KILL_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        return flag or self.path.exists()

    def engage(self, reason: str, *, agent_did: str = HARNESS_DID,
               session_id: str = "harness") -> KillResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(reason.strip() + "\n", encoding="utf-8")
        return self.switch.kill(agent_did, session_id, KillReason.MANUAL, details=reason)

    def release(self) -> bool:
        """Remove the kill file. Returns False if still engaged (e.g. via the env var)."""
        self.path.unlink(missing_ok=True)
        return not self.engaged()


class ActionGuard:
    """Classifies actions into rings and enforces kill switch, approval and per-ring rate limits."""

    def __init__(self, run_id: str = "local", *, kill_switch: KillSwitchAdapter | None = None,
                 ring_limits: Mapping[int, tuple[float, float]] | None = None):
        self.run_id = run_id
        self.kill_switch = kill_switch or KillSwitchAdapter()
        self.classifier = ActionClassifier()
        limits = None if ring_limits is None else {
            ExecutionRing(r): (float(rate), float(cap)) for r, (rate, cap) in ring_limits.items()}
        self.limiter = AgentRateLimiter(ring_limits=limits)

    def classify(self, action: str, *, reversible: bool | None = None,
                 read_only: bool | None = None) -> ExecutionRing:
        ro, rev = _profile(action, reversible, read_only)
        desc = ActionDescriptor(action_id=action, name=action, execute_api=f"ci_lab:{action}",
                                reversibility=rev, is_read_only=ro)
        return ExecutionRing(self.classifier.classify(desc).ring)

    def check_action(self, agent: object, action: str, *, reversible: bool | None = None,
                     read_only: bool | None = None, approved: bool = False) -> RingDecision:
        ring = self.classify(action, reversible=reversible, read_only=read_only)
        needs_approval = ring <= ExecutionRing.RING_1_PRIVILEGED
        if self.kill_switch.engaged():
            return RingDecision(int(ring), False, "kill_switch_engaged", needs_approval)
        if needs_approval and not approved:
            return RingDecision(int(ring), False, "approval_required", True)
        did = str(getattr(agent, "did", agent))
        if not self.limiter.try_check(did, f"{self.run_id}:ring{int(ring)}", ring):
            return RingDecision(int(ring), False, "rate_limited", needs_approval)
        return RingDecision(int(ring), True, "approved" if needs_approval else "ok", needs_approval)


def check_action(agent: object, action: str, *, reversible: bool | None = None,
                 read_only: bool | None = None, approved: bool = False,
                 guard: ActionGuard | None = None) -> RingDecision:
    """One-shot check with a fresh guard (default kill file/env); pass ``guard`` to share limits."""
    return (guard or ActionGuard()).check_action(
        agent, action, reversible=reversible, read_only=read_only, approved=approved)
