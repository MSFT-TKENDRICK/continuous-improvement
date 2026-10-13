"""ACS host obligations (spec §5, §17, §19): enforcement, the approval path and content-free audit.

:class:`AgentControl` wraps an :class:`~ci_lab.governance.acs.runtime.AcsRuntime` (or anything with
the same ``evaluate_intervention_point`` coroutine). In ``enforce`` mode a final ``deny`` raises
:class:`AgentControlBlocked`; a liftable ``deny`` is routed to the approval resolver, whose approval
is honoured only when the enforced identity rederived from the current policy input still matches.
In ``evaluate_only`` mode verdicts are recorded and the action proceeds untransformed.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ci_lab.governance.acs.canonical import AcsError, JsonValue, action_identity
from ci_lab.governance.acs.manifest import parse_path, resolve
from ci_lab.governance.acs.runtime import (
    MODES,
    Decision,
    InterventionPointResult,
    Verdict,
    _call,
    _replace,
)

DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0
DEFAULT_FATIGUE_WINDOW_SECONDS = 3600
UNRESOLVED = "host_error:approval_unresolved"
RESOLVER_FAILED = "host_error:approval_resolver_failed"
IDENTITY_MISMATCH = "host_error:approval_identity_mismatch"

DecisionLog = Callable[[Mapping[str, JsonValue]], Any]


class AgentControlInterruption(RuntimeError):
    """A policy-driven interruption (block or suspension), distinct from ordinary errors."""

    def __init__(self, verb: str, point: str, result: InterventionPointResult) -> None:
        reason = f" ({result.verdict.reason})" if result.verdict.reason else ""
        super().__init__(f"Agent Control Specification {verb} {point}{reason}.")
        self.intervention_point = point
        self.result = result


class AgentControlBlocked(AgentControlInterruption):
    """A ``deny`` was enforced; at a ``post_*`` point the action has already run."""

    def __init__(
        self, intervention_point: str, result: InterventionPointResult
    ) -> None:
        super().__init__("blocked", intervention_point, result)


class AgentControlSuspended(AgentControlInterruption):
    """The approval path suspended the run; resumption is owned by the host via ``handle``."""

    def __init__(
        self,
        intervention_point: str,
        result: InterventionPointResult,
        handle: JsonValue = None,
    ) -> None:
        super().__init__("suspended", intervention_point, result)
        self.handle = handle


class ApprovalOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    SUSPEND = "suspend"


@dataclass(frozen=True, slots=True)
class ApprovalResolution:
    """An approval decision; allow and suspend carry the approved enforced identity (§17.1)."""

    outcome: ApprovalOutcome
    handle: JsonValue = None
    action_identity: str | None = None
    reason: str | None = None

    @classmethod
    def allow(cls, action_identity: str) -> ApprovalResolution:
        return cls(ApprovalOutcome.ALLOW, action_identity=action_identity)

    @classmethod
    def deny(cls, reason: str | None = None) -> ApprovalResolution:
        return cls(ApprovalOutcome.DENY, reason=reason)

    @classmethod
    def suspend(
        cls, handle: JsonValue = None, action_identity: str | None = None
    ) -> ApprovalResolution:
        return cls(ApprovalOutcome.SUSPEND, handle, action_identity)


def _host_deny(
    result: InterventionPointResult, reason: str, message: str
) -> InterventionPointResult:
    return replace(
        result, verdict=Verdict(Decision.DENY, reason=reason, message=message)
    )


class AgentControl:
    """Host enforcement over a runtime; ``approval_resolver`` is a callable or a name → callable map."""

    def __init__(
        self,
        runtime: Any,
        approval_resolver: Callable[..., Any]
        | Mapping[str, Callable[..., Any]]
        | None = None,
        mode: str = "enforce",
        *,
        decision_log: DecisionLog | None = None,
        approval_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        manifest = getattr(runtime, "manifest", None)
        self.runtime, self.mode, self.decision_log = runtime, mode, decision_log
        self.approval: Mapping[str, JsonValue] = manifest.approval if manifest else {}
        self._resolver = approval_resolver
        self._clock = clock
        self._timeout = approval_timeout_seconds or self.approval.get(
            "timeout_seconds", DEFAULT_APPROVAL_TIMEOUT_SECONDS
        )
        self._consulted: deque[float] = deque()

    async def guard(
        self, point: str, snapshot: Mapping[str, JsonValue], tool: str | None = None
    ) -> InterventionPointResult:
        """Evaluate and enforce one intervention point; returns the result when the action may proceed."""
        result = await self.runtime.evaluate_intervention_point(
            point, snapshot, self.mode, tool
        )
        try:
            approval = await self._enforce(point, result)
        except AgentControlInterruption as exc:
            outcome = "suspend" if isinstance(exc, AgentControlSuspended) else None
            self._record(point, exc.result, outcome)
            raise
        self._record(point, result, approval)
        return result

    async def run(
        self, value: JsonValue, action: Callable[[JsonValue], Any]
    ) -> JsonValue:
        """Guard ``input``, run ``action`` on the effective input, guard ``output``; returns the output."""
        snapshot = {"input": value}
        effective = self._effective(
            await self.guard("input", snapshot), snapshot, "input"
        )
        output = await _call(action, effective)
        snapshot = {"input": effective, "output": output}
        return self._effective(await self.guard("output", snapshot), snapshot, "output")

    def _effective(
        self, result: InterventionPointResult, snapshot: dict[str, JsonValue], key: str
    ) -> JsonValue:
        """The value at ``key`` with an enforced transform reinserted at the policy target path."""
        if result.verdict.decision is not Decision.TRANSFORM or self.mode != "enforce":
            return snapshot[key]
        try:
            path = parse_path(result.policy_input["policy_target"]["path"])
            updated = _replace(
                snapshot, path.segments, result.transformed_policy_target
            )
            return resolve(updated, (key,))
        except (AcsError, ValueError, KeyError, TypeError):
            denial = _host_deny(
                result, "host_error:transform_invalid", "Transform not applicable."
            )
            raise AgentControlBlocked(key, denial) from None

    def _record(
        self, point: str, result: InterventionPointResult, approval: str | None
    ) -> None:
        if self.decision_log is None:
            return
        self.decision_log(
            {
                "point": point,
                "decision": str(result.verdict.decision),
                "reason": result.verdict.reason,
                "approval": approval,
                "input_identity": result.input_identity,
                "enforced_identity": result.enforced_identity,
                "mode": self.mode,
                "ts": self._clock(),
            }
        )

    def _resolver_for(self) -> Callable[..., Any] | None:
        if isinstance(self._resolver, Mapping):
            return self._resolver.get(self.approval.get("default_resolver"))
        return self._resolver

    def _fatigued(self) -> bool:
        threshold = self.approval.get("fatigue_threshold")
        if not threshold:
            return False
        now = self._clock()
        window = self.approval.get(
            "fatigue_window_seconds", DEFAULT_FATIGUE_WINDOW_SECONDS
        )
        while self._consulted and self._consulted[0] <= now - window:
            self._consulted.popleft()
        if len(self._consulted) >= threshold:
            return True
        self._consulted.append(now)
        return False

    async def _enforce(self, point: str, result: InterventionPointResult) -> str | None:
        """Apply §17 to one result; returns the approval outcome when a liftable deny was approved."""
        if self.mode != "enforce" or result.verdict.decision is not Decision.DENY:
            return None
        if not result.verdict.liftable:
            raise AgentControlBlocked(point, result)
        resolver = self._resolver_for()
        if resolver is None or self._fatigued():
            raise AgentControlBlocked(
                point, _host_deny(result, UNRESOLVED, "Approval did not resolve.")
            )
        original = result.action_identity
        try:
            resolution = await asyncio.wait_for(
                _call(resolver, point, result), self._timeout
            )
        except TimeoutError:
            on_timeout = self.approval.get("on_timeout", "deny")
            if on_timeout == "deny":
                raise AgentControlBlocked(
                    point, _host_deny(result, UNRESOLVED, "Approval timed out.")
                ) from None
            resolution = ApprovalResolution(
                ApprovalOutcome(on_timeout), action_identity=original
            )
        except AgentControlInterruption:
            raise
        except Exception:  # noqa: BLE001 - a failing resolver fails closed
            raise AgentControlBlocked(
                point, _host_deny(result, RESOLVER_FAILED, "Approval resolver failed.")
            ) from None
        if isinstance(resolution, ApprovalOutcome):
            resolution = ApprovalResolution(resolution, action_identity=original)
        if not isinstance(resolution, ApprovalResolution):
            raise AgentControlBlocked(
                point, _host_deny(result, RESOLVER_FAILED, "Unrecognized approval.")
            )
        if resolution.outcome is ApprovalOutcome.DENY:
            raise AgentControlBlocked(
                point,
                replace(
                    result,
                    verdict=replace(
                        result.verdict,
                        message=resolution.reason or result.verdict.message,
                    ),
                ),
            )
        current = (
            action_identity(result.policy_input)
            if result.policy_input is not None
            else None
        )
        if original is None or not original == current == resolution.action_identity:
            raise AgentControlBlocked(
                point, _host_deny(result, IDENTITY_MISMATCH, "Approved action changed.")
            )
        if resolution.outcome is ApprovalOutcome.SUSPEND:
            raise AgentControlSuspended(point, result, resolution.handle)
        return str(resolution.outcome)
