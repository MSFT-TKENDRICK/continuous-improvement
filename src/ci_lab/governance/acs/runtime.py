"""ACS verdicts and intervention point evaluation (spec §5-§7, §10-§16).

Dispatcher output is normalized into a :class:`Verdict`. :class:`AcsRuntime` is stateless and never mutates the caller's snapshot. Every failure yields a
``deny`` verdict carrying a reserved reason and the policy input built so far. Policy and annotator
execution is host supplied: a policy dispatcher exposes ``evaluate(invocation)`` and an annotator
dispatcher exposes ``dispatch(name, config, preliminary_policy_input)``; either may be a plain
callable and may return an awaitable.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ci_lab.governance.acs.canonical import (
    DEFAULT_LIMITS,
    AcsError,
    JsonValue,
    Limits,
    action_identity,
    canonical_bytes,
    json_depth,
)
from ci_lab.governance.acs.manifest import (
    TOOL_POINTS,
    AcsPath,
    Manifest,
    load_manifest,
    parse_path,
    resolve,
    schema_validator,
)

MODES = frozenset({"enforce", "evaluate_only"})
BLOCKED_MESSAGE = "Request blocked by Agent Control Specification."
_NO_TRANSFORM_POINTS = frozenset({"agent_startup", "agent_shutdown"})
_LIMIT = "runtime_error:resource_limit_exceeded"
_OUTPUT_INVALID = "runtime_error:policy_output_invalid"
_log = logging.getLogger(__name__)


class Decision(StrEnum):
    """Normalized decisions; the ``warn``/``escalate`` intents never survive normalization (§13.1)."""

    ALLOW = "allow"
    DENY = "deny"
    TRANSFORM = "transform"


@dataclass(frozen=True, slots=True)
class Verdict:
    decision: Decision
    reason: str | None = None
    message: str | None = None
    transform: Mapping[str, JsonValue] | None = None
    evidence: Mapping[str, JsonValue] | None = None
    result_labels: tuple[str, ...] = ()
    warnings: tuple[Mapping[str, JsonValue], ...] = ()
    approval: Mapping[str, JsonValue] | None = None

    @property
    def liftable(self) -> bool:
        """A ``deny`` carrying an ``approval`` block defers to the host approval path (§17.1)."""
        return self.decision is Decision.DENY and self.approval is not None


@dataclass(frozen=True, slots=True)
class InterventionPointResult:
    verdict: Verdict
    transformed_policy_target: JsonValue = None
    policy_input: dict[str, JsonValue] | None = None
    input_identity: str | None = None
    enforced_identity: str | None = None

    @property
    def action_identity(self) -> str | None:
        """The identity an approval binds to: the enforced identity (§13.1)."""
        return self.enforced_identity


def _plain(value: JsonValue, reason: str) -> JsonValue:
    """A detached plain-JSON copy; raises ``reason`` for non-JSON values."""
    try:
        return json.loads(canonical_bytes(value))
    except (TypeError, ValueError) as exc:
        raise AcsError(reason, f"not JSON: {exc}") from None


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    result = fn(*args)
    return await result if inspect.isawaitable(result) else result


def _replace(
    root: JsonValue, segments: tuple[str | int, ...], value: JsonValue
) -> JsonValue:
    """Return a copy of ``root`` with the existing location ``segments`` set to ``value``."""
    if not segments:
        return copy.deepcopy(value)
    out = copy.deepcopy(root)
    parent = resolve(out, segments[:-1])
    resolve(parent, segments[-1:])
    parent[segments[-1]] = copy.deepcopy(value)
    return out


def _transform_path(raw: Mapping[str, Any]) -> None:
    """Transform checks that precede schema validation so they surface their own reasons (§14)."""
    body = raw.get("transform")
    if raw.get("decision") != "transform" or not isinstance(body, Mapping):
        return
    if not isinstance(body.get("path"), str):
        raise AcsError(
            "runtime_error:transform_invalid", "transform path is not a string"
        )
    try:
        path = parse_path(body["path"])
    except ValueError as exc:
        raise AcsError("runtime_error:transform_invalid", str(exc)) from None
    if path.root != "target":
        raise AcsError("runtime_error:transform_target_forbidden", path.text)
    if "value" not in body:
        raise AcsError("runtime_error:transform_invalid", "transform value is missing")


def normalize_verdict(raw: JsonValue, *, limits: Limits = DEFAULT_LIMITS) -> Verdict:
    """Normalize dispatcher output (§13): ``warn`` → ``allow`` + warning, ``escalate`` → liftable deny."""
    raw = _plain(raw, _OUTPUT_INVALID)
    if len(canonical_bytes(raw)) > limits.max_policy_output_bytes:
        raise AcsError(_LIMIT, "max_policy_output_bytes")
    if isinstance(raw, dict):
        _transform_path(raw)
    error = next(
        iter(schema_validator("wire/verdict.schema.json").iter_errors(raw)), None
    )
    if error is not None:
        raise AcsError(_OUTPUT_INVALID, error.message)
    decision, reason, message = raw["decision"], raw.get("reason"), raw.get("message")
    warnings = list(raw.get("warnings") or [])
    approval = raw.get("approval")
    if decision == "warn":
        decision = "allow"
        warnings.append({"reason": reason, "message": message})
    elif decision == "escalate":
        decision = "deny"
        approval = {} if approval is None else approval
    if approval is not None and decision != "deny":
        raise AcsError(_OUTPUT_INVALID, "approval is only valid on a deny")
    return Verdict(
        decision=Decision(decision),
        reason=reason,
        message=message,
        transform=raw.get("transform"),
        evidence=raw.get("evidence"),
        result_labels=tuple(raw.get("result_labels") or ()),
        warnings=tuple(warnings),
        approval=approval,
    )


def denied(
    reason: str, policy_input: dict[str, JsonValue] | None = None
) -> InterventionPointResult:
    """A fail-closed ``deny`` with a reserved reason and the policy input built so far (§6)."""
    identity = action_identity(policy_input) if policy_input is not None else None
    return InterventionPointResult(
        Verdict(Decision.DENY, reason=reason, message=BLOCKED_MESSAGE),
        policy_input=policy_input,
        input_identity=identity,
        enforced_identity=identity,
    )


class AcsRuntime:
    """Evaluates intervention points of one validated manifest (spec §6 order)."""

    def __init__(
        self,
        manifest: Manifest | str | bytes | Mapping[str, Any],
        *,
        dispatcher: Any = None,
        annotator: Any = None,
        limits: Limits = DEFAULT_LIMITS,
    ) -> None:
        self.limits = limits
        self.manifest = (
            manifest
            if isinstance(manifest, Manifest)
            else load_manifest(manifest, limits=limits)
        )
        self._dispatcher = dispatcher
        self._annotator = annotator

    async def evaluate_intervention_point(
        self,
        point: str,
        snapshot: Mapping[str, JsonValue],
        mode: str = "enforce",
        tool: str | None = None,
    ) -> InterventionPointResult:
        """Evaluate one intervention point; ``tool`` names the tool when no ``tool_name_from`` is set."""
        built: list[dict[str, JsonValue]] = []
        try:
            return await self._evaluate(point, snapshot, mode, tool, built)
        except AcsError as exc:
            _log.debug("acs %s denied: %s", point, exc)
            return denied(exc.reason, built[-1] if built else None)

    def _check_depth(self, policy_input: JsonValue) -> None:
        if json_depth(policy_input) > self.limits.max_policy_input_depth:
            raise AcsError(_LIMIT, "max_policy_input_depth")

    async def _evaluate(
        self,
        point: str,
        snapshot: Mapping[str, JsonValue],
        mode: str,
        tool: str | None,
        built: list[dict[str, JsonValue]],
    ) -> InterventionPointResult:
        if mode not in MODES or not isinstance(snapshot, Mapping):
            raise AcsError("host_error:context_invalid", "mode or snapshot")
        snap = _plain(snapshot, "host_error:context_invalid")
        cfg = self.manifest.points.get(point)
        if cfg is None:
            raise AcsError("runtime_error:intervention_point_unknown", str(point))
        if len(canonical_bytes(snap)) > self.limits.max_snapshot_bytes:
            raise AcsError(_LIMIT, "max_snapshot_bytes")
        target = resolve(snap, cfg.policy_target.segments)
        projected = self._project_tool(point, cfg.tool_name_from, snap, tool)
        policy_input: dict[str, JsonValue] = {
            "intervention_point": point,
            "policy_target": {
                "kind": cfg.policy_target_kind,
                "path": cfg.policy_target.text,
                "value": target,
            },
            "snapshot": snap,
            "annotations": {},
            "tool": projected,
        }
        built.append(policy_input)
        self._check_depth(policy_input)
        if len(cfg.annotations) > self.limits.max_annotators_per_point:
            raise AcsError(_LIMIT, "max_annotators_per_point")
        roots = {"pi": policy_input, "target": target, "tool": projected, "snap": snap}
        outputs: dict[str, JsonValue] = {}
        for ann in cfg.annotations:
            value = resolve(roots[ann.source.root], ann.source.segments)
            config = {
                **self.manifest.annotators[ann.name],
                **ann.binding,
                "value": value,
            }
            outputs[ann.name] = await self._annotate(
                ann.name,
                _plain(config, "runtime_error:annotation_failed"),
                policy_input,
            )
        policy_input = {**policy_input, "annotations": outputs}
        built.append(policy_input)
        self._check_depth(policy_input)
        verdict = normalize_verdict(
            await self._dispatch(cfg.policy_id, cfg.binding, policy_input),
            limits=self.limits,
        )
        identity = action_identity(policy_input)
        if verdict.decision is not Decision.TRANSFORM:
            return InterventionPointResult(
                verdict, None, policy_input, identity, identity
            )
        if point in _NO_TRANSFORM_POINTS:
            raise AcsError("host_error:transform_target_forbidden", point)
        try:
            path = parse_path(verdict.transform["path"])
            new_target = _replace(target, path.segments, verdict.transform["value"])
            new_snap = _replace(snap, cfg.policy_target.segments, new_target)
        except AcsError as exc:
            raise AcsError("host_error:transform_invalid", exc.detail) from None
        if len(canonical_bytes(new_snap)) > self.limits.max_snapshot_bytes:
            raise AcsError(_LIMIT, "max_snapshot_bytes after transform")
        if mode != "enforce":
            return InterventionPointResult(
                verdict, None, policy_input, identity, identity
            )
        enforced = {
            **policy_input,
            "policy_target": {**policy_input["policy_target"], "value": new_target},
        }
        return InterventionPointResult(
            verdict, new_target, policy_input, identity, action_identity(enforced)
        )

    def _project_tool(
        self, point: str, name_from: AcsPath | None, snap: JsonValue, tool: str | None
    ) -> JsonValue:
        """Project the catalog entry for the invoked tool (§9); ``null`` at non-tool points."""
        if point not in TOOL_POINTS or (name_from is None and tool is None):
            if tool is not None:
                raise AcsError(
                    "host_error:context_invalid", "tool given at a non-tool point"
                )
            return None
        name = resolve(snap, name_from.segments) if name_from is not None else tool
        if not isinstance(name, str):
            raise AcsError(
                "runtime_error:path_type_mismatch", "tool name is not a string"
            )
        if tool is not None and tool != name:
            raise AcsError(
                "host_error:context_invalid", "tool disagrees with tool_name_from"
            )
        entry = self.manifest.tools.get(name)
        if entry is None:
            raise AcsError("runtime_error:tool_unknown", name)
        return {**copy.deepcopy(entry), "name": name}

    async def _annotate(
        self, name: str, config: JsonValue, policy_input: JsonValue
    ) -> JsonValue:
        """Run one annotator on an isolated copy of the preliminary policy input (§10)."""
        failed = "runtime_error:annotation_failed"
        if self._annotator is None:
            raise AcsError(failed, "no annotator dispatcher configured")
        fn = getattr(self._annotator, "dispatch", self._annotator)
        try:
            out = await _call(fn, name, config, copy.deepcopy(policy_input))
        except TimeoutError:
            raise AcsError("runtime_error:annotation_timeout", name) from None
        except AcsError as exc:
            raise AcsError(
                exc.reason
                if exc.reason == "runtime_error:annotation_timeout"
                else failed,
                name,
            ) from None
        except Exception as exc:  # noqa: BLE001 - host code fails closed
            raise AcsError(failed, f"{name}: {type(exc).__name__}") from None
        out = _plain(out, failed)
        if len(canonical_bytes(out)) > self.limits.max_annotator_output_bytes:
            raise AcsError(failed, f"{name}: output exceeds max_annotator_output_bytes")
        reason = out.get("reason") if isinstance(out, dict) else None
        if isinstance(reason, str) and reason.startswith("runtime_error:"):
            raise AcsError(failed, f"{name}: reserved reason in output")
        return out

    def _policy_fn(self, policy: Mapping[str, Any]) -> Callable[..., Any] | None:
        dispatcher = self._dispatcher
        if isinstance(dispatcher, Mapping):
            key = (
                policy.get("adapter") if policy["type"] == "custom" else policy["type"]
            )
            dispatcher = dispatcher.get(key)
        return (
            getattr(dispatcher, "evaluate", dispatcher)
            if dispatcher is not None
            else None
        )

    async def _dispatch(
        self, policy_id: str, binding: Mapping[str, JsonValue], policy_input: JsonValue
    ) -> JsonValue:
        """Call the host policy dispatcher with a typed invocation (§12.3)."""
        policy = self.manifest.policies[policy_id]
        fn = self._policy_fn(policy)
        if fn is None:
            raise AcsError(
                "host_error:adapter_unsupported", f"no dispatcher for {policy['type']}"
            )
        invocation = copy.deepcopy(
            {
                "policy_id": policy_id,
                "policy": policy,
                "binding": binding,
                "input": policy_input,
            }
        )
        try:
            return await _call(fn, invocation)
        except Exception as exc:  # noqa: BLE001 - host code fails closed
            raise AcsError(
                "runtime_error:policy_invocation_failed", type(exc).__name__
            ) from None
