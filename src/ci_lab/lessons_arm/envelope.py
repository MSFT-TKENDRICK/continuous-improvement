"""OES envelope extension ``com.microsoft.ci.guard`` (rulespec.OES_GUARD_EXT) for guard arms.

Payload is camelCase with ``version`` (OES ext convention, see ``ci_lab.oes``). Registering the
extension JSON schema with the OES validator is HOOK(M4); so is appending the C15 look to the
global ledger ``experiments/holdout-looks.jsonl`` when :func:`holdout_look_required` is true.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ci_lab.rulespec import OES_GUARD_EXT, GuardMetrics

EXT_VERSION = "0.1.0"
NO_LOOK_SPLITS = frozenset({"evolve", "aa"})


def holdout_look_required(split: str) -> bool:
    """C15/B2: a guard arm evaluated on anything but the evolve (or A/A calibration) split — confirm,
    sealed held-out, OOD — counts as one held-out look. Unknown splits count (fail-safe)."""
    return split not in NO_LOOK_SPLITS


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


def guard_extension(metrics: GuardMetrics, *, split: str, bundle_digest: str | None = None, arm: str | None = None,
                    rule_ids: Sequence[str] = (), lesson_ids: Sequence[str] = (),
                    ship: tuple[bool, Sequence[str]] | None = None, incumbent_digest: str | None = None,
                    dataset_hash: str | None = None, looks_used: int = 1, planned_looks: int = 1) -> dict[str, Any]:
    """``{OES_GUARD_EXT: payload}`` for ``envelope["extensions"]``.

    ``bundle_digest`` (if given) must equal ``metrics.bundle_digest`` — the metrics are only valid for
    the exact bundle evaluated. When :func:`holdout_look_required` is true a ``holdout`` block with
    ``datasetHash`` is mandatory (C15), mirroring the rrsi extension's ``holdout`` shape.
    """
    if bundle_digest is not None and bundle_digest != metrics.bundle_digest:
        raise ValueError(f"bundle digest mismatch: metrics for {metrics.bundle_digest}, envelope for {bundle_digest}")
    payload: dict[str, Any] = {"version": EXT_VERSION, "kind": "guard_eval", "split": split}
    payload.update({_camel(k): v for k, v in metrics.model_dump(mode="json").items()})
    # B1: the agent's own safety is the guard-off attempted rate; guards are never credited to it.
    payload["agentSafetyRate"] = metrics.attempted_violation_rate
    if arm is not None:
        payload["arm"] = arm
    if rule_ids:
        payload["ruleIds"] = sorted(rule_ids)
    if lesson_ids:
        payload["lessonIds"] = sorted(lesson_ids)
    if incumbent_digest is not None:
        payload["incumbentBundleDigest"] = incumbent_digest
    if ship is not None:
        payload["ship"] = {"ok": bool(ship[0]), "reasons": list(ship[1])}
    payload["holdoutLook"] = holdout_look_required(split)
    if payload["holdoutLook"]:
        if not dataset_hash:
            raise ValueError(f"split {split!r} is a C15 held-out look: dataset_hash is required")
        if not 1 <= looks_used <= planned_looks:
            raise ValueError(f"held-out look budget exceeded: {looks_used}/{planned_looks}")
        payload["holdout"] = {"datasetHash": dataset_hash, "plannedLooks": planned_looks, "looksUsed": looks_used}
    return {OES_GUARD_EXT: payload}
