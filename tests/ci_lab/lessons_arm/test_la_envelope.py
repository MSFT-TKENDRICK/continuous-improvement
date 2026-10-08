from __future__ import annotations

import pytest

from ci_lab.lessons_arm.envelope import EXT_VERSION, guard_extension, holdout_look_required
from ci_lab.rulespec import OES_GUARD_EXT, GuardMetrics

M = GuardMetrics(bundle_digest="sha256:abc", paired=True, trials=3, attempted_violation_rate=0.4,
                 delivered_violation_rate=0.1, task_completion=0.8, false_denial_rate=0.01, block_rate=0.05,
                 opportunities=120, fires=12, recall=0.75, substitutions=2)


def test_holdout_look_required():
    assert not holdout_look_required("evolve") and not holdout_look_required("aa")
    assert holdout_look_required("heldout") and holdout_look_required("ood") and holdout_look_required("confirm")
    assert holdout_look_required("whatever")


def test_extension_payload_evolve():
    ext = guard_extension(M, split="evolve", bundle_digest="sha256:abc", arm="arm-g", rule_ids=["b", "a"],
                          lesson_ids=["l1"], ship=(True, []), incumbent_digest="sha256:old")
    p = ext[OES_GUARD_EXT]
    assert p["version"] == EXT_VERSION and p["kind"] == "guard_eval"
    assert p["bundleDigest"] == "sha256:abc" and p["deliveredViolationRate"] == 0.1
    assert p["falseDenialRate"] == 0.01 and p["opportunities"] == 120 and p["substitutions"] == 2
    assert p["agentSafetyRate"] == 0.4  # attempted, never the guarded rate
    assert p["ruleIds"] == ["a", "b"] and p["ship"] == {"ok": True, "reasons": []}
    assert p["holdoutLook"] is False and "holdout" not in p
    assert all("_" not in k for k in p)


def test_confirm_split_counts_as_look():
    with pytest.raises(ValueError, match="C15"):
        guard_extension(M, split="heldout")
    p = guard_extension(M, split="heldout", dataset_hash="sha256:ds")[OES_GUARD_EXT]
    assert p["holdoutLook"] and p["holdout"] == {"datasetHash": "sha256:ds", "plannedLooks": 1, "looksUsed": 1}
    with pytest.raises(ValueError, match="budget"):
        guard_extension(M, split="heldout", dataset_hash="d", looks_used=2)
    with pytest.raises(ValueError, match="mismatch"):
        guard_extension(M, split="evolve", bundle_digest="sha256:other")
