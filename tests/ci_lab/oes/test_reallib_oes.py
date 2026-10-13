"""Every committed OES envelope validates against the vendored schemas; tampering is caught.

Also builds a sleep envelope from the real order-support evaluator pin (a ``sha256:``-prefixed
tree digest), which the OES ``evaluatorTree`` pattern only accepts as bare hex.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ci_lab.domain.order_support import OrderSupportDomain
from ci_lab.oes import validate_envelope
from ci_lab.oes.build import sleep_envelope
from ci_lab.oes.validate import iter_rule_ids, validate_file

REPO = Path(__file__).resolve().parents[3]
DASHBOARD = REPO / ".github" / "extensions" / "ci-harness-dashboard" / "test" / "fixtures" / "repo" / "experiments"


def _committed_envelopes() -> list[Path]:
    fixtures = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
    dash = sorted(p for p in DASHBOARD.rglob("*.json")
                  if p.name in {"envelope.json", "experiment.json"} or p.parent.name == "envelopes")
    return fixtures + dash


ENVELOPES = _committed_envelopes()


def test_envelope_inventory_covers_every_kind():
    kinds = {json.loads(p.read_text(encoding="utf-8"))["extensions"].get("com.microsoft.ci.sleep") is not None
             for p in ENVELOPES}
    assert len(ENVELOPES) >= 8 and kinds == {True, False}


@pytest.mark.parametrize("path", ENVELOPES, ids=lambda p: p.relative_to(REPO).as_posix())
def test_committed_envelope_validates(path):
    assert validate_file(path) == []


@pytest.mark.parametrize("path", ENVELOPES[:4], ids=lambda p: p.stem)
def test_tampered_envelope_is_rejected(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(doc)
    tampered["experiment"]["name"] = str(tampered["experiment"].get("name", "")) + " (edited)"
    errors = validate_envelope(tampered)
    assert errors and any("contentHash" in e or "hash" in e.lower() for e in errors), errors
    broken = copy.deepcopy(doc)
    broken.pop("decision")
    assert validate_envelope(broken), "an envelope without a decision must not validate"
    assert iter_rule_ids(errors)


def test_sleep_envelope_from_real_order_support_pin_validates(fx):
    pin = OrderSupportDomain(cases=[]).pin()
    assert pin.evaluator_tree.startswith("sha256:") and pin.judge_provider == "s1"
    doc = sleep_envelope(
        "2026-10-08", incumbent=fx.make_eval(fx.T0, pin=pin), candidate=fx.make_eval(fx.T1, base=0.65, pin=pin),
        incumbent_commit=fx.H0, skillopt_version="0.2.0", tasks_by_origin={"reviewed": 9},
        tasks_by_split={"evolve": 9}, skillopt_gate={"passed": True, "mode": "on", "score": 0.7,
                                                   "baselineScore": 0.6},
        assert_gate={"passed": True, "ciLowerBound": 0.01}, delta=0.02, budget_used={"tasks": 9},
        budget_limits={"tasks": 50}, candidate_digest="sha256:" + "3" * 64,
        incumbent_digest="sha256:" + "4" * 64, night_index=1,
        skill_path="src/order_support/harness/skills/order-support/SKILL.md",
        exported_at=fx.AT, source_version=fx.VERSION)
    assert validate_envelope(doc) == []
    ext_pin = doc["extensions"]["com.microsoft.ci.sleep"]["evaluatorPin"]
    assert ext_pin["evaluatorTree"] == pin.evaluator_tree.removeprefix("sha256:")
    assert ext_pin["judgeModel"].startswith("s1/")
