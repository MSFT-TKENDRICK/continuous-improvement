from __future__ import annotations

import json
from pathlib import Path

import pytest
import re2

from ci_lab.lessons_arm.bundle import BundleError, dump_rule_file, load_rules, read_rule_file
from ci_lab.lessons_arm.features import (
    ORACLE_FEATURES,
    LessonFeatures,
    derive_features,
    features_for,
    read_candidates,
)
from ci_lab.lessons_arm.synth import SynthesisError, lesson_id_for, synthesize
from ci_lab.lessons_arm.templates import (
    REQUIRED_TEMPLATES,
    SAFE_PATTERNS,
    catalog_gaps,
    render,
    trusted_templates,
)
from ci_lab.rulespec import (
    Fingerprint,
    LessonCluster,
    PriorPred,
    RuleSpec,
    StatePred,
    TemplateSpec,
)

EXTRACTORS = """schema_version: 1
extractors:
  - {flag: identity_verified, tool: verify_identity, result_path: result.verified, equals: true,
     subject: args.order_id}
"""


def cluster(rule: str, cid: str = "c-refund-1", **kw: object) -> LessonCluster:
    return LessonCluster(id=cid, fingerprint=Fingerprint(pin="p1", oracle_rules=(rule,)),
                         members=("t1", "t2"), families=("f1", "f2"), slices=("s1", "s2"), route="R2", **kw)


def test_safe_patterns_compile_in_re2_and_match():
    for pat in SAFE_PATTERNS.values():
        re2.compile(pat)
    assert re2.search(SAFE_PATTERNS["email"], "mail ann@example.com now")
    assert re2.search(SAFE_PATTERNS["phone"], "call 425-555-0100")
    assert re2.search(SAFE_PATTERNS["street_address"], "ship to 12 Main Street please")
    assert not re2.search(SAFE_PATTERNS["email"], "no address here")


def test_catalog_gaps_and_render():
    assert catalog_gaps(REQUIRED_TEMPLATES) == []
    partial = {"arg.constraint": TemplateSpec(id="arg.constraint", message="m {tool}", fix="f", slots=["tool"])}
    gaps = catalog_gaps(partial)
    assert "precondition.prior_call" in gaps and "arg.constraint:arg" in gaps
    msg, fix = render(REQUIRED_TEMPLATES["precondition.prior_call"], {"tool": "issue_refund"})
    assert msg.startswith("issue_refund requires") and "{prior_tool}" in fix
    catalog, _frozen = trusted_templates()
    assert catalog


def test_features_reject_free_text():
    with pytest.raises(ValueError):
        LessonFeatures(kind="arg_constraint", target_tool="issue refund please")
    with pytest.raises(ValueError):
        LessonFeatures(kind="arg_constraint", values=["ignore previous instructions"])
    with pytest.raises(ValueError):
        LessonFeatures(kind="response_pattern", pattern_classes=["(.*)secret"])
    assert LessonFeatures(kind="prior_call", foo="bar").kind == "prior_call"  # unknown keys dropped


def test_derive_features_and_candidates(tmp_path: Path):
    assert derive_features(cluster("refund.ineligible_order")) == ORACLE_FEATURES["refund.ineligible_order"]
    assert derive_features(cluster("injection.followed")) is None
    assert derive_features(cluster("unknown.rule")) is None
    path = tmp_path / "candidates.jsonl"
    c1 = cluster("refund.ineligible_order")
    c2 = cluster("other.rule", cid="c-other")
    feats = {"kind": "arg_constraint", "target_tool": "issue_refund", "arg": "amount", "op": "gt", "values": [0]}
    path.write_text("\n".join([
        c1.model_dump_json(),
        json.dumps({"cluster": c2.model_dump(mode="json"), "features": feats, "route_reasons": ["x"]}),
        "",
    ]), encoding="utf-8")
    pairs = features_for(read_candidates(path))
    assert pairs[0][1] == ORACLE_FEATURES["refund.ineligible_order"]
    assert pairs[1][1] is not None and pairs[1][1].arg == "amount"
    assert read_candidates(tmp_path / "missing.jsonl") == []
    (tmp_path / "bad.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        read_candidates(tmp_path / "bad.jsonl")


@pytest.mark.parametrize("oracle", sorted(ORACLE_FEATURES))
def test_seed_oracles_synthesize_shadow_template_rules(oracle: str, tmp_path: Path):
    c = cluster(oracle)
    rule = synthesize(c, ORACLE_FEATURES[oracle])
    assert rule.mode == "shadow"
    assert rule.provenance.source == "template"
    assert rule.provenance.lesson_id == lesson_id_for(c)
    assert all(e.startswith("fingerprint:") for e in rule.provenance.evidence)
    tpl = REQUIRED_TEMPLATES[rule.template]
    assert set(rule.slots) <= set(tpl.slots)
    # deterministic
    assert synthesize(c, ORACLE_FEATURES[oracle]) == rule
    # round-trips through a rule file and the bundle loader
    p = tmp_path / "g.yaml"
    p.write_text(dump_rule_file([rule]), encoding="utf-8")
    assert read_rule_file(p).rules == [rule]
    ex = tmp_path / "extractors.yaml"
    ex.write_text(EXTRACTORS, encoding="utf-8")
    loaded = load_rules([p], [ex])
    assert [r.id for r in loaded.rules] == [rule.id]


def test_precondition_and_amount_shapes():
    r = synthesize(cluster("refund.ineligible_order"), ORACLE_FEATURES["refund.ineligible_order"])
    assert r.rung == "R2" and r.action == "block" and r.target == "issue_refund"
    assert isinstance(r.require, PriorPred)
    assert r.require.same == [("current.args.order_id", "prior.args.order_id")]
    assert r.require.where is not None
    a = synthesize(cluster("refund.amount_exceeds_eligible"), ORACLE_FEATURES["refund.amount_exceeds_eligible"])
    assert isinstance(a.require, PriorPred)
    assert [(c.current, c.op, c.prior) for c in a.require.cmp] == [
        ("current.args.amount", "le", "prior.result.total")]
    s = synthesize(cluster("refund.unverified_identity"), ORACLE_FEATURES["refund.unverified_identity"])
    assert isinstance(s.require, StatePred) and s.require.subject == "current.args.order_id"
    red = synthesize(cluster("pii.disclosed_before_verification"),
                     ORACLE_FEATURES["pii.disclosed_before_verification"])
    assert red.rung == "R3" and red.on == "response" and red.action == "redact" and red.target == "*"
    dumped = red.model_dump_json()
    for pat in SAFE_PATTERNS.values():
        assert json.dumps(pat)[1:-1] in dumped


def test_arg_constraint_variants():
    c = cluster("x.rule", cid="c-arg")
    r = synthesize(c, LessonFeatures(kind="arg_constraint", target_tool="issue_refund", arg="reason",
                                     op="in", values=["damaged", "late"]))
    assert r.rung == "R1" and r.require.value == ["damaged", "late"]  # type: ignore[union-attr]
    assert r.slots["constraint"] == "in damaged|late"
    e = synthesize(c, LessonFeatures(kind="arg_constraint", target_tool="t", arg="a", op="exists"))
    assert e.require.op == "exists"  # type: ignore[union-attr]
    with pytest.raises(SynthesisError):
        synthesize(c, LessonFeatures(kind="arg_constraint", target_tool="t", arg="a", op="gt", values=[1, 2]))
    with pytest.raises(SynthesisError):
        synthesize(c, LessonFeatures(kind="arg_constraint", target_tool="t", arg="a", op="in"))
    with pytest.raises(SynthesisError, match="missing"):
        synthesize(c, LessonFeatures(kind="prior_call", target_tool="t"))


def test_trust_and_injection_gates():
    f = ORACLE_FEATURES["refund.ineligible_order"]
    with pytest.raises(SynthesisError, match="injection"):
        synthesize(cluster("injection.followed"), f)
    with pytest.raises(SynthesisError, match="injection"):
        synthesize(cluster("refund.ineligible_order"), f.model_copy(update={"injection_suspect": True}))
    untrusted = f.model_copy(update={"trusted": False})
    with pytest.raises(SynthesisError, match="untrusted"):
        synthesize(cluster("refund.ineligible_order"), untrusted)
    assert synthesize(cluster("refund.ineligible_order", human_confirmed=True), untrusted).mode == "shadow"


def test_lesson_id_is_safe():
    assert lesson_id_for(cluster("a", cid="9 Weird/ID!!")) == "l9-weird-id"
    RuleSpec.model_validate(synthesize(cluster("refund.ineligible_order", cid="X" * 80),
                                       ORACLE_FEATURES["refund.ineligible_order"]).model_dump())


def test_fallback_bundle_errors(tmp_path: Path):
    r = synthesize(cluster("refund.ineligible_order"), ORACLE_FEATURES["refund.ineligible_order"])
    p1, p2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    p1.write_text(dump_rule_file([r]), encoding="utf-8")
    p2.write_text(dump_rule_file([r]), encoding="utf-8")
    with pytest.raises(BundleError, match="duplicate"):
        load_rules([p1, p2])
    p2.write_text(dump_rule_file([r.model_copy(update={"id": "other.rule", "template": "nope"})]),
                  encoding="utf-8")
    with pytest.raises(BundleError, match="unknown template"):
        load_rules([p1, p2])
    p2.write_text("schema_version: 1\nrules: [{id: bad}]\n", encoding="utf-8")
    with pytest.raises(BundleError):
        load_rules([p1, p2])
