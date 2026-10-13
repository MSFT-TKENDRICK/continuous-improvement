import json

import pytest
from pydantic import ValidationError

from ci_lab import contracts
from ci_lab.rulespec import (
    CorrectionRecord,
    Fingerprint,
    GuardView,
    RuleFile,
    RuleSpec,
    TemplateSpec,
    Trajectory,
    TrajectoryStep,
    attempt_digest,
    bundle_digest,
    guard_result_json,
)

REFUND_RULE = {
    "id": "refund.requires_verified_identity",
    "version": 1,
    "rung": "R2",
    "on": "tool_call",
    "target": "issue_refund",
    "require": {"kind": "all", "of": [
        {"kind": "state", "flag": "identity_verified", "subject": "current.args.customer_id"},
        {"kind": "prior", "tool": "lookup_order", "status": "ok",
         "where": {"kind": "arg", "path": "prior.result.status", "op": "in", "value": ["delivered", "lost"]},
         "same": [["current.args.order_id", "prior.args.order_id"]]},
    ]},
    "action": "block",
    "template": "precondition.missing",
    "slots": {"tool": "issue_refund", "precondition": "verify_identity"},
    "mode": "shadow",
}


def test_rule_roundtrip_and_digest_is_order_independent():
    r = RuleSpec.model_validate(REFUND_RULE)
    r2 = RuleSpec.model_validate({**REFUND_RULE, "id": "refund.cap", "require": {
        "kind": "arg", "path": "current.args.amount", "op": "le", "value": 500}})
    assert RuleSpec.model_validate_json(r.model_dump_json()) == r
    assert bundle_digest([r, r2]) == bundle_digest([r2, r]) != bundle_digest([r])
    RuleFile.model_validate({"schema_version": 1, "rules": [REFUND_RULE]})


@pytest.mark.parametrize("patch", [
    {"extra_key": 1},                                      # unknown keys rejected
    {"target": "*"},                                       # '*' cannot block (B2)
    {"rung": "R3"},                                        # rung/on mismatch
    {"action": "redact"},                                  # redact only on response
    {"id": "Bad Id"},
    {"require": {"kind": "arg", "path": "prior.args.x", "op": "eq", "value": 1}},     # namespace (B5)
    {"require": {"kind": "prior", "tool": "t", "where": {
        "kind": "arg", "path": "current.args.x", "op": "eq", "value": 1}}},           # current inside prior
    {"require": {"kind": "prior", "tool": "t", "where": {"kind": "prior", "tool": "u"}}},  # nested prior
    {"require": {"kind": "text", "matches": "x"}},          # text only on response
    {"require": {"kind": "arg", "path": "current.args.x", "op": "matches", "value": "a" * 300}},  # B7 cap
    {"require": {"kind": "arg", "path": "current.args.x", "op": "in", "value": 3}},
    {"require": {"kind": "eval", "code": "1"}},             # closed union
    {"require": {"kind": "state", "flag": "f", "subject": "prior.args.x"}},
    {"rung": "R1", "require": {"kind": "state", "flag": "identity_verified"}},  # R1 arg-only
])
def test_rule_validation_rejects(patch):
    with pytest.raises(ValidationError):
        RuleSpec.model_validate({**REFUND_RULE, **patch})


def test_response_rule_and_templates():
    RuleSpec.model_validate({"id": "pii.before_verify", "version": 1, "rung": "R3", "on": "response",
                             "target": "*", "action": "redact", "template": "pii.redacted",
                             "when": {"kind": "not", "of": {"kind": "state", "flag": "identity_verified"}},
                             "require": {"kind": "not", "of": {"kind": "text", "matches": r"\b\d{16}\b"}}})
    TemplateSpec(id="precondition.missing", message="{tool} requires {precondition} first.",
                 fix="Call {precondition}, then retry.", slots=["tool", "precondition"])
    with pytest.raises(ValidationError):
        TemplateSpec(id="x", message="{undeclared}", fix="", slots=[])


def test_guard_view_is_closed_and_results_are_canonical_json():
    assert set(GuardView.model_fields) == {"steps", "pending"}
    with pytest.raises(ValidationError):
        GuardView.model_validate({"steps": [], "case_id": "c01"})
    step = TrajectoryStep(i=3, kind="tool_call", tool="issue_refund", args={"order_id": "A1", "amount": 5})
    assert attempt_digest(step) == attempt_digest(TrajectoryStep.model_validate(step.model_dump()))
    s = guard_result_json("r.x", "no", "do y", "§13")
    assert isinstance(s, str) and json.loads(s) == {"guard": {"rule": "r.x", "violation": "no",
                                                              "fix": "do y", "see": "§13"}}
    assert s == guard_result_json("r.x", "no", "do y", "§13")


def test_lessons_never_ingest_sealed_splits_and_records_have_no_text():
    with pytest.raises(ValidationError):
        Trajectory(id="t", source="assert", split="heldout", family="f", slice="s", pin="p",
                   trusted=True, steps=())
    assert "excerpt" not in CorrectionRecord.model_fields and "text" not in CorrectionRecord.model_fields
    fp = Fingerprint(pin="p", oracle_rules=("refund.unverified_identity",),
                     tool_ngrams=(("lookup_order", "issue_refund", "respond"),))
    assert fp.digest() == Fingerprint.model_validate(fp.model_dump()).digest()


def test_guard_strategy_and_span_constants():
    assert "guard" in contracts.STRATEGIES
    assert contracts.SPAN_GUARD == "ci.guard" and contracts.ATTR_GUARD_RULE.startswith("ci.guard.")


def test_cmp_and_extractors():
    from ci_lab.rulespec import ExtractorFile, normalize_subject

    r = RuleSpec.model_validate({**REFUND_RULE, "require": {
        "kind": "prior", "tool": "lookup_order", "same": [["current.args.order_id", "prior.args.order_id"]],
        "cmp": [{"current": "current.args.amount", "op": "le", "prior": "prior.result.total"}]}})
    assert r.require.cmp[0].op == "le"
    with pytest.raises(ValidationError):
        RuleSpec.model_validate({**REFUND_RULE, "require": {
            "kind": "prior", "tool": "t", "cmp": [{"current": "prior.args.a", "op": "le", "prior": "prior.result.b"}]}})
    ExtractorFile.model_validate({"extractors": [{"flag": "identity_verified", "tool": "verify_identity",
                                                  "result_path": "result.verified", "subject": "args.order_id",
                                                  "ttl_steps": 50}]})
    with pytest.raises(ValidationError):
        ExtractorFile.model_validate({"extractors": [{"flag": "x_y", "tool": "t", "result_path": "text",
                                                      "subject": "args.order_id"}]})
    assert normalize_subject(" A1001 ") == normalize_subject("a1001")
