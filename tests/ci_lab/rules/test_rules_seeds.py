"""Golden tests: §13.7 seed lessons over order-support fixture trajectories."""

from ci_lab.rules import TRAJECTORY_END, compute_flags, evaluate, evaluate_trajectory
from ci_lab.rulespec import GuardView, TrajectoryStep

REFUND_RULES = ["refund.amount_exceeds_eligible", "refund.ineligible_order", "refund.requires_verified_identity"]


def test_happy_path_refund_passes_all_seeds(seeds, tb, fires):
    tb.lookup("A1001", total=80.0)
    tb.verify("A1001")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="a1001 ", amount=80.0)) == []


def test_unverified_refund_blocked(seeds, tb, fires):
    tb.lookup("A1001")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1001", amount=10)) == [
        "refund.requires_verified_identity"]


def test_b5_identity_for_different_order_does_not_satisfy(seeds, tb, fires):
    tb.lookup("A1001")
    tb.lookup("B2002")
    tb.verify("B2002")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1001", amount=10)) == [
        "refund.requires_verified_identity"]
    assert compute_flags(seeds, tb.steps) == {"identity_verified": {"b2002"}}


def test_failed_or_errored_verification_sets_no_flag_and_latest_result_wins(seeds, tb):
    tb.verify("A1", verified=False)
    assert compute_flags(seeds, tb.steps) == {}
    tb.call("verify_identity", order_id="A2")
    tb.result({"verified": True, "order_id": "A2"}, status="error")
    assert compute_flags(seeds, tb.steps) == {}
    tb.verify("A3")
    assert compute_flags(seeds, tb.steps) == {"identity_verified": {"a3"}}
    tb.verify("A3", verified=False)  # later evidence revokes
    assert compute_flags(seeds, tb.steps) == {}
    tb.verify("A3", verified="true")  # type-strict: string "true" is not True
    assert compute_flags(seeds, tb.steps) == {}


def test_flag_requires_paired_successful_result(seeds, tb):
    tb.call("verify_identity", order_id="A1")  # no result yet
    assert compute_flags(seeds, tb.steps) == {}
    tb.call("verify_identity", status="blocked", order_id="A9")
    tb.result({"verified": True, "order_id": "A9"}, status="ok")  # result paired to a blocked call
    assert "a9" not in compute_flags(seeds, tb.steps).get("identity_verified", set())


def test_ttl_expiry_counts_tool_calls(seeds, tb, fires):
    tb.lookup("A1")
    tb.verify("A1")                 # ttl_steps: 6
    for _ in range(5):
        tb.ok("search_kb", {"hits": 0}, query="policy")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1", amount=1)) == []  # 6th call: valid
    tb.ok("search_kb", {"hits": 0}, query="policy")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1", amount=1)) == [
        "refund.requires_verified_identity"]  # 7th call: expired
    assert compute_flags(seeds, tb.steps) == {}
    tb.verify("A1")                 # re-verification refreshes
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1", amount=1)) == []


def test_ineligible_and_limit_exceeded_orders(seeds, tb, fires):
    tb.lookup("A1", eligible=False)
    tb.verify("A1")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1", amount=1)) == ["refund.ineligible_order"]
    tb2 = type(tb)()
    tb2.lookup("A1", exceeded=True)
    tb2.verify("A1")
    assert fires(seeds, tb2.steps, tb2.pending("issue_refund", order_id="A1", amount=1)) == [
        "refund.ineligible_order"]


def test_amount_cmp_against_same_prior(seeds, tb, fires):
    tb.lookup("A1", total=40.0)
    tb.lookup("B2", total=500.0)
    tb.verify("A1")
    p = tb.pending("issue_refund", order_id="A1", amount=40)
    assert fires(seeds, tb.steps, p) == []
    p = tb.pending("issue_refund", order_id="A1", amount=40.01)
    assert fires(seeds, tb.steps, p) == ["refund.amount_exceeds_eligible"]  # B2's total must not satisfy
    p = tb.pending("issue_refund", order_id="A1", amount="40")
    assert fires(seeds, tb.steps, p) == ["refund.amount_exceeds_eligible"]  # str amount: no coercion


def test_errored_lookup_does_not_satisfy_prior_ok(seeds, tb, fires):
    tb.call("lookup_order", order_id="A1")
    tb.result({"order_id": "A1", "status": "delivered", "total": 9.0, "refund_eligible": True,
               "refund_limit_exceeded": False}, status="error")
    tb.verify("A1")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1", amount=1)) == [
        "refund.amount_exceeds_eligible", "refund.ineligible_order"]


def test_pairing_by_call_id_and_by_next_result(seeds, tb, fires):
    # Parallel calls; results arrive out of order but carry call ids.
    tb.call("lookup_order", call_id="x1", order_id="A1")
    tb.call("lookup_order", call_id="x2", order_id="B2")
    tb.result({"order_id": "B2", "total": 5.0, "refund_eligible": False, "refund_limit_exceeded": False},
              call_id="x2")
    tb.result({"order_id": "A1", "total": 50.0, "refund_eligible": True, "refund_limit_exceeded": False},
              call_id="x1")
    tb.verify("A1")
    assert fires(seeds, tb.steps, tb.pending("issue_refund", order_id="A1", amount=50)) == []
    # Without call ids: the next tool_result for the same tool pairs with the earliest open call.
    t2 = type(tb)()
    t2.call("lookup_order", call_id="", order_id="A1")
    t2.result({"order_id": "A1", "total": 50.0, "refund_eligible": True, "refund_limit_exceeded": False},
              call_id="")
    t2.verify("A1")
    assert fires(seeds, t2.steps, t2.pending("issue_refund", order_id="A1", amount=50)) == []


def test_result_after_pending_is_invisible(seeds, tb):
    tb.lookup("A1")
    tb.call("verify_identity", order_id="A1")
    refund = tb.call("issue_refund", order_id="A1", amount=5)
    tb.result({"verified": True, "order_id": "A1"}, tool="verify_identity", call_id="c2")
    got = [m.rule.id for m in evaluate_trajectory(seeds, tb.steps) if m.step_index == refund.i]
    assert got == ["refund.requires_verified_identity"]


def test_verify_before_lookup_anti_gaming(seeds, tb, fires):
    tb.lookup("A1", pii=True)
    assert fires(seeds, tb.steps, tb.pending("verify_identity", order_id="A1")) == ["verify.before_lookup"]
    assert fires(seeds, tb.steps, tb.pending("verify_identity", order_id="B2")) == []
    t2 = type(tb)()
    t2.lookup("A1")
    assert fires(seeds, t2.steps, t2.pending("verify_identity", order_id="A1")) == []


def test_blocked_status_and_count_and_within(mk, fires, tb):
    b = mk(
        {"id": "t.no_retry_after_block", "require": {"kind": "count", "tool": "issue_refund", "status": "blocked",
                                                     "op": "lt", "n": 1}},
        {"id": "t.recent_lookup", "require": {"kind": "prior", "tool": "lookup_order", "status": "any",
                                              "within": 2}},
        {"id": "t.max_calls", "target": "escalate_to_human", "action": "warn",
         "require": {"kind": "count", "tool": "*", "op": "le", "n": 3}},
    )
    tb.lookup("A1")
    assert fires(b, tb.steps, tb.pending("issue_refund", order_id="A1")) == []
    tb.call("issue_refund", status="blocked", order_id="A1")
    assert fires(b, tb.steps, tb.pending("issue_refund", order_id="A1")) == ["t.no_retry_after_block"]
    tb.ok("search_kb", {"hits": 1}, query="q")
    # lookup_order is now 3 tool calls back => outside within: 2
    assert fires(b, tb.steps, tb.pending("issue_refund", order_id="A1")) == ["t.no_retry_after_block",
                                                                              "t.recent_lookup"]
    assert fires(b, tb.steps, tb.pending("escalate_to_human")) == []
    tb.ok("search_kb", {"hits": 1}, query="q")
    assert fires(b, tb.steps, tb.pending("escalate_to_human")) == ["t.max_calls"]


def test_prior_any_excludes_blocked_calls(mk, fires, tb):
    b = mk({"require": {"kind": "prior", "tool": "lookup_order", "status": "any"}})
    tb.call("lookup_order", status="blocked", order_id="A1")
    assert fires(b, tb.steps, tb.pending("issue_refund")) == ["t.rule0"]
    tb.call("lookup_order", order_id="A1")  # attempted, no result yet => counts for `any`
    assert fires(b, tb.steps, tb.pending("issue_refund")) == []


def test_response_pii_redaction_before_verification(seeds, tb, fires):
    tb.lookup("A1")
    p = tb.pending_response("Email jo@example.com or call +1 (555) 010-0000; ship to 12 Elm Street.")
    assert fires(seeds, tb.steps, p, on="response") == ["pii.disclosed_before_verification"]
    assert fires(seeds, tb.steps, tb.pending_response("Your order A1 shipped."), on="response") == []
    tb.verify("A1")
    assert fires(seeds, tb.steps, p, on="response") == []


def test_r4_per_step_and_whole_trajectory(seeds, tb):
    esc = tb.call("escalate_to_human", reason="x")
    tb.result({"ticket": "T1"})
    tb.lookup("A1", total=100.0)
    tb.verify("A1")
    for _ in range(2):
        tb.call("issue_refund", order_id="A1", amount=10)
        tb.result({"refund_id": "R"})
    got = [(m.rule.id, m.step_index) for m in evaluate_trajectory(seeds, tb.steps)]
    assert got == [("traj.escalate_after_lookup", esc.i), ("traj.single_refund", TRAJECTORY_END)]
    whole = evaluate(seeds, GuardView(steps=tuple(tb.steps)), on="trajectory")
    assert [m.rule.id for m in whole] == ["traj.single_refund"] and whole[0].step_index == -1


def test_evaluate_trajectory_order_and_all_matches(seeds, tb):
    tb.lookup("A1", pii=True)
    tb.call("verify_identity", order_id="A1", status="blocked")
    tb.call("issue_refund", order_id="A1", amount=999)
    tb.respond("Reach jo@example.com")
    got = [(m.step_index, m.rule.rung, m.rule.id) for m in evaluate_trajectory(seeds, tb.steps)]
    assert got == [(2, "R2", "verify.before_lookup"),
                   (3, "R2", "refund.amount_exceeds_eligible"), (3, "R2", "refund.requires_verified_identity"),
                   (4, "R3", "pii.disclosed_before_verification")]
    assert got == sorted(got, key=lambda x: (x[0], x[1], x[2]))


def test_evaluate_rejects_wrong_pending_kind(seeds, tb):
    import pytest

    with pytest.raises(ValueError):
        evaluate(seeds, GuardView(pending=tb.pending_response("x")), on="tool_call")
    with pytest.raises(ValueError):
        evaluate(seeds, GuardView(), on="response")
    assert evaluate(seeds, GuardView(pending=TrajectoryStep(i=0, kind="tool_call", tool="nope")),
                    on="tool_call") == []


def test_engine_is_pure(seeds, tb):
    """Same input => same output; inputs are not mutated."""
    tb.lookup("A1")
    before = [s.model_dump() for s in tb.steps]
    a = evaluate_trajectory(seeds, tb.steps)
    assert a == evaluate_trajectory(seeds, tb.steps)
    assert [s.model_dump() for s in tb.steps] == before
