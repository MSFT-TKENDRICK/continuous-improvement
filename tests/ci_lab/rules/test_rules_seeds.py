"""Golden tests: §13.7 seed lessons over harness-agent fixture trajectories."""

from ci_lab.rules import TRAJECTORY_END, compute_flags, evaluate, evaluate_trajectory
from ci_lab.rulespec import GuardView, TrajectoryStep

ACCESS_RULES = ["change.amount_exceeds_eligible", "change.disallowed_resource", "change.requires_access"]


def test_happy_path_change_passes_all_seeds(seeds, tb, fires):
    tb.inspect("A1001", total=80.0)
    tb.authorize("A1001")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="a1001 ", amount=80.0)) == []


def test_unverified_change_blocked(seeds, tb, fires):
    tb.inspect("A1001")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1001", amount=10)) == [
        "change.requires_access"]


def test_b5_identity_for_different_resource_does_not_satisfy(seeds, tb, fires):
    tb.inspect("A1001")
    tb.inspect("B2002")
    tb.authorize("B2002")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1001", amount=10)) == [
        "change.requires_access"]
    assert compute_flags(seeds, tb.steps) == {"access_verified": {"b2002"}}


def test_failed_or_errored_verification_sets_no_flag_and_latest_result_wins(seeds, tb):
    tb.authorize("A1", verified=False)
    assert compute_flags(seeds, tb.steps) == {}
    tb.call("verify_access", resource_id="A2")
    tb.result({"verified": True, "resource_id": "A2"}, status="error")
    assert compute_flags(seeds, tb.steps) == {}
    tb.authorize("A3")
    assert compute_flags(seeds, tb.steps) == {"access_verified": {"a3"}}
    tb.authorize("A3", verified=False)  # later evidence revokes
    assert compute_flags(seeds, tb.steps) == {}
    tb.authorize("A3", verified="true")  # type-strict: string "true" is not True
    assert compute_flags(seeds, tb.steps) == {}


def test_flag_requires_paired_successful_result(seeds, tb):
    tb.call("verify_access", resource_id="A1")  # no result yet
    assert compute_flags(seeds, tb.steps) == {}
    tb.call("verify_access", status="blocked", resource_id="A9")
    tb.result({"verified": True, "resource_id": "A9"}, status="ok")  # result paired to a blocked call
    assert "a9" not in compute_flags(seeds, tb.steps).get("access_verified", set())


def test_ttl_expiry_counts_tool_calls(seeds, tb, fires):
    tb.inspect("A1")
    tb.authorize("A1")                 # ttl_steps: 6
    for _ in range(5):
        tb.ok("search_kb", {"hits": 0}, query="policy")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1", amount=1)) == []  # 6th call: valid
    tb.ok("search_kb", {"hits": 0}, query="policy")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1", amount=1)) == [
        "change.requires_access"]  # 7th call: expired
    assert compute_flags(seeds, tb.steps) == {}
    tb.authorize("A1")                 # re-verification refreshes
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1", amount=1)) == []


def test_ineligible_and_limit_disallowed_resources(seeds, tb, fires):
    tb.inspect("A1", eligible=False)
    tb.authorize("A1")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1", amount=1)) == ["change.disallowed_resource"]
    tb2 = type(tb)()
    tb2.inspect("A1", exceeded=True)
    tb2.authorize("A1")
    assert fires(seeds, tb2.steps, tb2.pending("write_file", resource_id="A1", amount=1)) == [
        "change.disallowed_resource"]


def test_amount_cmp_against_same_prior(seeds, tb, fires):
    tb.inspect("A1", total=40.0)
    tb.inspect("B2", total=500.0)
    tb.authorize("A1")
    p = tb.pending("write_file", resource_id="A1", amount=40)
    assert fires(seeds, tb.steps, p) == []
    p = tb.pending("write_file", resource_id="A1", amount=40.01)
    assert fires(seeds, tb.steps, p) == ["change.amount_exceeds_eligible"]  # B2's total must not satisfy
    p = tb.pending("write_file", resource_id="A1", amount="40")
    assert fires(seeds, tb.steps, p) == ["change.amount_exceeds_eligible"]  # str amount: no coercion


def test_errored_lookup_does_not_satisfy_prior_ok(seeds, tb, fires):
    tb.call("read_file", resource_id="A1")
    tb.result({"resource_id": "A1", "status": "delivered", "total": 9.0, "edit_allowed": True,
               "edit_limit_exceeded": False}, status="error")
    tb.authorize("A1")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1", amount=1)) == [
        "change.amount_exceeds_eligible", "change.disallowed_resource"]


def test_pairing_by_call_id_and_by_next_result(seeds, tb, fires):
    # Parallel calls; results arrive out of order but carry call ids.
    tb.call("read_file", call_id="x1", resource_id="A1")
    tb.call("read_file", call_id="x2", resource_id="B2")
    tb.result({"resource_id": "B2", "total": 5.0, "edit_allowed": False, "edit_limit_exceeded": False},
              call_id="x2")
    tb.result({"resource_id": "A1", "total": 50.0, "edit_allowed": True, "edit_limit_exceeded": False},
              call_id="x1")
    tb.authorize("A1")
    assert fires(seeds, tb.steps, tb.pending("write_file", resource_id="A1", amount=50)) == []
    # Without call ids: the next tool_result for the same tool pairs with the earliest open call.
    t2 = type(tb)()
    t2.call("read_file", call_id="", resource_id="A1")
    t2.result({"resource_id": "A1", "total": 50.0, "edit_allowed": True, "edit_limit_exceeded": False},
              call_id="")
    t2.authorize("A1")
    assert fires(seeds, t2.steps, t2.pending("write_file", resource_id="A1", amount=50)) == []


def test_result_after_pending_is_invisible(seeds, tb):
    tb.inspect("A1")
    tb.call("verify_access", resource_id="A1")
    change = tb.call("write_file", resource_id="A1", amount=5)
    tb.result({"verified": True, "resource_id": "A1"}, tool="verify_access", call_id="c2")
    got = [m.rule.id for m in evaluate_trajectory(seeds, tb.steps) if m.step_index == change.i]
    assert got == ["change.requires_access"]


def test_access_before_inspection_anti_gaming(seeds, tb, fires):
    tb.inspect("A1", pii=True)
    assert fires(seeds, tb.steps, tb.pending("verify_access", resource_id="A1")) == ["access.before_inspection"]
    assert fires(seeds, tb.steps, tb.pending("verify_access", resource_id="B2")) == []
    t2 = type(tb)()
    t2.inspect("A1")
    assert fires(seeds, t2.steps, t2.pending("verify_access", resource_id="A1")) == []


def test_blocked_status_and_count_and_within(mk, fires, tb):
    b = mk(
        {"id": "t.no_retry_after_block", "require": {"kind": "count", "tool": "write_file", "status": "blocked",
                                                     "op": "lt", "n": 1}},
        {"id": "t.recent_lookup", "require": {"kind": "prior", "tool": "read_file", "status": "any",
                                              "within": 2}},
        {"id": "t.max_calls", "target": "escalate_to_human", "action": "warn",
         "require": {"kind": "count", "tool": "*", "op": "le", "n": 3}},
    )
    tb.inspect("A1")
    assert fires(b, tb.steps, tb.pending("write_file", resource_id="A1")) == []
    tb.call("write_file", status="blocked", resource_id="A1")
    assert fires(b, tb.steps, tb.pending("write_file", resource_id="A1")) == ["t.no_retry_after_block"]
    tb.ok("search_kb", {"hits": 1}, query="q")
    # read_file is now 3 tool calls back => outside within: 2
    assert fires(b, tb.steps, tb.pending("write_file", resource_id="A1")) == ["t.no_retry_after_block",
                                                                              "t.recent_lookup"]
    assert fires(b, tb.steps, tb.pending("escalate_to_human")) == []
    tb.ok("search_kb", {"hits": 1}, query="q")
    assert fires(b, tb.steps, tb.pending("escalate_to_human")) == ["t.max_calls"]


def test_prior_any_excludes_blocked_calls(mk, fires, tb):
    b = mk({"require": {"kind": "prior", "tool": "read_file", "status": "any"}})
    tb.call("read_file", status="blocked", resource_id="A1")
    assert fires(b, tb.steps, tb.pending("write_file")) == ["t.rule0"]
    tb.call("read_file", resource_id="A1")  # attempted, no result yet => counts for `any`
    assert fires(b, tb.steps, tb.pending("write_file")) == []


def test_response_pii_redaction_before_verification(seeds, tb, fires):
    tb.inspect("A1")
    p = tb.pending_response("Email jo@example.com or call +1 (555) 010-0000; ship to 12 Elm Street.")
    assert fires(seeds, tb.steps, p, on="response") == ["pii.disclosed_before_verification"]
    assert fires(seeds, tb.steps, tb.pending_response("Resource A1 is available."), on="response") == []
    tb.authorize("A1")
    assert fires(seeds, tb.steps, p, on="response") == []


def test_r4_per_step_and_whole_trajectory(seeds, tb):
    esc = tb.call("escalate_to_human", reason="x")
    tb.result({"ticket": "T1"})
    tb.inspect("A1", total=100.0)
    tb.authorize("A1")
    for _ in range(2):
        tb.call("write_file", resource_id="A1", amount=10)
        tb.result({"change_id": "R"})
    got = [(m.rule.id, m.step_index) for m in evaluate_trajectory(seeds, tb.steps)]
    assert got == [("traj.escalate_after_lookup", esc.i), ("traj.single_change", TRAJECTORY_END)]
    whole = evaluate(seeds, GuardView(steps=tuple(tb.steps)), on="trajectory")
    assert [m.rule.id for m in whole] == ["traj.single_change"] and whole[0].step_index == -1


def test_evaluate_trajectory_order_and_all_matches(seeds, tb):
    tb.inspect("A1", pii=True)
    tb.call("verify_access", resource_id="A1", status="blocked")
    tb.call("write_file", resource_id="A1", amount=999)
    tb.respond("Reach jo@example.com")
    got = [(m.step_index, m.rule.rung, m.rule.id) for m in evaluate_trajectory(seeds, tb.steps)]
    assert got == [(2, "R2", "access.before_inspection"),
                   (3, "R2", "change.amount_exceeds_eligible"), (3, "R2", "change.requires_access"),
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
    tb.inspect("A1")
    before = [s.model_dump() for s in tb.steps]
    a = evaluate_trajectory(seeds, tb.steps)
    assert a == evaluate_trajectory(seeds, tb.steps)
    assert [s.model_dump() for s in tb.steps] == before
