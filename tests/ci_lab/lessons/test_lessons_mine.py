from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ci_lab.lessons.cluster import (
    MineConfig,
    append_confirmation,
    is_holdout,
    load_confirmations,
    mine,
    split_by_family,
)
from ci_lab.lessons.fingerprint import (
    canonical_calls,
    fingerprint,
    is_failure,
    is_good,
    tool_ngrams,
)
from ci_lab.lessons.harvest import StepBuilder, make_trajectory
from ci_lab.lessons.registry import Registry
from ci_lab.lessons.route import route_all
from ci_lab.rulespec import LessonEntry, Trajectory

Make = Callable[..., Trajectory]


def _retry_traj() -> Trajectory:
    b = StepBuilder()
    b.user()
    for _ in range(3):
        b.call("read_file", {"resource_id": "CASE-1"}, None)
        b.result(b.steps[-1].call_id, {"status": "delivered"}, "read_file")
    b.call("lookup_customer", {"email": "a@b.c"}, None)
    b.result(b.steps[-1].call_id, {"id": "C1"}, "lookup_customer")
    b.call("read_file", {"resource_id": "CASE-1"}, None)  # repeated lookup, collapsed
    b.result(b.steps[-1].call_id, {"status": "delivered"}, "read_file")
    b.call("write_file", {"resource_id": "CASE-1", "amount": 10}, None)
    b.result(b.steps[-1].call_id, {"error": "change_denied"}, "write_file")
    b.call("write_file", {"resource_id": "CASE-1", "amount": 10}, None)  # immediate retry, deduped
    b.result(b.steps[-1].call_id, {"error_code": "change_denied"}, "write_file")
    b.call("escalate", {}, None)
    b.result(b.steps[-1].call_id, {"ok": True}, "escalate")
    return make_trajectory(source="assert", split="evolve", case_id="x", steps=tuple(b.steps), pin="p@1",
                           passed=False, oracle_rules=["change.denied"])


def test_fingerprint_canonicalization() -> None:
    t = _retry_traj()
    assert [c.tool for c in canonical_calls(t)] == ["read_file", "lookup_customer", "write_file", "escalate"]
    assert tool_ngrams(t) == (("read_file", "lookup_customer", "write_file"),)
    fp = fingerprint(t)
    assert fp.pin == "p@1" and fp.error_class == "change_denied" and fp.oracle_rules == ("change.denied",)
    assert fingerprint(t.model_copy(update={"pin": "p@2"})).digest() != fp.digest()  # versioned by pin (N2)


def test_failure_and_good(make: Make) -> None:
    assert is_good(make("good", 1)) and not is_failure(make("good", 1))
    assert is_failure(make("unverified", 1)) and is_failure(make("ungrounded", 1))


def test_convergence_across_slices(make: Make) -> None:
    # the same failure recurring across all four time slices converges into one cluster ...
    spread = [make("unverified", i, slice=f"2025-01-0{1 + i % 4}") for i in range(40)]
    # ... while an equally frequent failure confined to a single slice does not
    burst = [make("overlimit", i, slice="2025-01-02") for i in range(40)]
    # ... nor one that only appears in the early half of the slices (unstable)
    early = [make("pii", i, slice=f"2025-01-0{1 + i % 2}") for i in range(40)]
    res = mine(spread + burst + early)
    assert len(res.clusters) == 1
    (c,) = res.clusters
    assert c.fingerprint.oracle_rules == ("harness.write_without_read",)
    assert len(c.slices) == 4 and c.status == "candidate" and not c.human_confirmed
    # dropped patterns are reported with a reason, never silently lost
    flat = [r for rs in res.dropped.values() for r in rs]
    assert len(res.dropped) == 2, res.dropped
    assert any("slices" in r for r in flat) and any("unstable" in r for r in flat)


def test_min_support(make: Make) -> None:
    few = [make("unverified", i, slice=f"2025-01-0{1 + i % 4}") for i in range(4)]
    res = mine(few, MineConfig(min_support=50))
    assert res.clusters == [] and res.dropped


def test_family_split_prevents_near_duplicate_leakage(make: Make) -> None:
    # every family has 4 near-duplicate paraphrases; they must all land on the same side
    trajs = [make("unverified", i, slice=f"2025-01-0{1 + (i + v) % 4}", variant=v)
             for i in range(30) for v in range(4)]
    assert len({t.family for t in trajs}) == 30
    mined, hold = split_by_family(trajs)
    assert mined and hold
    assert {t.family for t in mined}.isdisjoint({t.family for t in hold})
    res = mine(trajs)
    (c,) = res.clusters
    hold_ids = set(res.holdout_members[c.id])
    by_id = {t.id: t for t in trajs}
    hold_fams = {by_id[i].family for i in hold_ids}
    assert hold_fams and hold_fams.isdisjoint(c.families)
    assert all(not is_holdout(f) for f in c.families) and all(is_holdout(f) for f in hold_fams)
    assert set(c.members).isdisjoint(hold_ids)


def test_untrusted_only_cluster_is_backlog(make: Make) -> None:
    trajs = [make("unverified", i, slice=f"2025-01-0{1 + i % 4}", source="usage") for i in range(40)]
    assert all(not t.trusted for t in trajs)
    (c,) = mine(trajs).clusters
    assert c.status == "backlog"


def test_confirmation_is_human_and_persists(tmp_path: Path, make: Make) -> None:
    trajs = [make("unverified", i, slice=f"2025-01-0{1 + i % 4}") for i in range(40)]
    (c,) = mine(trajs).clusters
    append_confirmation(tmp_path, c.id, by="alice")
    (c2,) = mine(trajs, decisions=load_confirmations(tmp_path)).clusters
    assert c2.status == "confirmed" and c2.human_confirmed
    append_confirmation(tmp_path, c.id, by="bob", decision="rejected")
    (c3,) = mine(trajs, decisions=load_confirmations(tmp_path)).clusters
    assert c3.status == "rejected" and not c3.human_confirmed


def _routes(make: Make, kind: str, registry: Registry | None = None) -> tuple[str, dict, list[str], str]:
    fails = [make(kind, i, slice=f"2025-01-0{1 + i % 4}") for i in range(40)]
    good = [make("good", 1000 + i, slice=f"2025-01-0{1 + i % 4}") for i in range(40)]
    res = mine(fails + good)
    (r,) = route_all(res.clusters, res.members, fails + good, registry)
    return r.cluster.route, r.features, r.reasons, r.cluster.id


def test_router_ladder(make: Make) -> None:
    route, f, _, _ = _routes(make, "unverified")
    assert route == "R2"
    assert f == {"target_tool": "write_file", "subject_arg": "resource_id", "prior_tool": "read_file",
                 "prior_subject_field": "resource_id", "trusted": True}
    route, f, _, _ = _routes(make, "overlimit")
    assert route == "R1"
    assert f == {"target_tool": "write_file", "arg": "amount", "op": "le", "values": [80.0], "trusted": True}
    route, f, _, _ = _routes(make, "pii")
    assert route == "R3" and f["pattern_classes"] == ["email"]
    route, f, _, _ = _routes(make, "ungrounded")
    assert route == "R6" and f == {}


def test_router_forces_structural_after_repeated_prose_fixes(make: Make) -> None:
    _, _, _, cid = _routes(make, "ungrounded")
    reg = Registry([LessonEntry(lesson_id="L1", cluster_id=cid, prose_anchors=["AGENTS.md#grounding"]),
                    LessonEntry(lesson_id="L2", cluster_id=cid, prose_anchors=["AGENTS.md#grounding-2"])])
    route, _, reasons, _ = _routes(make, "ungrounded", reg)
    assert route == "R4" and any("forced structural" in r for r in reasons)
    one = Registry([LessonEntry(lesson_id="L1", cluster_id=cid, prose_anchors=["AGENTS.md#grounding"])])
    assert _routes(make, "ungrounded", one)[0] == "R6"


def test_injection_routes_to_prose_without_features(make: Make) -> None:
    fails = [make("unverified", i, slice=f"2025-01-0{1 + i % 4}") for i in range(40)]
    fails = [t.model_copy(update={"outcome": t.outcome.model_copy(update={"injection_suspect": True})})
             for t in fails]
    good = [make("good", 1000 + i, slice=f"2025-01-0{1 + i % 4}") for i in range(40)]
    res = mine(fails + good)
    (r,) = route_all(res.clusters, res.members, fails + good)
    assert r.cluster.route == "R6" and r.features == {}
