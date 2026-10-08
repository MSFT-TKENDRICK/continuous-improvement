from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path

import pytest

from ci_lab.lessons.cluster import mine
from ci_lab.lessons.replay import (
    ReplayConfig,
    leak_screen,
    paraphrases,
    regex_literals,
    safety_check,
    validate,
    validate_ok,
)
from ci_lab.rulespec import (
    ArgPred,
    LessonCluster,
    NotPred,
    PriorPred,
    RuleSpec,
    TextPred,
    Trajectory,
)

Make = Callable[..., Trajectory]
SL = ("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04")


def _corpus(make: Make, n_fail: int = 20, n_good: int = 220) -> list[Trajectory]:
    return ([make("unverified", i, slice=SL[i % 4]) for i in range(n_fail)]
            + [make("good", 1000 + i, slice=SL[i % 4]) for i in range(n_good)])


def _cluster(trajs: list[Trajectory]) -> LessonCluster:
    (c,) = mine(trajs).clusters
    return c


def narrow_rule(rid: str = "refund.needs_status") -> RuleSpec:
    return RuleSpec(id=rid, version=1, rung="R2", on="tool_call", target="issue_refund",
                    require=PriorPred(kind="prior", tool="get_order_status",
                                      same=[("current.args.order_id", "prior.args.order_id")]),
                    action="block", template="precondition.prior_call",
                    slots={"tool": "issue_refund", "prior_tool": "get_order_status", "subject": "order_id"})


def broad_rule() -> RuleSpec:  # "refunds over 10 are never allowed" — blocks almost every good refund
    return RuleSpec(id="refund.tiny_only", version=1, rung="R1", on="tool_call", target="issue_refund",
                    require=ArgPred(kind="arg", path="current.args.amount", op="le", value=10),
                    action="block", template="arg.out_of_range", slots={"tool": "issue_refund", "arg": "amount"})


def text_rule(pattern: str, rid: str = "reply.no_promise") -> RuleSpec:
    return RuleSpec(id=rid, version=1, rung="R3", on="response", target="*",
                    require=NotPred(kind="not", of=TextPred(kind="text", matches=pattern)),
                    action="warn", template="pii.redacted", slots={})


def test_narrow_rule_passes_to_closed_loop_never_accepts(make: Make, engine: types.SimpleNamespace,
                                                          tmp_path: Path) -> None:
    trajs = _corpus(make)
    rep = validate(narrow_rule(), trajs, cluster=_cluster(trajs), engine=engine, work_dir=tmp_path)
    assert rep.verdict == "pass_to_closed_loop" and rep.ok, rep.reasons
    assert rep.verdict != "accept"  # B4: the strongest verdict is "hand to the closed loop"
    assert rep.holdout_n > 0 and rep.recall == 1.0 and rep.fp == 0 and rep.negatives == 220
    assert rep.fp_ucb is not None and rep.fp_ucb <= 0.02
    assert rep.per_rule[0].block_rate <= 0.10
    assert validate_ok(narrow_rule(), trajs, cluster=_cluster(trajs), engine=engine) == (True, [])


def test_over_broad_rule_rejected(make: Make, engine: types.SimpleNamespace) -> None:
    trajs = _corpus(make)
    rep = validate(broad_rule(), trajs, cluster=_cluster(trajs), engine=engine)
    assert rep.verdict == "reject" and not rep.ok
    joined = " ".join(rep.reasons)
    assert "fp_ucb" in joined and "block rate" in joined and "aggregate block rate" in joined
    assert rep.fp == 220


def test_rejects_without_holdout_or_enough_negatives(make: Make, engine: types.SimpleNamespace) -> None:
    trajs = _corpus(make, n_good=10)
    ok, reasons = validate_ok(narrow_rule(), trajs, engine=engine)
    assert not ok
    assert any("holdout recall unmeasurable" in r for r in reasons)
    assert any("negatives 10 < 30" in r for r in reasons)


def test_low_recall_rejected(make: Make, engine: types.SimpleNamespace) -> None:
    trajs = _corpus(make)
    # a rule that never fires on the cluster's held-out families
    other = narrow_rule().model_copy(update={"target": "cancel_order"})
    rep = validate(other, trajs, cluster=_cluster(trajs), engine=engine, config=ReplayConfig(min_negatives=0))
    assert not rep.ok and rep.recall == 0.0 and any("holdout recall" in r for r in rep.reasons)


def test_sealed_and_usage_trajectories_never_replayed(make: Make, engine: types.SimpleNamespace) -> None:
    usage = [make("good", i, slice=SL[i % 4], source="usage") for i in range(50)]
    rep = validate(narrow_rule(), usage, engine=engine, config=ReplayConfig(require_holdout=False))
    assert rep.n_evolve == 0 and not rep.ok


def test_load_failure_and_unsafe_regex_rejected(make: Make, engine: types.SimpleNamespace) -> None:
    trajs = _corpus(make)
    bad_tpl = narrow_rule().model_copy(update={"template": "unknown_template"})
    rep = validate(bad_tpl, trajs, cluster=_cluster(trajs), engine=engine)
    assert not rep.ok and any("rule load failed" in r and "unknown template" in r for r in rep.reasons)
    lookbehind = text_rule(r"(?<=will )arrive")
    assert safety_check([lookbehind]) == ["reply.no_promise: regex does not compile under RE2"]
    rep = validate(lookbehind, trajs, engine=engine, config=ReplayConfig(require_holdout=False))
    assert not rep.ok and any("RE2" in r for r in rep.reasons)
    assert validate([{"id": "X"}], trajs, engine=engine).reasons[0].startswith("invalid rule spec")


def test_engine_unavailable_rejects(make: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    import ci_lab.lessons.replay as rp

    def boom(engine: object = None) -> object:
        raise ImportError("no engine")

    monkeypatch.setattr(rp, "_engine", boom)
    rep = validate(narrow_rule(), _corpus(make))
    assert not rep.ok and any("unavailable" in r for r in rep.reasons)


def test_leak_screen() -> None:
    texts = ["Customer: where is my parcel? It will arrive tomorrow for sure, right?",
             "Please refund order NW-55555 because the item is broken",
             "My Order Number Is Missing From The Portal Again"]
    ngram = text_rule(r"arrive tomorrow for sure")
    memo = RuleSpec(id="refund.memo_id", version=1, rung="R1", on="tool_call", target="issue_refund",
                    require=ArgPred(kind="arg", path="current.args.order_id", op="nin", value=["NW-55555"]),
                    action="block", template="arg.not_allowed", slots={"tool": "issue_refund", "arg": "order_id"})
    fragile = text_rule(r"Order Number Is Missing", rid="reply.fragile")
    robust = text_rule(r"(?i)refund", rid="reply.robust")
    found = {(f.rule_id, f.kind) for f in leak_screen([ngram, memo, fragile, robust], texts)}
    assert ("reply.no_promise", "ngram") in found
    assert ("refund.memo_id", "id") in found
    assert ("reply.fragile", "paraphrase_fragile") in found
    assert not any(r == "reply.robust" for r, _ in found)
    # findings carry digests only, never the literal
    for f in leak_screen([ngram, memo], texts):
        assert "arrive" not in f.model_dump_json() and "NW-55555" not in f.model_dump_json()
    assert leak_screen([ngram], []) == []


def test_leak_rejects_in_validate(make: Make, engine: types.SimpleNamespace) -> None:
    trajs = _corpus(make)
    rep = validate(text_rule(r"arrive tomorrow for sure"), trajs, engine=engine,
                   dataset_texts=["It will arrive tomorrow for sure."], config=ReplayConfig(require_holdout=False))
    assert not rep.ok and rep.leaks and any("leak (ngram)" in r for r in rep.reasons)
    assert "arrive tomorrow" not in rep.model_dump_json()


def test_paraphrases_deterministic() -> None:
    p1 = paraphrases("Please refund my order.")
    assert p1 == paraphrases("Please refund my order.") and len(p1) >= 4
    assert "kindly reimbursement my purchase" in p1
    assert regex_literals(r"(?i)\border\s+number\b|refund-\d+") == ["order", "number", "refund-"]


def test_real_rules_engine(make: Make, tmp_path: Path) -> None:
    """The pinned ci_lab.rules engine (M14) with a real template id."""
    pytest.importorskip("ci_lab.rules")
    trajs = _corpus(make)
    c = _cluster(trajs)
    rep = validate(narrow_rule(), trajs, cluster=c, work_dir=tmp_path)
    assert rep.ok, rep.reasons
    assert rep.recall == 1.0 and rep.fp == 0
    rep = validate(broad_rule(), trajs, cluster=c)
    assert not rep.ok and rep.fp == 220
    rep = validate(narrow_rule().model_copy(update={"template": "no.such_template"}), trajs, cluster=c)
    assert not rep.ok and any("rule load failed" in r for r in rep.reasons)
