import asyncio
import threading

import pytest

from ci_lab.contracts import FailureRecord
from ci_lab.optim.scoring import (
    BudgetExhausted,
    CaseOutcome,
    DomainEvolveScorer,
    EvolveGuard,
    HeldOutAccessError,
    MetricBudget,
    candidate_hash,
    render_failure,
    resolve_scorer_result,
    stable_split,
    subsample,
)

PROMPT = "harness/prompts/system.md"


def test_metric_budget_hard_cap_threadsafe():
    b = MetricBudget(50)
    errors = []

    def work():
        for _ in range(20):
            try:
                b.charge()
            except BudgetExhausted:
                errors.append(1)

    ts = [threading.Thread(target=work) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert b.used == 50 and b.exhausted and len(errors) == 50 and b.refused == 50
    with pytest.raises(ValueError):
        MetricBudget(-1)


def test_stable_split_deterministic_and_disjoint():
    ids = [f"c{i}" for i in range(9)]
    tr, va = stable_split(ids)
    assert (tr, va) == stable_split(list(reversed(ids)))
    assert set(tr) | set(va) == set(ids) and not set(tr) & set(va)
    assert len(va) == 3
    assert stable_split(["x"]) == (["x"], ["x"])


def test_subsample_deterministic_order_preserving():
    ids = [f"c{i}" for i in range(10)]
    a = subsample(ids, 4, salt="s")
    assert len(a) == 4 and a == subsample(ids, 4, salt="s") and a == [c for c in ids if c in a]
    assert subsample(ids, None) == ids and subsample(ids + ["c1"], 99) == ids


def test_render_failure_typed_and_bounded():
    f = FailureRecord("c1", "s", "cat", ("R1", "R2"), {"b": 0.5, "a": 1.0}, excerpt="x" * 1000)
    text = render_failure(f, 0.25)
    assert text.startswith("score=0.250; suite=s; category=cat; violated_rules=R1,R2; rubric=a:1.00,b:0.50")
    assert len(text) < 600
    assert render_failure(None) == "passed"


def test_guard_rejects_heldout_and_charges_budget():
    calls = []

    def scorer(cand, cases):
        calls.append(list(cases))
        return [CaseOutcome(c, 0.0, tokens=3) for c in cases]

    inc = [FailureRecord("c1", "s", "cat", (), {}), FailureRecord("h1", "s", "cat", (), {})]
    g = EvolveGuard(scorer, ["c1", "c2"], MetricBudget(3), incumbent_failures=inc)
    assert set(g.incumbent) == {"c1"}  # held-out failure dropped
    with pytest.raises(HeldOutAccessError):
        g.score({"p": "x"}, ["c1", "h1"])
    assert calls == [] and g.budget.used == 0
    out = g.score({"p": "x"}, ["c1", "c2"])
    assert out[0].failure is inc[0] and out[1].failure is None
    assert g.tokens == 6 and g.calls == 2
    with pytest.raises(BudgetExhausted):
        g.score({"p": "y"}, ["c1", "c2"])
    assert len(calls) == 1
    with pytest.raises(ValueError):
        EvolveGuard(scorer, [], MetricBudget(1))


def test_resolve_async_from_worker_thread_uses_main_loop():
    seen = {}

    async def coro():
        seen["loop"] = asyncio.get_running_loop()
        return 7

    async def main():
        loop = asyncio.get_running_loop()
        val = await asyncio.to_thread(resolve_scorer_result, coro(), loop)
        return val, loop

    val, loop = asyncio.run(main())
    assert val == 7 and seen["loop"] is loop
    assert resolve_scorer_result(coro()) == 7  # no loop: runs its own
    assert resolve_scorer_result(5) == 5


def test_domain_scorer_evolve_only_and_cached(worktree, domain, tmp_path):
    s = DomainEvolveScorer(domain, worktree, tmp_path / "scratch", experiment_id="e", variant="a1")
    cand = {PROMPT: "Offer a change and tracking."}

    async def run():
        r1 = await s(cand, ["c1", "c2", "c3"])
        r2 = await s(cand, ["c4"])
        with pytest.raises(HeldOutAccessError):
            await s(cand, ["h1"])
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert [o.score for o in r1] == [1.0, 1.0, 0.0]
    assert r1[2].failure is not None and r1[2].failure.rule_ids == ("R1",)
    assert r2[0].score == 0.0
    assert domain.splits_called == ["evolve"]  # one cached evaluation, never held-out
    assert s.evaluations == 1 and s.tokens_spent == 60
    assert candidate_hash(cand)[:8] in domain.dirs[0].name
    assert not domain.dirs[0].exists()  # scratch removed
    assert (worktree / PROMPT).read_text() == \
        "Improve this repository's self-hosted harness.\n"  # worktree untouched
