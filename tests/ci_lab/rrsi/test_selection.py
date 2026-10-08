import json
import random

import pytest

from ci_lab.contracts import CriticVerdict, EvaluatorPin
from ci_lab.rrsi.selection import SelectionInputs, select

DELTA = 0.125  # dyadic so every boundary below is exact in binary floating point


def run(mk, inc, arms, *, s_star=None, delta=DELTA, hp=None, history=(), cases=None, commit=None, guard=None):
    hp = hp or mk.hp()
    if s_star is None:
        from ci_lab.rrsi import stats
        idx = stats.index_scores(inc.scores)
        s_star = stats.task_score(idx, stats.universe([inc]))
    inp = SelectionInputs(round=1, incumbent=inc, arms=tuple(arms), s_star=s_star, delta=delta, hp=hp,
                          history=tuple(history), cases=None if cases is None else tuple(cases),
                          incumbent_commit=commit)
    return select(inp, guard=guard)


def tr(dec, name):
    return next(a for a in dec.arms if a.arm == name)


# ---------------------------------------------------------------- cost rule branch (dS > delta)


def test_cost_rule_accepts(mk):
    inc = mk.ev([0.5] * 4)
    d = run(mk, inc, [mk.arm("a", mk.ev([0.75] * 4))])
    a = tr(d, "a")
    assert d.decision == "ship" and d.winner == "a"
    assert a.branch == "cost" and a.delta_s == 0.25 and a.delta_c == 0.0
    assert a.rule["allowance"] == 0.5  # beta0 + beta1 * dS = 0.25 + 0.25
    assert a.ci["required"] and a.ci["lower"] == 0.25 and a.ci["passed"]
    assert d.s_star_before == 0.5 and d.s_star_after == 0.75 and d.score_next == 0.75
    assert d.incumbent["score"] == 0.5 and d.incumbent["cost"] == 100.0


def test_cost_rule_boundary_and_rejection(mk):
    inc = mk.ev([0.5] * 4)
    at = run(mk, inc, [mk.arm("a", mk.ev([0.75] * 4, tokens=150))])  # dC = 0.5 == allowance
    assert at.decision == "ship" and tr(at, "a").delta_c == 0.5
    over = run(mk, inc, [mk.arm("a", mk.ev([0.75] * 4, tokens=151))])  # dC = 0.51 > 0.5
    a = tr(over, "a")
    assert over.decision == "do_not_ship" and not a.admissible and not a.rule["passed"]
    assert any(r.startswith("cost_rule") for r in a.reasons)


def test_paper_table5_cost_example(mk):
    # Coding: beta1 = 44.5 ~ 25% token growth per extra pass of 178 trials; +7 passes -> 10 + 7*25 = 185%.
    hp = mk.hp(beta0=0.10, beta1=44.5, require_ci_lower=False)
    inc = mk.ev([0.0] * 178)
    arm = mk.ev([1.0] * 7 + [0.0] * 171, tokens=230)  # +130% tokens
    d = run(mk, inc, [mk.arm("a", arm)], hp=hp, delta=0.017)
    assert d.decision == "ship" and tr(d, "a").rule["allowance"] == pytest.approx(0.10 + 44.5 * 7 / 178)
    one = mk.ev([1.0] * 1 + [0.0] * 177, tokens=230)  # +1 pass is still inside the band (1/178 < 0.017)
    assert tr(run(mk, inc, [mk.arm("a", one)], hp=hp, delta=0.017), "a").branch == "weighted"


# ---------------------------------------------------------------- C16 CI lower bound


def test_ci_gate_blocks_noisy_gain_without_fallback(mk):
    inc = mk.ev([0.0] * 4)
    arm = mk.ev([0.0, 0.0, 0.0, 1.0], tokens=50)  # dS = 0.25 > delta, carried by one case; cheaper
    d = run(mk, inc, [mk.arm("a", arm)])
    a = tr(d, "a")
    # P(bootstrap misses the improved case) = 0.75^4 = 0.32 > 2.5% -> lower bound 0
    assert a.branch == "cost" and a.rule["allowance"] == 0.5 and a.ci["lower"] == 0.0
    assert d.decision == "do_not_ship" and any(r.startswith("ci_lower") for r in a.reasons)
    # no fall-back to the weighted rule even though tokens fell by 50% (which would pass it)
    assert "value" not in a.rule
    # the paper's rule alone accepts it
    assert run(mk, inc, [mk.arm("a", arm)], hp=mk.hp(require_ci_lower=False)).decision == "ship"


def test_ci_threshold_zero_vs_delta(mk):
    inc = mk.ev([0.0] * 4)
    arm = mk.ev([0.25, 0.25, 0.25, 0.0])  # dS = 0.1875; bootstrap lower bound = 0.0625 (0 < lb < delta)
    zero = run(mk, inc, [mk.arm("a", arm)], hp=mk.hp(n_bootstrap=4000))
    assert tr(zero, "a").ci["lower"] == 0.0625 and zero.decision == "ship"
    strict = run(mk, inc, [mk.arm("a", arm)], hp=mk.hp(n_bootstrap=4000, ci_lower_threshold="delta"))
    assert tr(strict, "a").ci["threshold"] == DELTA and strict.decision == "do_not_ship"


# ---------------------------------------------------------------- weighted rule branch (dS <= delta)


def test_weighted_rule_token_savings(mk):
    inc = mk.ev([0.5] * 4)
    d = run(mk, inc, [mk.arm("a", mk.ev([0.5] * 4, tokens=50))])
    a = tr(d, "a")
    assert a.branch == "weighted" and a.rule["value"] == 7.5  # 0 - 15 * (-0.5) + 0
    assert not a.ci["required"] and d.decision == "ship"


def test_weighted_rule_zero_is_rejected(mk):
    d = run(mk, mk.ev([0.5] * 4), [mk.arm("a", mk.ev([0.5] * 4))])
    assert tr(d, "a").rule["value"] == 0.0 and d.decision == "do_not_ship"


def test_weighted_rule_novelty(mk):
    inc = mk.ev([0.5] * 4)
    novel = run(mk, inc, [mk.arm("a", mk.ev([0.5] * 4), components=["skill", "prompt"])])
    assert tr(novel, "a").novelty == 1 and tr(novel, "a").rule["value"] == 0.5 and novel.decision == "ship"
    # skill accepted before -> no longer novel (tried-but-rejected would still be novel)
    hist = [mk.rec(0, "v1", ["skill"], 0.25, accepted=True)]
    old = run(mk, inc, [mk.arm("a", mk.ev([0.5] * 4), components=["skill"])], history=hist)
    assert tr(old, "a").novelty == 0 and old.decision == "do_not_ship"
    tried = [mk.rec(0, "v1", ["skill"], -0.25, accepted=False)]
    assert tr(run(mk, inc, [mk.arm("a", mk.ev([0.5] * 4), components=["skill"])], history=tried), "a").novelty == 1
    # prompt is not structural
    assert tr(run(mk, inc, [mk.arm("a", mk.ev([0.5] * 4), components=["prompt"])]), "a").novelty == 0


def test_band_edge_uses_weighted_rule(mk):
    inc = mk.ev([0.5] * 4)
    arm = mk.ev([0.5, 0.5, 0.5, 1.0])  # dS = 0.125 == delta -> not > delta
    d = run(mk, inc, [mk.arm("a", arm)])
    assert tr(d, "a").delta_s == DELTA and tr(d, "a").branch == "weighted" and d.decision == "do_not_ship"
    assert run(mk, inc, [mk.arm("a", arm)], hp=mk.hp(w_s=1.0)).decision == "ship"  # 1 * 0.125 > 0


def test_cheaper_small_regression_inside_band_ships(mk):
    inc = mk.ev([0.5] * 4)
    d = run(mk, inc, [mk.arm("a", mk.ev([0.5, 0.5, 0.5, 0.25], tokens=50))])  # dS = -0.0625, dC = -0.5
    assert d.decision == "ship" and tr(d, "a").floor["passed"]
    assert d.s_star_after == 0.5  # S* never moves down


# ---------------------------------------------------------------- floor


def test_floor(mk):
    inc = mk.ev([0.5] * 4)
    arm = mk.arm("a", mk.ev([0.75] * 4))
    low = run(mk, inc, [arm], s_star=1.0)  # floor 0.875 > 0.75
    assert low.decision == "do_not_ship" and not tr(low, "a").floor["passed"]
    assert low.s_star_after == 1.0
    edge = run(mk, inc, [arm], s_star=0.875)  # floor 0.75 == S'
    assert edge.decision == "ship" and tr(edge, "a").floor["threshold"] == 0.75
    assert edge.s_star_after == 0.875  # winner below S*: S* unchanged


# ---------------------------------------------------------------- guards


def test_critical_safety_guard_is_non_compensatory(mk):
    inc = mk.ev([0.5] * 4, crit={"c1": 1})
    worse = run(mk, inc, [mk.arm("a", mk.ev([1.0] * 4, tokens=10, crit={"c0": 1, "c2": 1}))])
    a = tr(worse, "a")
    assert worse.decision == "do_not_ship" and a.rule["passed"] and a.floor["passed"]
    assert a.guards["critical_safety"] == {"incumbent": 1, "arm": 2, "passed": False}
    same = run(mk, inc, [mk.arm("a", mk.ev([1.0] * 4, crit={"c3": 1}))])
    assert same.decision == "ship"


def test_other_guards(mk):
    inc = mk.ev([0.5] * 4)
    good = mk.ev([0.75] * 4)
    critic = mk.arm("a", good)
    critic.critic = CriticVerdict(passed=False, reasons=["leak"])
    assert run(mk, inc, [critic]).decision == "do_not_ship"
    stale = run(mk, inc, [mk.arm("a", good, base="old")], commit="inc")
    assert stale.decision == "do_not_ship" and not tr(stale, "a").guards["base_commit"]["passed"]
    assert run(mk, inc, [mk.arm("a", good)], commit="inc").decision == "ship"
    dom = run(mk, inc, [mk.arm("a", good)], guard=lambda arm, ev: ["too long"])
    assert dom.decision == "do_not_ship" and "guard: domain: too long" in tr(dom, "a").reasons


def test_zero_baseline_cost(mk):
    d = run(mk, mk.ev([0.5] * 4, tokens=0), [mk.arm("a", mk.ev([0.5] * 4, tokens=10))])
    a = tr(d, "a")
    assert d.decision == "do_not_ship" and "cost_baseline_zero" in a.reasons
    assert json.loads(json.dumps(d.to_dict(), allow_nan=False))["arms"][0]["delta_c"] is None


# ---------------------------------------------------------------- rerun (quality failures)


def test_missing_trial_threshold(mk):
    ok = mk.ev([0.5] * 9 + [None])  # 10% missing: not > 10%
    d = run(mk, ok, [mk.arm("a", mk.ev([0.5] * 10, tokens=50))], s_star=0.45)
    assert d.decision == "ship" and d.incumbent["missing_rate"] == 0.1
    bad = mk.ev([0.5] * 8 + [None, None])
    r = run(mk, bad, [mk.arm("a", mk.ev([0.5] * 10, tokens=50))], s_star=0.4)
    assert r.decision == "rerun" and r.winner is None and r.s_star_after == 0.4
    assert r.reasons[0].startswith("baseline_invalid") and r.arms[0].reasons == ("round_rerun",)


def test_missing_counts_as_zero_and_fixed_universe(mk):
    inc = mk.ev({"a": [0.5], "b": [0.5], "c": [0.5], "d": [0.5]})
    arm = mk.ev({"a": [0.75], "b": [0.75], "c": [0.75], "d": [None]})  # S' = 0.5625
    d = run(mk, inc, [mk.arm("x", arm)])
    assert tr(d, "x").score == 0.5625 and tr(d, "x").missing_rate == 0.25
    # fixed universe: an absent trial of the incumbent is missing too -> 1/5 = 20% -> rerun
    r = run(mk, inc, [mk.arm("x", arm)], cases=["a", "b", "c", "d", "e"])
    assert r.decision == "rerun" and r.n_keys == 5


def test_pin_and_split_mismatch_rerun(mk):
    inc = mk.ev([0.5] * 4)
    other = EvaluatorPin(evaluator_tree="other", judge_model="judge", judge_provider="prov")
    d = run(mk, inc, [mk.arm("a", mk.ev([0.75] * 4, pin=other))])
    assert d.decision == "rerun" and "evaluator_pin_mismatch: arm a" in d.reasons
    s = run(mk, inc, [mk.arm("a", mk.ev([0.75] * 4, split="heldout"))])
    assert s.decision == "rerun" and s.reasons[0].startswith("split_mismatch")
    e = run(mk, mk.ev([]), [], s_star=0.0)
    assert e.decision == "rerun" and e.reasons == ("no_trials",)


# ---------------------------------------------------------------- argmax / ties / none admissible


def test_argmax_and_ties(mk):
    inc = mk.ev([0.5] * 4)
    best = run(mk, inc, [mk.arm("a", mk.ev([0.75] * 4)), mk.arm("b", mk.ev([1.0] * 4))])
    assert best.winner == "b" and best.s_star_after == 1.0
    cheaper = run(mk, inc, [mk.arm("a", mk.ev([0.75] * 4)), mk.arm("b", mk.ev([0.75] * 4, tokens=90))])
    assert cheaper.winner == "b"
    full_tie = run(mk, inc, [mk.arm("b", mk.ev([0.75] * 4)), mk.arm("a", mk.ev([0.75] * 4))])
    assert full_tie.winner == "a" and [t.arm for t in full_tie.arms] == ["a", "b"]
    # the tie-break never promotes an inadmissible arm
    mixed = run(mk, inc, [mk.arm("a", mk.ev([1.0] * 4, crit={"c0": 1})), mk.arm("b", mk.ev([0.75] * 4))])
    assert mixed.winner == "b"


def test_all_inadmissible(mk):
    inc = mk.ev([0.5] * 4)
    arms = [mk.arm("a", mk.ev([0.25] * 4)),                    # below floor, weighted rule fails
            mk.arm("b", mk.ev([0.75] * 4, tokens=1000)),       # cost rule fails
            mk.arm("c", mk.ev([1.0] * 4, crit={"c0": 1})),      # safety guard
            mk.arm("d", None, status="rejected")]              # critic dropped it
    d = run(mk, inc, arms, s_star=0.5)
    assert d.decision == "do_not_ship" and d.winner is None and d.winner_trace is None
    assert not any(t.admissible for t in d.arms)
    assert d.score_next == 0.5 and d.s_star_after == 0.5
    assert not tr(d, "d").evaluated and tr(d, "d").reasons == ("not_evaluated: status=rejected",)
    assert json.dumps(d.to_dict(), allow_nan=False)
    with pytest.raises(ValueError):
        run(mk, inc, [mk.arm("a", None), mk.arm("a", None)])


def test_s_star_ratchets_on_remeasured_incumbent(mk):
    d = run(mk, mk.ev([0.75] * 4), [], s_star=0.5)
    assert d.decision == "do_not_ship" and d.s_star_after == 0.75


# ---------------------------------------------------------------- properties


def test_selection_properties(mk):
    rng = random.Random(11)
    levels = [0.0, 0.25, 0.5, 0.75, 1.0, None]
    for it in range(120):
        n = rng.randint(3, 8)
        inc = mk.ev([rng.choice(levels[:-1]) for _ in range(n)], tokens=[rng.choice([50, 100]) for _ in range(n)])
        arms = []
        for j in range(rng.randint(0, 4)):
            e = mk.ev([rng.choice(levels) for _ in range(n)], tokens=[rng.choice([40, 100, 160]) for _ in range(n)],
                      crit={"c0": rng.choice([0, 0, 1])})
            arms.append(mk.arm(f"v{j}", e, components=rng.sample(["prompt", "skill", "memory"], 1)))
        s_star = rng.choice([0.0, 0.5, 0.75])
        hp = mk.hp(n_bootstrap=200, seed=it, w_s=rng.choice([0.0, 1.0]))
        d = run(mk, inc, arms, s_star=s_star, hp=hp)
        json.dumps(d.to_dict(), allow_nan=False)
        assert d.s_star_after >= d.s_star_before
        if d.decision == "rerun":
            assert d.winner is None and d.incumbent["missing_rate"] > hp.missing_invalid_frac
            continue
        adm = [t for t in d.arms if t.admissible]
        assert (d.decision == "ship") == bool(adm)
        for t in adm:
            assert t.floor["passed"] and t.guards["critical_safety"]["passed"] and t.rule["passed"]
            assert t.score >= s_star - DELTA
            if t.branch == "cost":
                assert t.delta_s > DELTA and t.ci["lower"] > 0
            else:
                assert t.delta_s <= DELTA
        if adm:
            assert d.winner_trace.score == max(t.score for t in adm)
        # same inputs -> identical decision (seeded bootstrap)
        assert run(mk, inc, arms, s_star=s_star, hp=hp).to_dict() == d.to_dict()
