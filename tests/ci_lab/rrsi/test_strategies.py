import json
import random

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.contracts import ATTR_EXPERIMENT, ATTR_PHASE, ATTR_ROUND, ATTR_STRATEGY, SPAN_STEP
from ci_lab.rrsi import schedule as S
from ci_lab.rrsi.attribution import attribute, component_stats
from ci_lab.rrsi.params import DEFAULT_STRATEGIES, Hyperparams, profile
from ci_lab.rrsi.readjudicate import readjudicate, round_record
from ci_lab.rrsi.selection import SelectionInputs, select
from ci_lab.rrsi.strategies import allocate_strategies, starved, strategy_stats


def hp(**kw):
    return profile("paper", **kw)


# ---------------------------------------------------------------- stats (C23: per strategy, per arm)


def test_strategy_stats_hand_computed(mk):
    recs = [mk.rec(0, "v1", ["prompt", "skill"], 0.1, accepted=True, strategy="gepa"),
            mk.rec(0, "v2", ["prompt"], -0.1, strategy="agent"),
            mk.rec(1, "v1", ["skill"], 0.0, strategy="gepa"),
            mk.rec(2, "v2", ["memory"], None, strategy="skillopt"),   # not measured -> ignored
            mk.rec(2, "v3", ["memory"], 0.2, accepted=True, strategy="unknown")]  # outside vocabulary
    st = strategy_stats(recs, prior=(1.0, 2.0))
    assert set(st) == set(DEFAULT_STRATEGIES)
    g = st["gepa"]
    # one arm = one trial, however many edits/components it carries
    assert (g.tried, g.accepted, g.last_round, g.alpha, g.beta) == (2, 1, 1, 2.0, 3.0)
    assert g.success_rate == 0.5 and g.posterior_mean == 0.4
    assert (st["agent"].tried, st["agent"].alpha, st["agent"].beta) == (1, 1.0, 3.0)
    assert st["skillopt"].tried == 0 and st["skillopt"].last_round is None
    assert json.dumps({k: v.to_dict() for k, v in st.items()})


def test_component_stats_ignore_strategy(mk):
    a = [mk.rec(0, "v1", ["skill"], 0.1, accepted=True, strategy="gepa")]
    b = [mk.rec(0, "v1", ["skill"], 0.1, accepted=True, strategy="skillopt")]
    assert component_stats(a) == component_stats(b)
    assert strategy_stats(a)["gepa"].accepted == 1 and strategy_stats(b)["gepa"].tried == 0


# ---------------------------------------------------------------- floor


def test_floor_rotates_untried_strategies(mk):
    a0 = allocate_strategies(0, 2, [], hp())
    assert a0.strategies == ("agent", "gepa") and a0.reasons == ("floor", "floor")
    hist = [mk.rec(0, "v1", ["prompt"], 0.1, accepted=True, strategy="agent"),
            mk.rec(0, "v2", ["prompt"], -0.1, strategy="gepa")]
    a1 = allocate_strategies(1, 2, hist, hp())
    assert a1.strategies[0] == "skillopt" and a1.reasons == ("floor", "thompson")
    assert a1.samples[0] == {} and set(a1.samples[1]) == set(DEFAULT_STRATEGIES)


def test_floor_after_k_rounds(mk):
    hist = [mk.rec(0, "v1", ["prompt"], 0.0, strategy="gepa"),
            mk.rec(1, "v1", ["prompt"], 0.0, strategy="skillopt"),
            mk.rec(2, "v1", ["prompt"], 0.0, strategy="agent")]
    st = strategy_stats(hist)
    assert starved(st, 2, 3) == ()
    assert starved(st, 3, 3) == ("gepa",)                    # 3 - 0 >= K
    assert starved(st, 4, 3) == ("gepa", "skillopt")         # oldest first
    a = allocate_strategies(3, 2, hist, hp(strategy_floor_every=3))
    assert a.strategies[0] == "gepa" and a.reasons[0] == "floor"
    # more starved strategies than arms: oldest wins, the rest wait one round
    assert allocate_strategies(5, 1, hist, hp(n_arms=1, strategy_floor_every=3)).strategies == ("gepa",)


# ---------------------------------------------------------------- Thompson


def recent(mk, t, wins):
    """Every strategy ran an arm in round t-1; ``wins[s]`` = (accepted, tried) over rounds < t."""
    out = []
    for s, (acc, n) in wins.items():
        for i in range(n):
            out.append(mk.rec(t - 1 - (i % 2), f"{s}{i}", ["prompt"], 0.0, accepted=i < acc, strategy=s))
    return out


def test_thompson_is_seeded_and_argmax_of_draws(mk):
    hist = recent(mk, 10, {"agent": (3, 6), "gepa": (2, 6), "skillopt": (1, 6)})
    a = allocate_strategies(10, 2, hist, hp(seed=4))
    assert a.reasons == ("thompson", "thompson")
    assert allocate_strategies(10, 2, hist, hp(seed=4)) == a
    for s, draw in zip(a.strategies, a.samples):
        assert s == max(draw, key=lambda k: (draw[k], -DEFAULT_STRATEGIES.index(k)))
    picks = {allocate_strategies(10, 2, hist, hp(seed=i)).strategies for i in range(30)}
    assert len(picks) > 1  # different seeds explore


def test_thompson_prefers_successful_strategy(mk):
    hist = recent(mk, 10, {"agent": (0, 20), "gepa": (20, 20), "skillopt": (0, 20)})
    for seed in range(100):
        assert allocate_strategies(10, 2, hist, hp(seed=seed)).strategies == ("gepa", "gepa")


def test_strategy_cap(mk):
    hist = recent(mk, 10, {"agent": (0, 20), "gepa": (20, 20), "skillopt": (0, 20)})
    a = allocate_strategies(10, 3, hist, hp(n_arms=3, strategy_cap=1, seed=1))
    assert sorted(a.strategies) == sorted(DEFAULT_STRATEGIES) and a.strategies[0] == "gepa"


def test_strategy_hyperparam_validation():
    with pytest.raises(ValueError, match="subset"):
        hp(strategies=("agent", "dspy"))
    with pytest.raises(ValueError, match="subset"):
        hp(strategies=())
    with pytest.raises(ValueError, match="floor infeasible"):
        hp(n_arms=1, strategy_floor_every=2)
    with pytest.raises(ValueError, match="strategy_cap"):
        hp(n_arms=4, strategy_cap=1)
    with pytest.raises(ValueError, match="prior"):
        hp(strategy_prior=(0.0, 1.0))
    assert hp().strategies == DEFAULT_STRATEGIES and "guard" not in DEFAULT_STRATEGIES
    assert hp(strategies=(*DEFAULT_STRATEGIES, "guard")).strategies[-1] == "guard"
    h = hp(strategies=["gepa", "agent"], strategy_prior=[2, 3], strategy_cap=2)
    assert h.strategies == ("gepa", "agent") and h.strategy_prior == (2.0, 3.0)
    assert Hyperparams.from_dict(json.loads(json.dumps(h.to_dict()))) == h
    one = profile("smoke", strategies=("agent",))
    assert {d.strategy for d in S.plan_round(0, one, [], [0.5], 0.1).directives} == {"agent"}


def test_floor_guarantee_property(mk):
    rng = random.Random(3)
    for it in range(40):
        n_arms, k = rng.randint(1, 3), rng.randint(1, 4)
        if len(DEFAULT_STRATEGIES) > n_arms * k:
            continue
        h = hp(n_arms=n_arms, strategy_floor_every=k, seed=it)
        hist, seen = [], []
        for t in range(15):
            a = allocate_strategies(t, n_arms, hist, h)
            assert len(a.strategies) == n_arms and set(a.strategies) <= set(DEFAULT_STRATEGIES)
            assert a == allocate_strategies(t, n_arms, hist + [mk.rec(t + 1, "x", ["prompt"], 0.0)], h)
            win = rng.randrange(n_arms)
            hist += [mk.rec(t, f"v{i}", ["prompt"], rng.choice([-0.1, 0.0, 0.1]), accepted=i == win, strategy=s)
                     for i, s in enumerate(a.strategies)]
            seen.append(set(a.strategies))
        for t in range(len(seen) - k + 1):
            assert set().union(*seen[t:t + k]) == set(DEFAULT_STRATEGIES), (it, t)


# ---------------------------------------------------------------- directives + span


def test_plan_round_emits_contract_directives_with_strategy(mk):
    sched = S.plan_round(0, profile("smoke"), [], [0.5], 0.1)
    ad = sched.arm_directives
    assert [(d.arm, d.strategy) for d in ad] == [("v1", "agent"), ("v2", "gepa")]
    assert all(d.edit_budget == 2 and not d.explore and len(d.component_focus) == 1 for d in ad)
    doc = sched.to_dict()
    assert doc["directives"][1]["strategy"] == "gepa" and doc["allocation"]["reasons"] == ["floor", "floor"]
    assert json.dumps(doc, allow_nan=False)


def test_plan_round_span(monkeypatch):
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    S.directives(0, profile("smoke"), [], [0.5], 0.1, experiment_id="exp-1")
    (sp,) = exp.get_finished_spans()
    assert sp.name == SPAN_STEP
    assert sp.attributes[ATTR_PHASE] == "plan" and sp.attributes[ATTR_ROUND] == 0
    assert sp.attributes[ATTR_EXPERIMENT] == "exp-1" and sp.attributes[ATTR_STRATEGY] == "agent,gepa"
    assert sp.attributes["rrsi.budget"] == 2 and sp.attributes["rrsi.stalled"] is False
    S.plan_round(1, profile("smoke"), [], [0.5, 0.5], 0.1)  # experiment id optional -> attribute omitted
    assert ATTR_EXPERIMENT not in exp.get_finished_spans()[-1].attributes


# ---------------------------------------------------------------- selection stays strategy-blind


def test_selection_is_strategy_blind_and_attribution_records_strategy(mk, tmp_path):
    inc = mk.ev([0.5] * 4)
    evs = [mk.ev([0.75] * 4), mk.ev([0.5] * 4, tokens=50), mk.ev([1.0] * 4, crit={"c0": 1})]
    decisions = []
    for labels in (("agent", "gepa", "skillopt"), ("skillopt", "agent", "gepa"), ("gepa", "gepa", "gepa")):
        arms = tuple(mk.arm(f"v{i}", e, strategy=s) for i, (e, s) in enumerate(zip(evs, labels)))
        inp = SelectionInputs(round=0, incumbent=inc, arms=arms, s_star=0.5, delta=0.125, hp=mk.hp())
        d = select(inp)
        decisions.append(d.to_dict())
        recs = attribute(d, arms)
        assert [r.strategy for r in recs] == list(labels)
        rr = readjudicate(json.loads(json.dumps(round_record(inp, d))))
        assert not rr.changed and [r.strategy for r in rr.records] == list(labels)
    assert decisions[0] == decisions[1] == decisions[2]
    assert decisions[0]["winner"] == "v0"
