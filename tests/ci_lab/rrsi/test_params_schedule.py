import math
import random

import pytest

from ci_lab.contracts import COMPONENTS, ArmDirective
from ci_lab.rrsi import schedule as S
from ci_lab.rrsi.history import tried_components
from ci_lab.rrsi.params import PROFILES, TABLE5, Hyperparams, paper_reference, profile

# ---------------------------------------------------------------- params


def test_paper_profile_is_table5_coding_without_delta():
    p = profile("paper")
    assert (p.T, p.k, p.b_min, p.b_max, p.w, p.m_draft, p.n_prune) == (20, 2, 1, 4, 3, 1, 4)
    assert (p.beta0, p.beta1, p.w_s, p.w_c, p.w_n) == (0.10, 44.5, 0.0, 15.0, 0.5)
    assert p.delta is None  # delta must be calibrated, never copied from the paper


def test_table5_reference_columns():
    assert paper_reference("coding").delta == 0.017
    assert paper_reference("workspace").beta1 == 35.4 and paper_reference("workspace").b_max == 3
    eng = paper_reference("engineering")
    assert (eng.T, eng.k, eng.n_prune, eng.beta0, eng.delta) == (40, 4, 5, 0.15, 0.020)
    assert set(TABLE5) == {"coding", "workspace", "engineering"}


def test_profiles_and_roundtrip():
    assert set(PROFILES) == {"smoke", "local", "paper"}
    assert (profile("smoke").M, profile("smoke").k, profile("smoke").n_arms, profile("smoke").T) == (12, 1, 2, 3)
    assert (profile("local").M, profile("local").k, profile("local").T) == (40, 2, 8)
    for p in PROFILES.values():
        assert Hyperparams.from_dict(p.to_dict()) == p
    assert profile("paper", seed=7).seed == 7
    with pytest.raises(ValueError):
        profile("nope")


@pytest.mark.parametrize("bad", [dict(b_min=3, b_max=2), dict(b_min=0), dict(T=0), dict(ci_level=1.0),
                                 dict(ci_lower_threshold="half"), dict(delta=-0.1), dict(w=0)])
def test_hyperparam_validation(bad):
    with pytest.raises(ValueError):
        profile("paper", **bad)


# ---------------------------------------------------------------- budget


def test_cosine_budget_endpoints_and_hand_values():
    assert S.edit_budget(0, 20, 1, 4) == 4      # t=0 -> b_max
    assert S.edit_budget(20, 20, 1, 4) == 1     # t=T -> b_min (cos(pi) = -1 exactly)
    assert S.edit_budget(10, 20, 1, 4) == 3     # 1 + 3*0.5*(1+cos(pi/2)) = 2.5 -> 3
    assert S.edit_budget(5, 20, 1, 4) == 4      # 1 + 1.5*(1+0.7071) = 3.56 -> 4
    assert S.edit_budget(15, 20, 1, 4) == 2     # 1 + 1.5*(1-0.7071) = 1.44 -> 2
    assert S.edit_budget(19, 20, 1, 4) == 2     # 1 + 1.5*(1-0.9877) = 1.018 -> 2
    assert S.edit_budget(-3, 20, 1, 4) == 4 and S.edit_budget(99, 20, 1, 4) == 1  # clamped
    assert S.edit_budget(7, 20, 2, 2) == 2


def test_cosine_budget_properties():
    rng = random.Random(1)
    for _ in range(300):
        T = rng.randint(1, 50)
        b_min = rng.randint(1, 5)
        b_max = rng.randint(b_min, 9)
        seq = [S.edit_budget(t, T, b_min, b_max) for t in range(T + 1)]
        assert seq[0] == b_max and seq[-1] == b_min
        assert all(b_min <= b <= b_max for b in seq)
        assert all(a >= b for a, b in zip(seq, seq[1:]))  # non-increasing


# ---------------------------------------------------------------- stall


def test_stall_flag():
    # S_3 - S_0 = 0.03 > 0.017: not stalled
    assert not S.stall_flag([0.5, 0.5, 0.52, 0.53], 3, 3, 0.017)
    # S_3 - S_0 = 0.01 <= 0.017: stalled
    assert S.stall_flag([0.5, 0.51, 0.51, 0.51], 3, 3, 0.017)
    # boundary: difference == delta counts as a stall (<=)
    assert S.stall_flag([0.5, 0.0, 0.75], 2, 2, 0.25)
    # regression is a stall
    assert S.stall_flag([0.75, 0.5], 1, 1, 0.0)
    # not enough history yet
    assert not S.stall_flag([0.5, 0.5, 0.5], 2, 3, 0.017)
    assert not S.stall_flag([0.5], 3, 1, 0.017)  # t beyond trajectory


# ---------------------------------------------------------------- untried / prune / exploration


def test_untried_and_prune(mk):
    hist = [
        mk.rec(0, "v1", ["prompt"], 0.05, accepted=True),
        mk.rec(0, "v2", ["skill"], -0.02),
        mk.rec(1, "v1", ["config"], 0.0),
        mk.rec(1, "v2", ["memory"], None),  # never measured -> not "tried"
    ]
    assert S.untried(hist) == ("client_tool", "memory", "context_mgmt")
    # t=2, n_prune=4: g(prompt)=0.05, g(skill)=-0.02, g(config)=0.0 -> B = {skill, config}
    assert S.prune_set(hist, 2, 4) == ("skill", "config")
    # t=6, n_prune=4: round-0 edits fall out of the window -> g = -inf -> pruned too
    assert S.prune_set(hist, 6, 4) == ("prompt", "skill", "config")
    assert S.component_yield(hist, 6, 4)["prompt"] == -math.inf


def test_exploration_slots():
    assert S.exploration_slots(True, ("skill",), 1, 2) == 1
    assert S.exploration_slots(True, ("skill", "memory"), 3, 2) == 2  # capped by arms
    assert S.exploration_slots(True, (), 1, 2) == 0                   # nothing untried
    assert S.exploration_slots(False, ("skill",), 1, 2) == 0


def test_directives_when_stalled(mk):
    hp = profile("paper", n_arms=3)
    hist = [mk.rec(0, "v1", ["prompt"], 0.05, accepted=True), mk.rec(1, "v1", ["config"], -0.01)]
    traj = [0.55, 0.55, 0.55, 0.56]  # S_3 - S_0 = 0.01 <= 0.017 -> stalled
    sched = S.plan_round(3, hp, hist, traj, delta=0.017)
    assert sched.stalled and sched.budget == S.edit_budget(3, 20, 1, 4)
    assert sched.prune == ("config",) and sched.exploration_slots == 1
    d = sched.directives
    assert [x.arm for x in d] == ["v1", "v2", "v3"]
    assert d[0].explore and d[0].focus == sched.untried[0] == "skill"
    assert not d[1].explore and d[1].focus == "prompt"  # accepted before -> highest success rate
    assert all(x.avoid == ("config",) and x.budget == sched.budget for x in d)
    assert S.directives(3, hp, hist, traj, 0.017) == sched.arm_directives
    assert sched.arm_directives[0] == ArmDirective(arm="v1", strategy=d[0].strategy, component_focus=("skill",),
                                                   edit_budget=sched.budget, explore=True)
    assert sched.to_dict()["directives"][0]["focus"] == "skill"


def test_directives_not_stalled_and_custom_arms(mk):
    hp = profile("paper")
    sched = S.plan_round(0, hp, [], [0.5], delta=0.017, arms=["a", "b"])
    assert not sched.stalled and sched.exploration_slots == 0
    assert [d.focus for d in sched.directives] == ["prompt", "skill"]  # vocabulary order when untried
    assert not any(d.explore for d in sched.directives)
    with pytest.raises(ValueError):
        S.plan_round(0, hp, [], [0.5], 0.017, arms=["a", "a"])


def test_schedule_ignores_future_history(mk):
    hp = profile("paper")
    hist = [mk.rec(5, "v1", ["skill"], -0.5)]
    assert S.plan_round(2, hp, hist, [0.5] * 3, 0.017).prune == ()


def test_directives_property_loop(mk):
    rng = random.Random(7)
    for _ in range(200):
        n_arms = rng.randint(1, 5)
        hp = profile("paper", n_arms=n_arms, m_draft=rng.randint(0, 3), w=rng.randint(1, 3))
        t = rng.randint(0, 10)
        hist = [mk.rec(rng.randint(0, max(0, t - 1)), f"v{i}", rng.sample(COMPONENTS, rng.randint(1, 2)),
                       rng.choice([None, -0.1, 0.0, 0.05, 0.2]), accepted=rng.random() < 0.2)
                for i in range(rng.randint(0, 8))]
        traj = [rng.choice([0.5, 0.51, 0.6]) for _ in range(t + 1)]
        sched = S.plan_round(t, hp, hist, traj, 0.017)
        d = sched.directives
        assert len(d) == n_arms
        assert sum(x.explore for x in d) == sched.exploration_slots <= min(hp.m_draft, n_arms)
        for x in d:
            if x.explore:
                assert x.focus in sched.untried and sched.stalled
            elif x.focus is not None:
                assert x.focus not in sched.prune
            assert x.budget == S.edit_budget(t, hp.T, hp.b_min, hp.b_max)
        assert set(sched.untried).isdisjoint(tried_components([r for r in hist if r.round < t]))
