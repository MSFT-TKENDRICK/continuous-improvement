import json
import math
import random
from dataclasses import replace

import pytest

from ci_lab.rrsi import frontier as fr_mod
from ci_lab.rrsi.attribution import STRUCTURAL_COMPONENTS, attribute, component_stats, novelty
from ci_lab.rrsi.selection import SelectionInputs, select

DELTA = 0.125


def decide(mk, inc, arms, *, round_no=0, s_star=0.5, delta=DELTA, **hp):
    inp = SelectionInputs(round=round_no, incumbent=inc, arms=tuple(arms), s_star=s_star, delta=delta,
                          hp=mk.hp(**hp))
    return select(inp)


# ---------------------------------------------------------------- attribution


def test_structural_components_follow_contract_vocabulary():
    assert STRUCTURAL_COMPONENTS == ("client_tool", "skill", "memory")


def test_novelty(mk):
    edits = [mk.edit("skill"), mk.edit("memory"), mk.edit("prompt"), mk.edit("skill")]
    assert novelty(edits, {}) == 2
    assert novelty(edits, {"skill": 1}) == 1
    assert novelty(edits, [mk.rec(0, "v1", ["memory"], 0.1, accepted=True)]) == 1
    assert novelty(edits, [mk.rec(0, "v1", ["memory", "skill"], -0.1, accepted=False)]) == 2
    assert novelty([], {}) == 0


def test_attribute_marks_only_winner(mk):
    inc = mk.ev([0.5] * 4)
    arms = [mk.arm("a", mk.ev([0.75] * 4), components=["skill"]),
            mk.arm("b", mk.ev([1.0] * 4), components=["prompt", "config"]),
            mk.arm("c", mk.ev([0.25] * 4), components=["memory"]),
            mk.arm("d", None, status="failed")]
    d = decide(mk, inc, arms)
    assert d.winner == "b"
    recs = attribute(d, arms)
    assert [(r.arm, r.accepted) for r in recs] == [("a", False), ("b", True), ("c", False)]
    b = recs[1]
    assert b.round == 0 and b.delta_s == 0.5 and b.score == 1.0 and b.components == ("prompt", "config")
    assert recs[0].admissible and not recs[2].admissible
    assert all(json.dumps(r.to_dict(), allow_nan=False) for r in recs)


def test_attribute_no_ship_and_rerun(mk):
    inc = mk.ev([0.5] * 4)
    arms = [mk.arm("a", mk.ev([0.25] * 4))]
    assert [r.accepted for r in attribute(decide(mk, inc, arms), arms)] == [False]
    bad = mk.ev([0.5, None, None, 0.5])
    assert attribute(decide(mk, bad, arms), arms) == []


def test_component_stats_hand_computed(mk):
    recs = [mk.rec(0, "v1", ["prompt", "skill"], 0.25, accepted=True),
            mk.rec(0, "v2", ["prompt"], -0.5),
            mk.rec(1, "v1", ["skill"], 0.0),
            mk.rec(1, "v2", ["memory"], None)]  # not measured -> ignored
    st = component_stats(recs)
    assert st["prompt"].tried == 2 and st["prompt"].accepted == 1 and st["prompt"].mean_delta_s == -0.125
    assert st["prompt"].best_delta_s == 0.25 and st["prompt"].success_rate == 0.5
    assert st["skill"].tried == 2 and st["skill"].mean_delta_s == 0.125
    assert st["memory"].tried == 0 and st["memory"].success_rate == 0.0 and st["memory"].best_delta_s == -math.inf
    assert st["memory"].to_dict()["best_delta_s"] is None


# ---------------------------------------------------------------- frontier


def start(score=0.5):
    return fr_mod.initial("camp", commit="inc", harness_tree="t0", score=score, cost=100.0, delta=DELTA)


def test_initial_frontier():
    f = start()
    assert f.round == 0 and f.s_star == 0.5 and f.incumbent.commit == "inc" and f.scores == [0.5]
    with pytest.raises(ValueError):
        fr_mod.initial("camp", commit="c", harness_tree="t", score=0.5, cost=1.0, delta=-0.1)


def test_frontier_ship(mk):
    f = start()
    arms = [mk.arm("a", mk.ev([0.75] * 4, tokens=120))]
    d = decide(mk, mk.ev([0.5] * 4), arms)
    g = fr_mod.advance(f, d, arms)
    assert g.round == 1 and g.s_star == 0.75 and g.reruns == 0
    assert g.incumbent == fr_mod.TrajectoryPoint(1, 0.75, 120.0, "head-a", "tree-a", "a")


def test_frontier_do_not_ship_keeps_incumbent_and_ratchets(mk):
    f = start()
    arms = [mk.arm("a", mk.ev([0.25] * 4))]
    d = decide(mk, mk.ev([0.75] * 4), arms)  # re-measured incumbent scored higher than recorded
    g = fr_mod.advance(f, d, arms)
    assert d.decision == "do_not_ship" and g.incumbent.commit == "inc" and g.incumbent.arm is None
    assert g.incumbent.score == 0.75 and g.s_star == 0.75
    d2 = decide(mk, mk.ev([0.25] * 4), [], round_no=1, s_star=g.s_star)
    h = fr_mod.advance(g, d2, [])
    assert h.incumbent.score == 0.25 and h.s_star == 0.75  # S* never decreases


def test_frontier_rerun_only_counts(mk):
    f = start()
    d = decide(mk, mk.ev([None, None, 0.5, 0.5]), [])
    g = fr_mod.advance(f, d, [])
    assert d.decision == "rerun" and g.round == 0 and g.reruns == 1 and g.trajectory == f.trajectory
    assert g.note.startswith("baseline_invalid")
    assert fr_mod.advance(g, d, []).reruns == 2


def test_frontier_errors(mk):
    f = start()
    arms = [mk.arm("a", mk.ev([0.75] * 4))]
    d = decide(mk, mk.ev([0.5] * 4), arms)
    with pytest.raises(ValueError, match="round"):
        fr_mod.advance(replace(f, round=3), d, arms)
    with pytest.raises(ValueError, match="delta"):
        fr_mod.advance(replace(f, delta=0.25), d, arms)
    with pytest.raises(ValueError, match="not among"):
        fr_mod.advance(f, d, [])
    with pytest.raises(ValueError, match="based on"):
        fr_mod.advance(f, d, [mk.arm("a", mk.ev([0.75] * 4), base="other")])
    no_head = replace(arms[0], head_commit=None)
    with pytest.raises(ValueError, match="head_commit"):
        fr_mod.advance(f, d, [no_head])
    with pytest.raises(ValueError, match="paused"):
        fr_mod.advance(fr_mod.pause(f, "budget"), d, arms)


def test_pause_resume_finish_rollback_roundtrip(mk):
    f = start()
    p = fr_mod.pause(f, "budget")
    assert p.status == "paused" and p.note == "budget"
    assert fr_mod.resume(p) == f
    with pytest.raises(ValueError):
        fr_mod.resume(f)
    assert fr_mod.finish(f).status == "done"
    arms = [mk.arm("a", mk.ev([1.0] * 4))]
    g = fr_mod.advance(f, decide(mk, mk.ev([0.5] * 4), arms), arms)
    assert g.s_star == 1.0
    r = fr_mod.rollback(g, 0)
    assert r.round == 0 and r.trajectory == f.trajectory and r.s_star == 0.5
    with pytest.raises(ValueError):
        fr_mod.rollback(g, 2)
    assert fr_mod.Frontier.from_dict(json.loads(json.dumps(g.to_dict(), allow_nan=False))) == g


def test_frontier_property_loop(mk):
    rng = random.Random(5)
    for it in range(40):
        f = start(rng.choice([0.25, 0.5]))
        for t in range(6):
            inc_commit = f.incumbent.commit
            inc = mk.ev([rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, None]) for _ in range(5)])
            arms = [mk.arm(f"v{j}", mk.ev([rng.choice([0.0, 0.5, 0.75, 1.0]) for _ in range(5)],
                                          tokens=rng.choice([50, 100, 200])), base=inc_commit)
                    for j in range(rng.randint(0, 3))]
            inp = SelectionInputs(round=f.round, incumbent=inc, arms=tuple(arms), s_star=f.s_star, delta=DELTA,
                                  hp=mk.hp(n_bootstrap=100, seed=it), incumbent_commit=inc_commit)
            d = select(inp)
            g = fr_mod.advance(f, d, arms)
            assert g.s_star >= f.s_star
            if d.decision == "rerun":
                assert g.round == f.round and g.trajectory == f.trajectory
            else:
                assert g.round == f.round + 1 and len(g.trajectory) == len(f.trajectory) + 1
                assert (g.incumbent.arm is not None) == (d.decision == "ship")
                if d.decision == "do_not_ship":
                    assert g.incumbent.commit == inc_commit
            f = g
