import json
import random

import pytest

from ci_lab.rrsi.readjudicate import load_round, readjudicate, round_record, save_round, verify
from ci_lab.rrsi.selection import SelectionInputs, select

DELTA = 0.125


def inputs(mk, inc, arms, *, s_star=0.5, delta=DELTA, round_no=2, **hp):
    hist = (mk.rec(0, "v1", ["skill"], 0.25, accepted=True), mk.rec(1, "v2", ["prompt"], -0.25))
    return SelectionInputs(round=round_no, incumbent=inc, arms=tuple(arms), s_star=s_star, delta=delta,
                           hp=mk.hp(**hp), history=hist, cases=None, incumbent_commit="inc")


def test_save_load_verify_roundtrip(mk, tmp_path):
    inc = mk.ev({"a": [0.5, 0.75], "b": [0.25, 0.25], "c": [1.0, 1.0]}, crit={"a": 1})
    arms = [mk.arm("v1", mk.ev({"a": [0.75, 0.75], "b": [0.5, None], "c": [1.0, 1.0]}, tokens=110)),
            mk.arm("v2", mk.ev({"a": [0.5, 0.5], "b": [0.25, 0.25], "c": [1.0, 0.75]}, tokens=60),
                   components=["memory"]),
            mk.arm("v3", None, status="rejected")]
    inp = inputs(mk, inc, arms)
    dec = select(inp)
    assert dec.decision != "rerun"
    path = tmp_path / "rounds" / "r2.json"
    save_round(path, inp, dec)
    assert path.read_bytes().count(b"\r") == 0
    loaded, stored = load_round(path)
    assert loaded.to_dict() == inp.to_dict() and stored == dec.to_dict()
    assert verify(path)
    again = readjudicate(path)
    assert not again.changed and again.decision.to_dict() == dec.to_dict()
    assert [r.accepted for r in again.records] == [a == dec.winner for a in ("v1", "v2")]
    assert readjudicate(inp).decision.to_dict() == dec.to_dict()


def test_round_record_round_mismatch(mk):
    inp = inputs(mk, mk.ev([0.5] * 4), [])
    dec = select(inputs(mk, mk.ev([0.5] * 4), [], round_no=3))
    with pytest.raises(ValueError):
        round_record(inp, dec)


def test_hp_override_redecides_but_delta_is_stored(mk):
    inc = mk.ev([0.5] * 4)
    arms = [mk.arm("v1", mk.ev([0.75] * 4, tokens=170))]  # dS 0.25, dC 0.7 > allowance 0.5
    inp = inputs(mk, inc, arms)
    rec = round_record(inp, select(inp))
    assert rec["decision"]["decision"] == "do_not_ship"
    loose = readjudicate(rec, hp=mk.hp(beta0=0.5))  # allowance 0.75
    assert loose.changed and loose.decision.decision == "ship" and loose.records[0].accepted
    # hp carries no delta for selection; the stored campaign delta is always used
    other = readjudicate(rec, hp=mk.hp(beta0=0.5, delta=0.5))
    assert other.decision.delta == DELTA and other.decision.decision == "ship"


def test_tampered_record_is_detected(mk):
    inp = inputs(mk, mk.ev([0.5] * 4), [mk.arm("v1", mk.ev([0.75] * 4))])
    rec = json.loads(json.dumps(round_record(inp, select(inp))))
    assert verify(rec)
    rec["decision"]["winner"] = None
    assert not verify(rec)
    rec2 = json.loads(json.dumps(round_record(inp, select(inp))))
    rec2["inputs"]["delta"] = 0.5  # a different stored delta re-decides the round
    assert readjudicate(rec2).decision.delta == 0.5


def test_readjudicate_equality_property(mk, tmp_path):
    rng = random.Random(23)
    for it in range(60):
        n = rng.randint(3, 7)
        inc = mk.ev({f"c{i}": [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]), rng.choice([0.0, 0.5, 1.0, None])]
                     for i in range(n)}, tokens=rng.choice([50, 100]))
        arms = [mk.arm(f"v{j}", mk.ev({f"c{i}": [rng.choice([0.0, 0.5, 0.75, 1.0, None]) for _ in range(2)]
                                       for i in range(n)}, tokens=rng.choice([40, 100, 150]),
                                      crit={"c0": rng.choice([0, 0, 1])}),
                       components=rng.sample(["prompt", "skill", "memory", "config"], 2))
                for j in range(rng.randint(0, 3))]
        inp = inputs(mk, inc, arms, s_star=rng.choice([0.25, 0.5, 0.75]), seed=it, n_bootstrap=150, k=2,
                     w_s=rng.choice([0.0, 1.0]), ci_lower_threshold=rng.choice(["zero", "delta"]))
        dec = select(inp)
        path = tmp_path / f"r{it}.json"
        save_round(path, inp, dec)
        assert verify(path), it
        assert readjudicate(json.loads(path.read_text(encoding="utf-8"))).decision == readjudicate(path).decision
