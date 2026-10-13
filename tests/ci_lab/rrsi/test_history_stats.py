import json
import random
import zlib

import numpy as np
import pytest

from ci_lab.rrsi import history as H
from ci_lab.rrsi import stats

# ---------------------------------------------------------------- history


def test_history_jsonl_roundtrip(tmp_path, mk):
    p = tmp_path / "campaign" / "history.jsonl"
    assert H.read_jsonl(p) == []
    recs = [mk.rec(0, "v1", ["prompt", "skill"], 0.25, accepted=True), mk.rec(0, "v2", ["config"], -0.125)]
    assert H.append_jsonl(p, recs) == 2
    assert H.read_jsonl(p) == recs
    raw = p.read_bytes()
    assert b"\r\n" not in raw and len(raw.splitlines()) == 2
    with pytest.raises(ValueError):
        H.append_jsonl(p, [mk.rec(0, "v1", ["prompt"], 0.0)])  # duplicate (round, arm)
    H.append_jsonl(p, [mk.rec(1, "v1", ["memory"], None)])
    assert len(H.read_jsonl(p)) == 3
    H.write_jsonl(p, H.replace_round(H.read_jsonl(p), 0, [mk.rec(0, "v9", ["prompt"], 0.5)]))
    assert [(r.round, r.arm) for r in H.read_jsonl(p)] == [(1, "v1"), (0, "v9")]


def test_history_bad_line(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"round": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="h.jsonl:1"):
        H.read_jsonl(p)


def test_history_queries(mk):
    recs = [mk.rec(0, "v1", ["prompt", "skill"], 0.25, accepted=True), mk.rec(0, "v2", ["skill"], -0.1),
            mk.rec(1, "v1", ["memory"], None)]
    assert H.tried_components(recs) == {"prompt", "skill"}
    assert H.accepted_counts(recs) == {"prompt": 1, "skill": 1}
    assert H.has(recs, 0, "v2") and not H.has(recs, 1, "v2")
    assert H.before(recs, 1) == recs[:2]
    assert recs[0].components == ("prompt", "skill")
    ev = list(H.edit_events(recs))
    assert [(e.component, e.accepted) for e in ev] == [("prompt", True), ("skill", True), ("skill", False),
                                                       ("memory", False)]
    with pytest.raises(ValueError):
        H.replace_round(recs, 0, [mk.rec(1, "x", ["prompt"], 0.0)])


def test_history_nonfinite_is_json_safe(mk):
    r = H.HistoryRecord(round=0, arm="v1", edits=(), score=1.0, cost=1.0, delta_s=0.0, delta_c=float("inf"),
                        accepted=False)
    assert json.loads(json.dumps(r.to_dict(), allow_nan=False))["delta_c"] is None


# ---------------------------------------------------------------- aggregation


def test_mean_missing_zero():
    assert stats.mean_missing_zero([1.0, None, 0.5]) == 0.5
    assert stats.mean_missing_zero([1.0], n_expected=4) == 0.25
    assert stats.mean_missing_zero([]) == 0.0


def test_task_score_missing_rate_cost(mk):
    e = mk.ev({"a": [1.0, None], "b": [0.5, 0.5]}, tokens=[100, 0, 200, 300])
    idx = stats.index_scores(e.scores)
    keys = stats.universe([e])
    assert keys == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    assert stats.case_means(idx, keys) == {"a": 0.5, "b": 0.5}
    assert stats.task_score(idx, keys) == 0.5
    assert stats.missing_rate(idx, keys) == 0.25
    assert stats.mean_cost(idx, keys) == 200.0  # missing trial's tokens excluded
    fixed = stats.universe([e], cases=["a", "b", "c"], k=2)
    assert len(fixed) == 6 and stats.missing_rate(idx, fixed) == 0.5
    assert stats.task_score(idx, fixed) == pytest.approx(1 / 3)  # absent case c counts 0
    with pytest.raises(ValueError):
        stats.universe([e], cases=["a"])
    with pytest.raises(ValueError):
        stats.index_scores(e.scores + e.scores[:1])


def test_relative_cost_and_critical(mk):
    assert stats.relative_cost(150, 100) == 0.5
    assert stats.relative_cost(0, 0) == 0.0
    assert stats.relative_cost(1, 0) == float("inf")
    e = mk.ev([1.0, 1.0], crit={"c0": 2})
    idx = stats.index_scores(e.scores)
    assert stats.critical_count(idx, stats.universe([e])) == 2


# ---------------------------------------------------------------- bootstrap


def test_bootstrap_determinism_with_seed():
    cand = {f"c{i}": (i % 3) / 2 for i in range(30)}
    inc = {f"c{i}": (i % 2) / 2 for i in range(30)}
    a = stats.paired_bootstrap(cand, inc, n_resamples=2000, seed=[1, 2, 3])
    b = stats.paired_bootstrap(cand, inc, n_resamples=2000, seed=[1, 2, 3])
    c = stats.paired_bootstrap(cand, inc, n_resamples=2000, seed=[1, 2, 4])
    assert a == b and (a.lower, a.upper) != (c.lower, c.upper)
    assert a.lower <= a.mean <= a.upper and a.n_cases == 30
    rng = random.Random(3)
    for _ in range(50):
        d = np.array([rng.choice([-1.0, 0.0, 0.5, 1.0]) for _ in range(rng.randint(1, 25))])
        seed = stats.derive_seed(rng.randint(0, 10**6), "r", rng.randint(0, 9), "arm")
        x = stats.bootstrap_means(d, n_resamples=300, seed=seed)
        assert np.array_equal(x, stats.bootstrap_means(d, n_resamples=300, seed=seed))
        assert x.min() >= d.min() - 1e-12 and x.max() <= d.max() + 1e-12


def test_derive_seed_is_stable():
    assert stats.derive_seed(0, 3, "v1") == [0, 3, zlib.crc32(b"v1")]
    assert stats.derive_seed(5, "a") != stats.derive_seed(5, "b")


def test_paired_bootstrap_constant_and_empty():
    ci = stats.paired_bootstrap({"a": 0.75, "b": 0.75}, {"a": 0.5, "b": 0.5}, n_resamples=100)
    assert (ci.mean, ci.lower, ci.upper) == (0.25, 0.25, 0.25)
    # absent case on one side counts as 0
    assert stats.paired_case_deltas({"a": 1.0}, {"b": 1.0}).tolist() == [1.0, -1.0]
    assert stats.paired_bootstrap({}, {}, n_resamples=10).mean == 0.0


# ---------------------------------------------------------------- A/A delta


def test_aa_delta_floor():
    rep = {f"c{i}": 0.5 for i in range(12)}
    d = stats.aa_delta([rep, dict(rep), dict(rep)], n_resamples=200)
    assert d.quantile_value == 0.0 and d.delta == pytest.approx(1 / 12) and d.bound_by == "floor"
    assert d.pairs == 3 and d.repeats == 3
    assert stats.aa_delta([rep, rep], n_resamples=50, n_cases=40).delta == pytest.approx(1 / 40)


def test_aa_delta_quantile():
    # every case shifts by exactly 1 between repeats -> |dS| bootstrap is constant 1 > 1/2
    d = stats.aa_delta([{"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}], n_resamples=100)
    assert d.delta == 1.0 and d.bound_by == "quantile" and d.observed_abs == (1.0,)
    # mixed: q-quantile within [0, 1]; deterministic for a seed
    reps = [{f"c{i}": float((i + r) % 2) for i in range(10)} for r in range(3)]
    x = stats.aa_delta(reps, n_resamples=500, seed=4)
    assert x == stats.aa_delta(reps, n_resamples=500, seed=4)
    assert x.floor == 0.1 and 0.1 <= x.delta <= 1.0
    with pytest.raises(ValueError):
        stats.aa_delta([reps[0]])


def test_aa_repeats_for_precision():
    # s = sd([0.5, 0.6]) = 0.0707; (1.96 * 0.0707 / 0.05)^2 = 7.68 -> 8
    assert stats.aa_repeats_for_precision(0.05, [0.5, 0.6]) == 8
    assert stats.aa_repeats_for_precision(0.5, [0.5, 0.6]) == 5        # C8 minimum
    assert stats.aa_repeats_for_precision(0.01, [0.5, 0.6], maximum=20) == 20
    assert stats.aa_repeats_for_precision(0.05, [0.5, 0.5, 0.5]) == 5  # zero variance
    with pytest.raises(ValueError):
        stats.aa_repeats_for_precision(0.05, [0.5])
    with pytest.raises(ValueError):
        stats.aa_repeats_for_precision(0.0, [0.5, 0.6])


# ---------------------------------------------------------------- confirm / non-inferiority


def test_confirm_test():
    inc = {f"c{i}": 0.5 for i in range(20)}
    better = {c: 0.75 for c in inc}
    t = stats.confirm_test(better, inc, n_resamples=500)
    assert t.passed and t.p_value == 0.0 and t.estimate == 0.25 and t.bound == 0.25
    same = stats.confirm_test(dict(inc), inc, n_resamples=500)
    assert not same.passed and same.p_value == 1.0  # bound == 0 is not > 0
    one = {**inc, "c0": 1.0}  # a single improved case: lower bound hits 0
    assert not stats.confirm_test(one, inc, alpha=0.05, n_resamples=2000).passed


def test_non_inferiority():
    inc = {f"c{i}": 1.0 for i in range(10)}
    slightly = {c: 0.99 for c in inc}
    worse = {c: 0.9 for c in inc}
    assert stats.non_inferiority(slightly, inc, margin=0.02, n_resamples=200).passed
    r = stats.non_inferiority(worse, inc, margin=0.02, n_resamples=200)
    assert not r.passed and r.threshold == -0.02
    # violation counts: lower is better
    v_inc = {c: 1.0 for c in inc}
    v_more = {c: 2.0 for c in inc}
    assert not stats.non_inferiority(v_more, v_inc, margin=0.5, higher_is_better=False, n_resamples=100).passed
    assert stats.non_inferiority(v_inc, v_more, margin=0.0, higher_is_better=False, n_resamples=100).passed
    with pytest.raises(ValueError):
        stats.non_inferiority(inc, inc, margin=-1)
