"""Layer 25: RRSI (M7) / OES (M4)-backed campaign collaborators (``ci_lab.campaign.rrsi_wiring``).

Every test runs the real ``ci_lab.rrsi`` / ``ci_lab.oes`` code (no mocks)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from ci_lab.campaign import defaults, records, rrsi_wiring
from ci_lab.campaign.deps import ROUND_CONTEXT
from ci_lab.campaign.driver import Campaign, merge_hyper
from ci_lab.campaign.fakes import fake_deps
from ci_lab.campaign.wiring import (
    AGL_DIR,
    AGL_URL_ENV,
    NetworkPolicyError,
    campaign_journal,
    wired_deps,
)
from ci_lab.contracts import (
    STRATEGIES,
    ArmResult,
    CriticVerdict,
    Edit,
    EvalResult,
    EvaluatorPin,
    TaskScore,
    Violation,
)
from ci_lab.oes.validate import validate_envelope
from ci_lab.rrsi import stats
from ci_lab.rrsi.params import HARNESS_STRATEGIES, Hyperparams

PIN = EvaluatorPin("5eedc0de", "stub-judge", "fake")
CASES = tuple(f"c{i:02d}" for i in range(12))
BASE = hashlib.sha1(b"base").hexdigest()
CID = "rrsi-camp"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ev(score: float | dict[str, float], *, split: str = "evolve", tree: str = "inc", tokens: int = 100,
       critical: tuple[str, ...] = (), cases: tuple[str, ...] = CASES) -> EvalResult:
    per = score if isinstance(score, dict) else dict.fromkeys(cases, score)
    return EvalResult(sha(tree), split, PIN,  # type: ignore[arg-type]
                      [TaskScore(c, 0, "stub", per[c], tokens_in=tokens,
                                 violations=(Violation("refund.unverified_identity", "critical", "x"),)
                                 if c in critical else ())
                       for c in cases])


def arm(name: str, score: float | dict[str, float], *, component: str = "prompt", strategy: str = "agent",
        base: str = BASE, **kw: Any) -> ArmResult:
    head = hashlib.sha1(name.encode()).hexdigest()
    return ArmResult(name, base, head_commit=head, harness_tree=sha(name),
                     edits=[Edit(component, f"hypothesis {name}", (f"harness/{component}/x.md",), head)],
                     critic=CriticVerdict(True), eval=ev(score, tree=name, **kw), status="evaluated",
                     strategy=strategy)


def hyper(round_no: int = 1, *, history: list[dict[str, Any]] | None = None, delta: float | None = 0.05,
          **over: Any) -> dict[str, Any]:
    h = merge_hyper({"rrsi": {"n_bootstrap": 1000}, **over})
    h[ROUND_CONTEXT] = {"round": round_no, "eid": f"{CID}-r{round_no:02d}", "delta": delta,
                        "incumbent_commit": BASE, "history": list(history or ())}
    return h


# ---------------------------------------------------------------- hyper / history

def test_hyperparams_maps_campaign_keys() -> None:
    hp = rrsi_wiring.hyperparams(merge_hyper({"arms": 3, "max_rounds": 5, "k": 2, "seed": 7}), delta=0.1)
    assert isinstance(hp, Hyperparams)
    assert (hp.n_arms, hp.T, hp.k, hp.seed, hp.delta) == (3, 5, 2, 7, 0.1)
    assert hp.strategies == HARNESS_STRATEGIES
    assert "agl" in defaults.DEFAULT_HYPER["strategies"]
    assert len(hp.strategies) <= hp.n_arms * hp.strategy_floor_every
    assert hp.b_max > hp.b_min  # no fixed budget -> RRSI anneals b_max -> b_min
    fixed = rrsi_wiring.hyperparams(merge_hyper({"budget": 2, "rrsi": {"n_bootstrap": 500}}))
    assert (fixed.b_min, fixed.b_max, fixed.n_bootstrap) == (2, 2, 500)
    one = rrsi_wiring.hyperparams(merge_hyper({"arms": 1}))
    assert one.strategy_floor_every == len(HARNESS_STRATEGIES)


def test_old_history_rows_parse_with_defaults() -> None:
    old = [{"eid": f"{CID}-r01", "round": 1, "decision": "ship", "winner": "v1", "tokens": 10,
            "arms": [{"arm": "v1", "component": "skill", "hypotheses": ["use refunds skill"], "accepted": True,
                      "score": 0.6, "status": "evaluated"},
                     {"arm": "v2", "component": "prompt", "hypotheses": [], "accepted": False, "score": None,
                      "status": "failed"}]},
           {"eid": f"{CID}-r02", "round": 2, "decision": "rerun", "winner": None, "tokens": 0, "arms": []}]
    recs = rrsi_wiring.history_records(old)
    assert len(recs) == 1
    (r,) = recs
    assert (r.round, r.arm, r.strategy, r.accepted, r.delta_s, r.novelty) == (0, "v1", "agent", True, None, 0)
    assert [(e.component, e.hypothesis) for e in r.edits] == [("skill", "use refunds skill")]
    assert rrsi_wiring.trajectory(old) == []  # no incumbent_score in old rows -> no stall signal
    assert rrsi_wiring.score_next(old[0]) == 0.6 and rrsi_wiring.best_score(old) == 0.6
    # And schedule still plans from them.
    directives = rrsi_wiring.schedule(3, hyper(3, history=old), old)
    assert len(directives) == 2


# ---------------------------------------------------------------- schedule

def _row(round_no: int, directives: list[dict[str, Any]], inc: float, winner: str | None) -> dict[str, Any]:
    arms = [{"arm": d["arm"], "component": d["component"], "hypotheses": [f"h{round_no}{d['arm']}"],
             "accepted": d["arm"] == winner, "score": inc + (0.2 if d["arm"] == winner else 0.0),
             "status": "evaluated", "strategy": d["strategy"], "evaluated": True, "admissible": d["arm"] == winner,
             "edits": [{"component": d["component"], "hypothesis": f"h{round_no}{d['arm']}", "files": [],
                        "commit": ""}], "cost": 100.0, "delta_s": 0.2 if d["arm"] == winner else 0.0,
             "delta_c": 0.0, "novelty": 0, "reasons": []} for d in directives]
    return {"eid": f"{CID}-r{round_no:02d}", "round": round_no, "decision": "ship" if winner else "do_not_ship",
            "winner": winner, "tokens": 0, "arms": arms, "incumbent_score": inc,
            "score_next": inc + 0.2 if winner else inc}


def test_schedule_thompson_allocates_all_strategies_with_floor() -> None:
    history: list[dict[str, Any]] = []
    per_round: list[list[str]] = []
    inc = 0.1
    for round_no in range(1, 9):
        directives = rrsi_wiring.schedule(round_no, hyper(round_no, history=history), history)
        assert [d["arm"] for d in directives] == ["v1", "v2"]
        for d in directives:
            assert d["strategy"] in HARNESS_STRATEGIES and d["budget"] >= 1
            assert d["strategy_reason"] in ("floor", "thompson")
            assert d["component"] != "guard"
            assert {"explore", "avoid", "stalled", "focus"} <= set(d)
        per_round.append([d["strategy"] for d in directives])
        winner = next((d["arm"] for d in directives if d["strategy"] == "gepa"), None)
        history.append(_row(round_no, directives, inc, winner))
        inc = history[-1]["score_next"]
    used = {s for r in per_round for s in r}
    assert used == set(HARNESS_STRATEGIES)
    first_floor = [s for round_strategies in per_round[:3] for s in round_strategies]
    assert set(first_floor) == set(HARNESS_STRATEGIES)  # never-run harness strategies are floored first
    hp = rrsi_wiring.hyperparams(merge_hyper({}))
    k = hp.strategy_floor_every
    for i in range(len(per_round) - k + 1):  # every strategy gets >= 1 arm in any K consecutive rounds
        assert {s for r in per_round[i:i + k] for s in r} == set(HARNESS_STRATEGIES), per_round
    # Budget anneals from b_max to b_min over T rounds.
    first = rrsi_wiring.schedule(1, hyper(1), [])[0]["budget"]
    last = rrsi_wiring.schedule(8, hyper(8, history=history), history)[0]["budget"]
    assert first == hp.b_max and hp.b_min <= last < first


def test_schedule_respects_strategy_subset_and_fixed_budget() -> None:
    h = hyper(1, strategies=["gepa", "skillopt"], budget=2)
    out = rrsi_wiring.schedule(1, h, [])
    assert sorted(d["strategy"] for d in out) == ["gepa", "skillopt"]
    assert {d["budget"] for d in out} == {2}


# ---------------------------------------------------------------- select (Alg. 2)

def test_select_ships_clearly_better_arm() -> None:
    inc = ev(0.2)
    arms = {"v1": arm("v1", 0.9), "v2": arm("v2", 0.25)}
    v = rrsi_wiring.select(inc, arms, 0.05, hyper(rrsi_profile="local"))
    assert v["decision"] == "ship" and v["winner"] == "v1"
    assert v["score_next"] == pytest.approx(0.9) and v["incumbent_score"] == pytest.approx(0.2)
    t1 = next(t for t in v["trace"] if t["arm"] == "v1")
    assert t1["admissible"] and t1["delta_s"] == pytest.approx(0.7)
    rr = v["rrsi"]
    v1 = next(a for a in rr["arms"] if a["arm"] == "v1")
    assert v1["branch"] == "cost" and v1["ci"]["required"] and v1["ci"]["lower"] > 0.05  # paired bootstrap CI
    assert {r["arm"] for r in v["attribution"]} == {"v1", "v2"}


def test_select_rejects_noise_level_arm() -> None:
    inc = ev(0.5)
    v = rrsi_wiring.select(inc, {"v1": arm("v1", 0.52), "v2": arm("v2", 0.5, tokens=130)}, 0.05, hyper())
    assert v["decision"] == "do_not_ship" and v["winner"] is None
    assert v["score_next"] == pytest.approx(0.5)
    reasons = {t["arm"]: t["reasons"] for t in v["trace"]}
    assert any(r.startswith("weighted_rule") for r in reasons["v1"])  # dS < delta, not novel: no gain
    assert any(r.startswith("weighted_rule") for r in reasons["v2"])  # pays cost for nothing


def test_select_rejects_safety_regression_even_if_better() -> None:
    inc = ev(0.2)
    v = rrsi_wiring.select(inc, {"v1": arm("v1", 0.9, critical=("c00",))}, 0.05, hyper())
    assert v["decision"] == "do_not_ship"
    assert any("critical safety" in r for r in v["trace"][0]["reasons"])


def test_select_rejects_arm_not_on_incumbent_and_reruns_on_pin_mismatch() -> None:
    inc = ev(0.2)
    v = rrsi_wiring.select(inc, {"v1": arm("v1", 0.9, base="0" * 40)}, 0.05, hyper())
    assert v["decision"] == "do_not_ship"
    bad = arm("v1", 0.9)
    assert bad.eval is not None
    bad.eval.pin = EvaluatorPin("deadbeef", "other", "fake")
    assert rrsi_wiring.select(inc, {"v1": bad}, 0.05, hyper())["decision"] == "rerun"


def test_select_floor_uses_best_so_far_from_history() -> None:
    hist = [_row(1, [{"arm": "v1", "component": "prompt", "strategy": "agent"}], 0.6, "v1")]  # S* = 0.8
    inc = ev(0.3)
    v = rrsi_wiring.select(inc, {"v1": arm("v1", 0.5)}, 0.05, hyper(2, history=hist))
    assert v["decision"] == "do_not_ship" and v["s_star"] == pytest.approx(0.8)
    assert any(r.startswith("floor") for r in v["trace"][0]["reasons"])


def _failed_rows(strategy: str, n: int) -> list[dict[str, Any]]:
    return [{"eid": f"{CID}-r{i:02d}", "round": i, "decision": "do_not_ship", "winner": None, "tokens": 0,
             "arms": [{"arm": "v1", "status": "failed", "strategy": strategy},
                      {"arm": "v2", "status": "evaluated", "strategy": "agent"}]} for i in range(1, n + 1)]


def test_select_excludes_sre_vetoed_strategy_without_changing_math() -> None:
    inc = ev(0.2)
    arms = {"v1": arm("v1", 0.95, strategy="gepa"), "v2": arm("v2", 0.9), "v3": arm("v3", 0.25)}
    rows = _failed_rows("gepa", 3)  # gepa error budget (3 failures) exhausted -> circuit break
    v = rrsi_wiring.select(inc, arms, 0.05, hyper(4, history=rows, rrsi_profile="local"))
    subset = rrsi_wiring.select(inc, {k: a for k, a in arms.items() if k != "v1"}, 0.05,
                                hyper(4, history=rows, rrsi_profile="local"))
    assert v["sre_vetoed"] == ["v1"] and v["winner"] == "v2" == subset["winner"]
    assert {k: v[k] for k in ("decision", "score_next", "s_star", "rrsi")} == \
        {k: subset[k] for k in ("decision", "score_next", "s_star", "rrsi")}
    assert v["trace"][:-1] == subset["trace"] and v["reasons"] == [*subset["reasons"], "sre_veto:v1"]
    assert v["trace"][-1] == {"arm": "v1", "admissible": False, "evaluated": False, "reason": "sre_vetoed",
                              "reasons": ["sre_vetoed"]}
    # Two failures: budget not yet exhausted -> output identical to the pre-governance selection.
    plain = rrsi_wiring.select(inc, arms, 0.05,
                               hyper(3, history=_failed_rows("gepa", 2), rrsi_profile="local"))
    assert "sre_vetoed" not in plain and plain["winner"] == "v1"
    # Every arm vetoed: no crash, nothing ships.
    v_all = rrsi_wiring.select(inc, {"v1": arms["v1"]}, 0.05, hyper(4, history=rows))
    assert v_all["decision"] != "ship" and v_all["sre_vetoed"] == ["v1"]


# ---------------------------------------------------------------- calibration / confirm

def test_calibrate_delta_is_bootstrap_quantile_with_floor() -> None:
    runs = [ev(0.5, split="aa", cases=CASES[:4]) for _ in range(5)]
    assert rrsi_wiring.calibrate_delta(runs, merge_hyper({})) == pytest.approx(0.25)  # 1/M floor
    noisy = [ev({c: round(random.Random(100 * i + j).uniform(0.2, 0.8), 3) for j, c in enumerate(CASES)},
                split="aa") for i in range(5)]
    h = merge_hyper({"rrsi": {"n_bootstrap": 2000}})
    hp = rrsi_wiring.hyperparams(h)
    got = rrsi_wiring.calibrate_delta(noisy, h)
    means = [stats.case_means(stats.index_scores(r.scores), stats.universe(noisy)) for r in noisy]
    want = stats.aa_delta(means, q=hp.aa_quantile, n_resamples=hp.n_bootstrap, seed=hp.seed).delta
    assert got == pytest.approx(want) and got >= 1 / len(CASES)
    assert got != pytest.approx(defaults.calibrate_delta(noisy, h))  # not the max-spread stand-in


def test_confirm_test_ships_significant_safe_improvement() -> None:
    h = merge_hyper({"rrsi": {"n_bootstrap": 2000}})
    out = rrsi_wiring.confirm_test(ev(0.2, split="heldout"), ev(0.8, split="heldout", tree="final"), h)
    assert out["decision"] == "ship" and out["p_value"] < out["alpha"]
    assert out["ci_lower"] > 0 and out["ci_upper"] >= out["ci_lower"] and out["safety_non_inferior"]


def test_confirm_test_rejects_no_gain_and_safety_regression() -> None:
    h = merge_hyper({"rrsi": {"n_bootstrap": 2000}})
    same = rrsi_wiring.confirm_test(ev(0.5, split="heldout"), ev(0.5, split="heldout", tree="f"), h)
    assert same["decision"] == "do_not_ship"
    unsafe = rrsi_wiring.confirm_test(ev(0.2, split="heldout"),
                                      ev(0.9, split="heldout", tree="f", critical=CASES[:6]), h)
    assert unsafe["decision"] == "do_not_ship" and not unsafe["safety_non_inferior"]
    assert unsafe["critical"] == {"h0": 0, "final": 6}


# ---------------------------------------------------------------- OES envelopes

SPLITS = {"evolve": "sha256:" + sha("evolve"), "heldout": "sha256:" + sha("heldout"), "aa": "sha256:" + sha("aa")}


def _round_record(sel: dict[str, Any], arms: dict[str, ArmResult], inc: ArmResult) -> dict[str, Any]:
    return {"eid": f"{CID}-r01", "campaignId": CID, "round": 1, "base_commit": BASE, "delta": 0.05,
            "decision": sel["decision"], "winner": sel["winner"], "selection": sel,
            "evals": {"incumbent": records.arm_to_dict(inc), "arms": {k: records.arm_to_dict(v)
                                                                      for k, v in arms.items()}},
            "directives": [{"arm": "v1", "budget": 2, "explore": True, "avoid": ["memory"], "stalled": False}],
            "incumbent_commit": BASE, "split_hashes": SPLITS}


def _incumbent(score: float) -> ArmResult:
    return ArmResult("inc", BASE, head_commit=BASE, harness_tree=sha("inc"), eval=ev(score), status="evaluated")


def test_round_envelope_validates_and_merges_extensions() -> None:
    arms = {"v1": arm("v1", 0.9), "v2": arm("v2", 0.25)}
    inc = _incumbent(0.2)
    sel = rrsi_wiring.select(inc.eval, arms, 0.05, hyper(rrsi_profile="local"))  # type: ignore[arg-type]
    rec = _round_record(sel, arms, inc)
    ext = {"com.example.note": {"note": "extra evidence"}}
    doc = rrsi_wiring.build_envelope("round", {**rec, "extensions": ext})
    assert validate_envelope(doc) == []
    assert doc["decision"]["outcome"] == "ship"
    assert doc["extensions"]["com.example.note"] == ext["com.example.note"]
    assert any(k.startswith("com.microsoft.ci") for k in doc["extensions"])  # builder's own extensions kept


def test_round_envelope_do_not_ship_validates() -> None:
    arms = {"v1": arm("v1", 0.52)}
    inc = _incumbent(0.5)
    sel = rrsi_wiring.select(inc.eval, arms, 0.05, hyper())  # type: ignore[arg-type]
    doc = rrsi_wiring.build_envelope("round", _round_record(sel, arms, inc))
    assert validate_envelope(doc) == [] and doc["decision"]["outcome"] != "ship"


def test_envelope_invalid_input_raises() -> None:
    arms = {"v1": arm("v1", 0.9)}
    inc = _incumbent(0.2)
    sel = rrsi_wiring.select(inc.eval, arms, 0.05, hyper())  # type: ignore[arg-type]
    rec = _round_record(sel, arms, inc)
    with pytest.raises(ValueError, match="extension-schema"):  # malformed guard evidence fails closed
        rrsi_wiring.build_envelope("round", {**rec, "extensions": {"com.microsoft.ci.guard": {"note": "x"}}})
    with pytest.raises(ValueError):
        rrsi_wiring.build_envelope("round", {**rec, "split_hashes": {"evolve": "not a hash"}})
    with pytest.raises(ValueError):
        rrsi_wiring.build_envelope("bogus", rec)
    with pytest.raises((ValueError, KeyError)):
        rrsi_wiring.build_envelope("calibration", {"eid": "x", "campaignId": CID, "delta": 0.1})


def test_calibration_and_confirm_envelopes_validate() -> None:
    runs = [ev(0.5, split="aa", tree="inc") for _ in range(5)]
    cal = rrsi_wiring.build_envelope("calibration", {
        "eid": f"{CID}-cal", "campaignId": CID, "decision": None, "delta": 0.25, "delta_method": "aa_bootstrap",
        "runs": [records.eval_to_dict(r) for r in runs], "harness_commit": BASE, "split_hashes": SPLITS})
    assert validate_envelope(cal) == []
    h = merge_hyper({"rrsi": {"n_bootstrap": 2000}})
    h0, final = ev(0.2, split="heldout"), ev(0.8, split="heldout", tree="final")
    decision = rrsi_wiring.confirm_test(h0, final, h)
    conf = rrsi_wiring.build_envelope("confirm", {
        "eid": f"{CID}-confirm", "campaignId": CID, **decision,
        "look": {"dataset_hash": SPLITS["heldout"], "look_no": 1, "planned": 1},
        "h0": records.eval_to_dict(h0), "final": records.eval_to_dict(final), "baseline_commit": BASE,
        "final_commit": hashlib.sha1(b"final").hexdigest(), "look_ledger_ref": "holdout-looks.jsonl",
        "planned_looks": 1, "split_hashes": SPLITS, "accepted_rounds": [f"{CID}-r01"]})
    assert validate_envelope(conf) == [] and conf["decision"]["outcome"] == "ship"


# ---------------------------------------------------------------- wiring

@pytest.fixture
def no_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "AGL_OPENAI_BASE_URL", AGL_URL_ENV, "CI_LAB_AGL_KEY",
                 "CI_S1_LLAMA_URL", "CI_S1_SYSTEMONE_URL"):
        monkeypatch.delenv(name, raising=False)


def _no_client(**_: Any) -> Any:
    raise AssertionError("building deps must not create chat clients")


def test_wired_deps_offline_uses_rrsi_adapters_and_agl_journal(tmp_path: Path, no_offline_env: None) -> None:
    from ci_lab.agl.journal import FileRolloutJournal

    deps = wired_deps("offline", run_root=tmp_path / "runs", ledger_dir=tmp_path / "experiments",
                      client_factory=_no_client, wt_root=tmp_path / "wt", domain_name="order_support")
    assert deps.schedule is rrsi_wiring.schedule and deps.select is rrsi_wiring.select
    assert deps.calibrate_delta is rrsi_wiring.calibrate_delta and deps.confirm_test is rrsi_wiring.confirm_test
    assert deps.build_envelope is rrsi_wiring.build_envelope
    journal = deps.domain.inner.journal  # type: ignore[attr-defined]
    assert isinstance(journal, FileRolloutJournal)
    assert Path(journal.root) == tmp_path / "runs" / AGL_DIR and journal.root.is_dir()
    over = wired_deps("offline", run_root=tmp_path / "runs", ledger_dir=tmp_path / "experiments",
                      client_factory=_no_client, wt_root=tmp_path / "wt", select=defaults.select)
    assert over.select is defaults.select  # overrides still win


def test_campaign_journal_mirrors_when_agl_url_set(tmp_path: Path) -> None:
    from ci_lab.agl.mirror import MirroringJournal

    env = {AGL_URL_ENV: "http://127.0.0.1:9", "CI_LAB_AGL_KEY": "k"}
    j = campaign_journal(tmp_path, offline=True, environ=env)
    assert isinstance(j, MirroringJournal) and Path(j.journal.root) == tmp_path / AGL_DIR
    assert j.client is not None
    with pytest.raises(NetworkPolicyError):
        campaign_journal(tmp_path, offline=True, environ={AGL_URL_ENV: "https://agl.example.com"})
    remote = campaign_journal(tmp_path, offline=False, environ={AGL_URL_ENV: "https://agl.example.com"})
    assert isinstance(remote, MirroringJournal)


# ---------------------------------------------------------------- campaign end-to-end

def _boost(edits, case_id, trial) -> float:
    return 0.9 if any("boost" in e for e in edits) else 0.1


def test_campaign_round_through_rrsi_adapters_writes_valid_envelopes(tmp_path: Path) -> None:
    deps = fake_deps(tmp_path, score_fn=_boost, schedule=rrsi_wiring.schedule, select=rrsi_wiring.select,
                     calibrate_delta=rrsi_wiring.calibrate_delta, confirm_test=rrsi_wiring.confirm_test,
                     build_envelope=rrsi_wiring.build_envelope)
    camp = Campaign.new(CID, "fake", {"arms": 2, "aa_repeats": 5, "max_rounds": 4,
                                      "rrsi_profile": "local",
                                      "strategies": list(STRATEGIES),
                                      "rrsi": {"n_bootstrap": 500}},
                        deps=deps, run_root=tmp_path / "runs")
    assert asyncio.run(camp.calibrate()) == pytest.approx(0.25)
    out = asyncio.run(camp.run(rounds=2))
    assert out["rounds"][0]["decision"] == "ship" and out["rounds"][0]["winner"] == "v1"
    assert out["rounds"][1]["decision"] == "do_not_ship"  # no further gain: Alg. 2 keeps the incumbent

    ledger = tmp_path / "experiments" / "campaigns" / CID
    for rel in ("calibration/envelope.json", f"rounds/{CID}-r01/envelope.json", f"rounds/{CID}-r02/envelope.json"):
        doc = json.loads((ledger / rel).read_text(encoding="utf-8"))
        assert validate_envelope(doc) == [], rel
    r1 = json.loads((ledger / "rounds" / f"{CID}-r01" / "envelope.json").read_text(encoding="utf-8"))
    assert r1["decision"]["outcome"] == "ship"

    rows = [json.loads(line) for line in (ledger / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["round"] for r in rows] == [1, 2]
    assert rows[0]["incumbent_score"] == pytest.approx(0.1) and rows[0]["score_next"] == pytest.approx(0.9)
    for a in rows[0]["arms"]:
        assert {"strategy", "edits", "cost", "delta_s", "delta_c", "novelty", "admissible", "reasons"} <= set(a)
    recs = rrsi_wiring.history_records(rows)
    assert {r.strategy for r in recs} <= set(STRATEGIES) and any(r.accepted for r in recs)
    assert len({a["strategy"] for r in rows for a in r["arms"]}) >= 3  # Thompson floor spreads strategies

    conf = asyncio.run(camp.confirm())
    assert conf["decision"] in ("ship", "do_not_ship")
    doc = json.loads((ledger / "confirm" / "envelope.json").read_text(encoding="utf-8"))
    assert validate_envelope(doc) == []

    again = camp.readjudicate(f"{CID}-r01")  # re-adjudicates via the same adapters + round context
    assert again["consistent"] and again["recomputed"] == {"decision": "ship", "winner": "v1"}
