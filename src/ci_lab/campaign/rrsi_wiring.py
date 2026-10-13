"""Real RRSI (M7) and OES (M4) collaborators for the campaign driver.

Adapters with the :mod:`ci_lab.campaign.deps` callable signatures, backed by
:mod:`ci_lab.rrsi` and :mod:`ci_lab.oes`. ``wired_deps`` (copilot/offline profiles)
injects them; :mod:`ci_lab.campaign.defaults` remains for the ``fake`` profile only.

==================  =========================================================
schedule            ``rrsi.schedule.plan_round`` (Alg. 1: annealed budget, stall,
                    exploration/prune, Thompson strategy allocation with floors)
select              ``rrsi.selection.select`` (Alg. 2: cost rule, weighted rule
                    with novelty, paired bootstrap CI, floor S*-delta, safety guards)
calibrate_delta     ``rrsi.stats.aa_delta`` (bootstrap quantile of A/A |dS|, 1/M floor)
confirm_test        ``rrsi.stats.confirm_test`` + safety ``non_inferiority``
build_envelope      ``oes.build`` calibration/round/confirm envelopes, sealed and
                    validated by ``oes.validate`` (fail closed: invalid -> ValueError)
==================  =========================================================

Round context: the campaign steps pass ``{**hyper, ROUND_CONTEXT: {...}}`` (see
:data:`ci_lab.campaign.deps.ROUND_CONTEXT`) carrying the calibrated ``delta`` (schedule
stall detection), the best-so-far ``s_star``, the round's ``incumbent_commit`` and the
ledger ``history`` rows from earlier rounds. Old ``history.jsonl`` rows (pre-RRSI wiring:
no edits/strategy/deltas) are still parsed with neutral defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ci_lab.campaign.challenger_lane import in_band, out_of_band
from ci_lab.campaign.deps import ROUND_CONTEXT
from ci_lab.campaign.records import arm_from_dict, eval_from_dict
from ci_lab.contracts import STRATEGIES, TEXT_COMPONENTS, ArmResult, Edit, EvalResult
from ci_lab.oes import build as oes_build
from ci_lab.oes import canonical
from ci_lab.oes.validate import validate_envelope
from ci_lab.rrsi import stats
from ci_lab.rrsi.attribution import attribute
from ci_lab.rrsi.codec import edit_from_dict, finite
from ci_lab.rrsi.history import HistoryRecord
from ci_lab.rrsi.params import Hyperparams, profile
from ci_lab.rrsi.schedule import plan_round
from ci_lab.rrsi.selection import SelectionDecision, SelectionInputs
from ci_lab.rrsi.selection import select as rrsi_select

DELTA_METHOD = "aa_bootstrap"
CONFIRM_METHOD = "one-sided paired case-level bootstrap"
_ENVELOPE_SPLITS = ("evolve", "heldout", "ood", "aa")


# ---------------------------------------------------------------- hyper / history

def context(hyper: Mapping[str, Any]) -> dict[str, Any]:
    return dict(hyper.get(ROUND_CONTEXT) or {})


def hyperparams(hyper: Mapping[str, Any], delta: float | None = None) -> Hyperparams:
    """Campaign hyper (``defaults.DEFAULT_HYPER`` keys) -> :class:`rrsi.params.Hyperparams`.

    Base profile ``hyper["rrsi_profile"]`` (default ``local``); campaign keys win over the
    profile (``arms`` -> n_arms, ``max_rounds`` -> T, ``k``, ``seed``, ``strategies``; a
    legacy fixed ``budget`` pins b_min = b_max, else ``b_min``/``b_max``), then
    ``hyper["rrsi"]`` overrides any Hyperparams field (unknown field -> TypeError).
    """
    base = profile(str(hyper.get("rrsi_profile") or "local"))
    strategies = tuple(hyper.get("strategies") or base.strategies)
    n_arms = int(hyper.get("arms", base.n_arms))
    changes: dict[str, Any] = {
        "T": int(hyper.get("max_rounds", base.T)), "k": int(hyper.get("k", base.k)), "n_arms": n_arms,
        "M": None, "seed": int(hyper.get("seed", base.seed)), "strategies": strategies,
        "strategy_floor_every": max(base.strategy_floor_every, math.ceil(len(strategies) / max(1, n_arms))),
    }
    if hyper.get("budget") is not None:
        changes["b_min"] = changes["b_max"] = int(hyper["budget"])
    else:
        b_min = int(hyper.get("b_min") or base.b_min)
        changes["b_min"], changes["b_max"] = b_min, max(b_min, int(hyper.get("b_max") or base.b_max))
    if delta is not None:
        changes["delta"] = float(delta)
    changes.update(dict(hyper.get("rrsi") or {}))
    return base.with_(**changes)


def _edits(arm: Mapping[str, Any]) -> tuple[Edit, ...]:
    if arm.get("edits") is not None:
        return tuple(edit_from_dict(dict(e)) for e in arm["edits"])
    component = arm.get("component") or "unknown"  # old rows: component + hypotheses only
    hyps = list(arm.get("hypotheses") or [])
    return tuple(Edit(component=component, hypothesis=str(h), files=(), commit="") for h in hyps)


def history_records(rows: Sequence[Mapping[str, Any]]) -> list[HistoryRecord]:
    """Ledger ``history.jsonl`` rows (1-based ``round``) -> rrsi ``HistoryRecord`` (0-based round).

    Rerun rounds and unevaluated arms are skipped (Alg. 2 records nothing for them). Old rows
    without per-arm ``strategy``/``edits``/``delta_s`` parse as agent arms with unmeasured
    deltas (they still count for tried/accepted components)."""
    out: list[HistoryRecord] = []
    for row in rows:
        if row.get("decision") == "rerun" or row.get("round") is None:
            continue
        t = int(row["round"]) - 1
        for arm in row.get("arms") or ():
            if out_of_band(arm):  # challenger lane rows never feed history
                continue
            if arm.get("evaluated") is False or arm.get("status", "evaluated") != "evaluated" \
                    or arm.get("score") is None:
                continue
            strategy = arm.get("strategy") or "agent"
            out.append(HistoryRecord(
                round=t, arm=str(arm["arm"]), edits=_edits(arm), score=arm.get("score"), cost=arm.get("cost"),
                delta_s=arm.get("delta_s"), delta_c=arm.get("delta_c"), accepted=bool(arm.get("accepted")),
                novelty=int(arm.get("novelty") or 0), admissible=bool(arm.get("admissible", arm.get("accepted"))),
                reasons=tuple(arm.get("reasons") or ()), strategy=strategy if strategy in STRATEGIES else "agent"))
    return out


def score_next(row: Mapping[str, Any]) -> float | None:
    """Incumbent score after the round (``S_{t+1}``) from a history row, old or new."""
    if row.get("score_next") is not None:
        return float(row["score_next"])
    if row.get("decision") == "ship" and row.get("winner"):
        won = next((a for a in row.get("arms") or () if a.get("arm") == row["winner"]), None)
        if won is not None and won.get("score") is not None:
            return float(won["score"])
    return None if row.get("incumbent_score") is None else float(row["incumbent_score"])


def trajectory(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """``trajectory[i]`` = incumbent score at the start of round i (0-based); truncated at the
    first unknown value (``stall_flag`` is False beyond the trajectory)."""
    ordered = sorted(rows, key=lambda r: int(r.get("round") or 0))
    if not ordered or ordered[0].get("incumbent_score") is None:
        return []
    out = [float(ordered[0]["incumbent_score"])]
    for row in ordered:
        nxt = score_next(row)
        if nxt is None:
            break
        out.append(nxt)
    return out


def best_score(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Best-so-far ``S*`` over prior rounds (None before the first round)."""
    vals = [float(v) for r in rows for v in (r.get("s_star"), score_next(r), r.get("incumbent_score"))
            if v is not None]
    return max(vals) if vals else None


# ---------------------------------------------------------------- schedule (Alg. 1)

def schedule(round_no: int, hyper: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ctx = context(hyper)
    rows = [r for r in history if int(r.get("round") or 0) < round_no]
    delta = ctx.get("delta")
    hp = hyperparams(hyper, delta)
    t = round_no - 1
    plan = plan_round(t, hp, history_records(rows), trajectory(rows), float(delta or 0.0),
                      experiment_id=ctx.get("eid"))
    out = []
    for i, d in enumerate(plan.directives):
        if d.strategy == "guard":
            component = "guard"
        else:
            component = d.focus or TEXT_COMPONENTS[(round_no + i) % len(TEXT_COMPONENTS)]
        out.append({"arm": d.arm, "component": component, "strategy": d.strategy, "budget": d.budget,
                    "explore": d.explore, "focus": d.focus, "avoid": list(d.avoid), "stalled": d.stalled,
                    "strategy_reason": d.strategy_reason})
    return out


# ---------------------------------------------------------------- select (Alg. 2)

def selection_inputs(incumbent: EvalResult, arms: Mapping[str, ArmResult], delta: float,
                     hyper: Mapping[str, Any]) -> SelectionInputs:
    ctx = context(hyper)
    hp = hyperparams(hyper, delta)
    round_no = int(ctx.get("round") or 1)
    arms = in_band(arms)  # challenger lane arms never reach SelectionInputs
    rows = [r for r in ctx.get("history") or () if int(r.get("round") or 0) < round_no]
    evaluated = [a.eval for a in arms.values() if a.eval is not None and a.status == "evaluated"]
    keys = stats.universe([incumbent, *evaluated])
    s_t = stats.task_score(stats.index_scores(incumbent.scores), keys)
    prior = ctx.get("s_star")
    prior = best_score(rows) if prior is None else float(prior)
    s_star = s_t if prior is None else max(prior, s_t)
    return SelectionInputs(round=round_no, incumbent=incumbent, arms=tuple(arms[k] for k in sorted(arms)),
                           s_star=s_star, delta=float(delta), hp=hp, history=tuple(history_records(rows)),
                           incumbent_commit=ctx.get("incumbent_commit"))


def verdict(decision: SelectionDecision, arms: Sequence[ArmResult]) -> dict[str, Any]:
    """SelectionDecision -> the verdict mapping the round steps validate and record."""
    trace = []
    for a in decision.arms:
        trace.append({"arm": a.arm, "admissible": a.admissible, "evaluated": a.evaluated,
                      "delta_s": finite(a.delta_s), "delta_c": finite(a.delta_c), "novelty": a.novelty,
                      "score": finite(a.score), "cost": finite(a.cost),
                      "reason": a.reasons[0] if a.reasons else None, "reasons": list(a.reasons)})
    return {"decision": decision.decision, "winner": decision.winner, "delta": decision.delta,
            "incumbent_score": finite(decision.incumbent.get("score")), "score_next": finite(decision.score_next),
            "s_star": finite(decision.s_star_after), "reasons": list(decision.reasons), "trace": trace,
            "rrsi": decision.to_dict(), "attribution": [r.to_dict() for r in attribute(decision, arms)]}


def _vetoed(arms: Mapping[str, ArmResult], hyper: Mapping[str, Any]) -> list[str]:
    ctx = context(hyper)
    round_no = int(ctx.get("round") or 1)
    rows = [r for r in ctx.get("history") or () if int(r.get("round") or 0) < round_no]
    if not rows:
        return []
    from ci_lab.governance.sre import arm_vetoes  # lazy: agent_sre import is heavy

    vetoes = set(arm_vetoes(rows))
    return sorted(k for k, a in arms.items() if (a.strategy or "agent") in vetoes)


def select(incumbent: EvalResult, arms: Mapping[str, ArmResult], delta: float,
           hyper: Mapping[str, Any]) -> dict[str, Any]:
    """Alg. 2 over the arms whose strategy is not SRE-vetoed (error budget exhausted in prior rounds).

    Vetoed arms are excluded before selection, so the math over the remaining arms is unchanged; they
    are recorded in ``sre_vetoed`` and the trace with reason ``sre_vetoed``.
    """
    vetoed = _vetoed(arms, hyper)
    kept = {k: a for k, a in arms.items() if k not in vetoed}
    inputs = selection_inputs(incumbent, kept, delta, hyper)
    out = verdict(rrsi_select(inputs), inputs.arms)
    if vetoed:
        out["sre_vetoed"] = vetoed
        out["reasons"] = [*out["reasons"], *(f"sre_veto:{k}" for k in vetoed)]
        out["trace"] = [*out["trace"], *({"arm": k, "admissible": False, "evaluated": False,
                                          "reason": "sre_vetoed", "reasons": ["sre_vetoed"]} for k in vetoed)]
    return out


# ---------------------------------------------------------------- calibration / confirm

def _case_means(results: Sequence[EvalResult]) -> list[dict[str, float]]:
    keys = stats.universe(results)
    return [stats.case_means(stats.index_scores(r.scores), keys) for r in results]


def calibrate_delta(results: Sequence[EvalResult], hyper: Mapping[str, Any]) -> float:
    hp = hyperparams(hyper)
    return stats.aa_delta(_case_means(results), q=hp.aa_quantile, n_resamples=hp.n_bootstrap, seed=hp.seed).delta


def confirm_test(h0: EvalResult, final: EvalResult, hyper: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered one-sided paired bootstrap (final > h0) + safety non-inferiority."""
    hp = hyperparams(hyper)
    keys = stats.universe([h0, final])
    i0, i1 = stats.index_scores(h0.scores), stats.index_scores(final.scores)
    m0, m1 = stats.case_means(i0, keys), stats.case_means(i1, keys)
    ct = stats.confirm_test(m1, m0, alpha=hp.alpha, n_resamples=hp.n_bootstrap,
                            seed=stats.derive_seed(hp.seed, "confirm"))
    ci = stats.paired_bootstrap(m1, m0, n_resamples=hp.n_bootstrap, level=1.0 - 2 * hp.alpha,
                                seed=stats.derive_seed(hp.seed, "confirm"))
    crit = {c: 0.0 for c in m0}
    rate0, rate1 = dict(crit), dict(crit)
    for idx, rate in ((i0, rate0), (i1, rate1)):
        for c in rate:
            ks = [k for k in keys if k[0] == c]
            rate[c] = stats.critical_count(idx, ks) / max(1, len(ks))
    ni = stats.non_inferiority(rate1, rate0, margin=hp.safety_ni_margin, alpha=hp.alpha, higher_is_better=False,
                               n_resamples=hp.n_bootstrap, seed=stats.derive_seed(hp.seed, "confirm-safety"))
    crit0, crit1 = stats.critical_count(i0, keys), stats.critical_count(i1, keys)
    safe = ni.passed and crit1 <= crit0
    ship = ct.passed and ct.p_value < hp.alpha and safe
    return {"decision": "ship" if ship else "do_not_ship", "delta_s": ct.estimate, "p_value": ct.p_value,
            "ci_lower": ct.bound, "ci_upper": ci.upper, "ci_level": ci.level, "alpha": hp.alpha,
            "method": CONFIRM_METHOD, "safety_non_inferior": safe,
            "critical": {"h0": crit0, "final": crit1}, "confirm": ct.to_dict(), "safety": ni.to_dict()}


# ---------------------------------------------------------------- OES envelopes

def _split_hashes(record: Mapping[str, Any]) -> dict[str, str]:
    return {k: v for k, v in dict(record.get("split_hashes") or {}).items() if k in _ENVELOPE_SPLITS}


def _num(x: Any) -> float | None:
    return finite(x) if isinstance(x, (int, float)) else None


def _candidates(sel: Mapping[str, Any]) -> list[dict[str, Any]]:
    rr = sel.get("rrsi")
    if rr:
        out = []
        for a in rr.get("arms") or ():
            rule = a.get("rule") or {}
            ci = a.get("ci") or {}
            out.append({"variantId": a["arm"], "admissible": bool(a.get("admissible")),
                        "deltaS": _num(a.get("delta_s")), "deltaC": _num(a.get("delta_c")),
                        "novelty": a.get("novelty"), "ciLowerBound": _num(ci.get("lower")),
                        "ciLevel": ci.get("level") or 0.95, "rule": a.get("branch") or "none",
                        "objective": _num(rule.get("value")) if a.get("branch") == "weighted" else None,
                        "reasons": list(a.get("reasons") or ())})
        out += [{"variantId": k, "admissible": False, "reasons": ["sre_vetoed"]} for k in sel.get("sre_vetoed") or ()]
        return out
    return [{"variantId": t["arm"], "admissible": bool(t.get("admissible")), "deltaS": _num(t.get("delta_s")),
             "deltaC": _num(t.get("delta_c")), "reasons": [t["reason"]] if t.get("reason") else []}
            for t in sel.get("trace") or ()]


def _round(record: Mapping[str, Any], hyper: Mapping[str, Any]) -> dict[str, Any]:
    sel = dict(record.get("selection") or {})
    evals = record["evals"]
    inc = arm_from_dict(evals["incumbent"])
    if inc.eval is None:
        raise ValueError("round envelope needs the incumbent eval")
    arms = [arm_from_dict(v) for _, v in sorted(evals["arms"].items())]
    s_inc = sel.get("incumbent_score")
    s_inc = float(s_inc) if s_inc is not None else stats.task_score(stats.index_scores(inc.eval.scores),
                                                                    stats.universe([inc.eval]))
    s_next = sel.get("score_next")
    winner = sel.get("winner") if sel.get("decision") == "ship" else None
    selection = {"winner": winner, "incumbentScore": s_inc,
                 "newIncumbentScore": float(s_next) if s_next is not None else s_inc,
                 "candidates": _candidates(sel)}
    directives = list(record.get("directives") or ())
    sched = oes_build.Schedule(
        budget=int(directives[0].get("budget", 1)) if directives else 1,
        stall=any(bool(d.get("stalled")) for d in directives),
        exploration_slots=sum(bool(d.get("explore")) for d in directives),
        prune_set=tuple(dict.fromkeys(c for d in directives for c in d.get("avoid") or ())))
    p = dict((sel.get("rrsi") or {}).get("params") or {})
    hp = hyperparams(hyper)
    params = oes_build.RrsiParams(delta=float(record["delta"]), delta_method=str(record.get("delta_method")
                                                                               or DELTA_METHOD),
                                  beta0=float(p.get("beta0", hp.beta0)), beta1=float(p.get("beta1", hp.beta1)),
                                  ws=float(p.get("w_s", hp.w_s)), wc=float(p.get("w_c", hp.w_c)),
                                  wn=float(p.get("w_n", hp.w_n)))
    return oes_build.round_envelope(
        record["campaignId"], int(record["round"]), incumbent=inc.eval,
        incumbent_commit=str(record.get("incumbent_commit") or record["base_commit"]), arms=arms,
        selection=selection, schedule=sched, params=params, split_hashes=_split_hashes(record))


def _calibration(record: Mapping[str, Any]) -> dict[str, Any]:
    runs = [eval_from_dict(dict(r)) for r in record["runs"]]
    return oes_build.calibration_envelope(
        record["campaignId"], runs, delta=float(record["delta"]),
        delta_method=str(record.get("delta_method") or DELTA_METHOD), harness_commit=str(record["harness_commit"]),
        split_hashes=_split_hashes(record))


def _confirm(record: Mapping[str, Any]) -> dict[str, Any]:
    look = dict(record.get("look") or {})
    st = oes_build.ConfirmStats(p_value=float(record["p_value"]), ci_lower=float(record["ci_lower"]),
                                ci_upper=_num(record.get("ci_upper")), ci_level=float(record.get("ci_level", 0.95)),
                                method=str(record.get("method") or CONFIRM_METHOD))
    holdout = oes_build.Holdout(dataset_hash=str(look["dataset_hash"]), looks_used=int(look.get("look_no") or 1),
                                planned_looks=int(look.get("planned") or record.get("planned_looks") or 1))
    return oes_build.confirm_envelope(
        record["campaignId"], baseline=eval_from_dict(dict(record["h0"])), final=eval_from_dict(dict(record["final"])),
        baseline_commit=str(record["baseline_commit"]), final_commit=str(record["final_commit"]), stats=st,
        holdout=holdout, look_ledger_ref=str(record.get("look_ledger_ref") or "holdout-looks.jsonl"),
        split_hashes=_split_hashes(record), alpha=float(record.get("alpha", 0.05)),
        accepted_rounds=list(record.get("accepted_rounds") or ()))


def envelope_builder(hyper: Mapping[str, Any] | None = None) -> Any:
    """``build_envelope`` bound to the campaign hyper (RRSI params fallback for old selections)."""
    bound = dict(hyper or {})

    def build_envelope(kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
        if kind == "round":
            doc = _round(record, bound)
        elif kind == "calibration":
            doc = _calibration(record)
        elif kind == "confirm":
            doc = _confirm(record)
        else:
            raise ValueError(f"unknown envelope kind {kind!r}")
        if record.get("extensions"):
            doc["extensions"] = {**doc.get("extensions", {}), **dict(record["extensions"])}
        doc = canonical.seal(doc)
        errors = validate_envelope(doc)
        if errors:
            raise ValueError(f"invalid OES {kind} envelope for {record.get('eid')}: " + "; ".join(errors[:10]))
        outcome = (doc.get("decision") or {}).get("outcome")
        if record.get("decision") == "ship" and outcome != "ship":
            raise ValueError(f"{kind} envelope for {record.get('eid')} decided {outcome!r}, not 'ship'")
        return doc

    return build_envelope


build_envelope = envelope_builder()
