"""Algorithm 2 (selection round) with the non-compensatory guards of design v1 §6 / v2 §9.

Per evaluated arm, against the *contemporaneous* incumbent evaluation (same round, paired
by (case, trial); no cross-round cache — C8):

* dS = S' - S_t (task-level means, missing = 0); dC = (C' - C_t) / C_t.
* If dS > delta: cost rule  dC <= beta0 + beta1 * dS  (Eq. 7) AND, when
  ``hp.require_ci_lower`` (C16), the bootstrap lower CI bound of dS > 0 (or > delta).
  Else (inside the band): weighted rule  w_s dS - w_c dC + w_n nu > 0  (Eq. 17).
* Guards (non-compensatory): any increase in critical safety violations, a critic
  rejection, a stale base commit, or a domain guard failure make the arm inadmissible.
* Floor: S' >= S* - delta (Eq. 5).
* Winner = argmax S' over admissible arms (ties: lower cost, higher CI lower bound, arm
  name); S* <- max(S*, S_{t+1}).

Round-level quality failures (incumbent missing-trial rate > ``missing_invalid_frac``,
evaluator-pin or split mismatch, no trials) yield decision ``rerun`` and change nothing.

C16 vs the paper: the CI gate only *narrows* the paper's acceptance region — an arm that
clears the paper's cost rule but whose lower CI bound does not clear the threshold is
rejected (it does NOT fall back to the within-band weighted rule). Within-band
acceptances (weighted rule) are never claims of a score gain, so the gate does not
apply to them; they still face the floor and every guard.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ci_lab.contracts import ArmResult, EvalResult

from . import stats
from .attribution import novelty
from .codec import arm_from_dict, arm_to_dict, eval_from_dict, eval_to_dict, finite
from .history import HistoryRecord, accepted_counts
from .params import Hyperparams

Decision = Literal["ship", "do_not_ship", "rerun"]
DomainGuard = Callable[[ArmResult, EvalResult], Sequence[str]]  # returns failure reasons; empty = pass


@dataclass(frozen=True)
class SelectionInputs:
    """Everything Algorithm 2 needs; stored per round so selection can be re-adjudicated."""

    round: int
    incumbent: EvalResult
    arms: tuple[ArmResult, ...]
    s_star: float
    delta: float
    hp: Hyperparams
    history: tuple[HistoryRecord, ...] = ()
    cases: tuple[str, ...] | None = None
    incumbent_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"round": self.round, "incumbent": eval_to_dict(self.incumbent),
                "arms": [arm_to_dict(a) for a in self.arms], "s_star": self.s_star, "delta": self.delta,
                "hp": self.hp.to_dict(), "history": [r.to_dict() for r in self.history],
                "cases": None if self.cases is None else list(self.cases), "incumbent_commit": self.incumbent_commit}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SelectionInputs:
        return cls(round=int(d["round"]), incumbent=eval_from_dict(d["incumbent"]),
                   arms=tuple(arm_from_dict(a) for a in d["arms"]), s_star=float(d["s_star"]),
                   delta=float(d["delta"]), hp=Hyperparams.from_dict(d["hp"]),
                   history=tuple(HistoryRecord.from_dict(r) for r in d.get("history", ())),
                   cases=None if d.get("cases") is None else tuple(d["cases"]),
                   incumbent_commit=d.get("incumbent_commit"))


@dataclass(frozen=True)
class ArmTrace:
    arm: str
    evaluated: bool
    components: tuple[str, ...] = ()
    score: float | None = None
    cost: float | None = None
    delta_s: float | None = None
    delta_c: float | None = None
    novelty: int = 0
    missing_rate: float | None = None
    critical: int | None = None
    branch: Literal["cost", "weighted"] | None = None
    rule: Mapping[str, Any] = field(default_factory=dict)
    ci: Mapping[str, Any] = field(default_factory=dict)
    floor: Mapping[str, Any] = field(default_factory=dict)
    guards: Mapping[str, Any] = field(default_factory=dict)
    admissible: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"arm": self.arm, "evaluated": self.evaluated, "components": list(self.components),
                          "score": self.score, "cost": self.cost, "delta_s": self.delta_s, "delta_c": self.delta_c,
                          "novelty": self.novelty, "missing_rate": self.missing_rate, "critical": self.critical,
                          "branch": self.branch, "rule": dict(self.rule), "ci": dict(self.ci),
                          "floor": dict(self.floor), "guards": dict(self.guards), "admissible": self.admissible,
                          "reasons": list(self.reasons)})


@dataclass(frozen=True)
class SelectionDecision:
    round: int
    decision: Decision
    winner: str | None
    delta: float
    s_star_before: float
    s_star_after: float
    score_next: float | None          # S_{t+1} (winner S' or contemporaneous S_t)
    incumbent: Mapping[str, Any]
    arms: tuple[ArmTrace, ...]
    reasons: tuple[str, ...] = ()     # round-level (rerun) reasons
    params: Mapping[str, Any] = field(default_factory=dict)
    n_keys: int = 0

    @property
    def winner_trace(self) -> ArmTrace | None:
        return next((a for a in self.arms if a.arm == self.winner), None)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"round": self.round, "decision": self.decision, "winner": self.winner, "delta": self.delta,
                          "s_star_before": self.s_star_before, "s_star_after": self.s_star_after,
                          "score_next": self.score_next, "incumbent": dict(self.incumbent),
                          "arms": [a.to_dict() for a in self.arms], "reasons": list(self.reasons),
                          "params": dict(self.params), "n_keys": self.n_keys})


def _jsonable(x: Any) -> Any:
    if isinstance(x, float):
        return finite(x)
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _params(hp: Hyperparams) -> dict[str, Any]:
    return {"beta0": hp.beta0, "beta1": hp.beta1, "w_s": hp.w_s, "w_c": hp.w_c, "w_n": hp.w_n,
            "missing_invalid_frac": hp.missing_invalid_frac, "require_ci_lower": hp.require_ci_lower,
            "ci_lower_threshold": hp.ci_lower_threshold, "ci_level": hp.ci_level, "n_bootstrap": hp.n_bootstrap,
            "seed": hp.seed, "k": hp.k}


def _quality_failures(inp: SelectionInputs, keys: Sequence[stats.Key], inc_missing: float) -> list[str]:
    out = []
    if not keys:
        out.append("no_trials")
    if inc_missing > inp.hp.missing_invalid_frac:
        out.append(f"baseline_invalid: incumbent missing-trial rate {inc_missing:.3f} > "
                   f"{inp.hp.missing_invalid_frac:.3f}")
    for a in inp.arms:
        if a.status == "evaluated" and a.eval is not None:
            if a.eval.pin != inp.incumbent.pin:
                out.append(f"evaluator_pin_mismatch: arm {a.arm}")
            if a.eval.split != inp.incumbent.split:
                out.append(f"split_mismatch: arm {a.arm} ({a.eval.split} != {inp.incumbent.split})")
    return out


def select(inp: SelectionInputs, *, guard: DomainGuard | None = None) -> SelectionDecision:
    hp, delta = inp.hp, float(inp.delta)
    names = [a.arm for a in inp.arms]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names")
    arms = sorted(inp.arms, key=lambda a: a.arm)
    evaluated = [a for a in arms if a.status == "evaluated" and a.eval is not None]
    done = {a.arm for a in evaluated}
    keys = stats.universe([inp.incumbent, *(a.eval for a in evaluated)], cases=inp.cases,
                          k=hp.k if inp.cases is not None else None)
    inc_idx = stats.index_scores(inp.incumbent.scores)
    s_t = stats.task_score(inc_idx, keys)
    c_t = stats.mean_cost(inc_idx, keys)
    inc_missing = stats.missing_rate(inc_idx, keys)
    crit_t = stats.critical_count(inc_idx, keys)
    incumbent = {"harness_tree": inp.incumbent.harness_tree, "commit": inp.incumbent_commit, "score": s_t,
                 "cost": c_t, "missing_rate": inc_missing, "critical": crit_t}
    common = dict(round=inp.round, delta=delta, s_star_before=inp.s_star, incumbent=incumbent,
                  params=_params(hp), n_keys=len(keys))

    failures = _quality_failures(inp, keys, inc_missing)
    if failures:
        traces = tuple(ArmTrace(arm=a.arm, evaluated=a.arm in done, reasons=("round_rerun",)) for a in arms)
        return SelectionDecision(decision="rerun", winner=None, s_star_after=inp.s_star, score_next=None,
                                 arms=traces, reasons=tuple(failures), **common)

    counts = accepted_counts(inp.history)
    inc_means = stats.case_means(inc_idx, keys)
    traces = []
    for a in arms:
        if a.arm not in done:
            traces.append(ArmTrace(arm=a.arm, evaluated=False, components=tuple(e.component for e in a.edits),
                                   reasons=(f"not_evaluated: status={a.status}",)))
            continue
        traces.append(_judge_arm(a, inp, keys, inc_idx, inc_means, s_t, c_t, crit_t, counts, guard))

    admissible = [t for t in traces if t.admissible]
    win = max(admissible, key=lambda t: (t.score, -t.cost, t.ci.get("lower", -math.inf), _rev(t.arm)), default=None)
    score_next = win.score if win is not None else s_t
    return SelectionDecision(decision="ship" if win else "do_not_ship", winner=win.arm if win else None,
                             s_star_after=max(inp.s_star, score_next), score_next=score_next, arms=tuple(traces),
                             **common)


def _rev(name: str) -> tuple[int, ...]:
    """Key so that max() prefers the lexicographically *smallest* arm name on full ties."""
    return tuple(-ord(ch) for ch in name) + (1,)


def _judge_arm(a: ArmResult, inp: SelectionInputs, keys: Sequence[stats.Key], inc_idx: Mapping, inc_means: Mapping,
               s_t: float, c_t: float, crit_t: int, counts: Mapping[str, int], guard: DomainGuard | None) -> ArmTrace:
    hp, delta = inp.hp, float(inp.delta)
    assert a.eval is not None
    idx = stats.index_scores(a.eval.scores)
    s_new, c_new = stats.task_score(idx, keys), stats.mean_cost(idx, keys)
    d_s = s_new - s_t
    d_c = stats.relative_cost(c_new, c_t)
    nu = novelty(a.edits, counts)
    crit = stats.critical_count(idx, keys)
    boot = stats.paired_bootstrap(stats.case_means(idx, keys), inc_means, n_resamples=hp.n_bootstrap,
                                  level=hp.ci_level, seed=stats.derive_seed(hp.seed, inp.round, a.arm))
    reasons: list[str] = []

    ci_threshold = 0.0 if hp.ci_lower_threshold == "zero" else delta
    ci = {"lower": boot.lower, "upper": boot.upper, "mean": boot.mean, "level": boot.level,
          "n_cases": boot.n_cases, "n_resamples": boot.n_resamples, "threshold": ci_threshold,
          "required": False, "passed": None}
    if not math.isfinite(d_c):
        rule = {"name": "cost" if d_s > delta else "weighted", "passed": False, "note": "incumbent cost is 0"}
        branch = rule["name"]
        reasons.append("cost_baseline_zero")
    elif d_s > delta:
        branch = "cost"
        allowance = hp.beta0 + hp.beta1 * d_s
        cost_ok = d_c <= allowance
        rule = {"name": "cost", "allowance": allowance, "delta_c": d_c, "passed": cost_ok}
        if not cost_ok:
            reasons.append(f"cost_rule: dC {d_c:.4f} > {allowance:.4f}")
        if hp.require_ci_lower:
            ci_ok = boot.lower > ci_threshold
            ci.update(required=True, passed=ci_ok)
            if not ci_ok:
                reasons.append(f"ci_lower: {boot.lower:.4f} <= {ci_threshold:.4f}")
            rule["passed"] = cost_ok and ci_ok
    else:
        branch = "weighted"
        value = hp.w_s * d_s - hp.w_c * d_c + hp.w_n * nu
        rule = {"name": "weighted", "value": value, "terms": {"w_s*dS": hp.w_s * d_s, "w_c*dC": hp.w_c * d_c,
                                                               "w_n*nu": hp.w_n * nu}, "passed": value > 0}
        if not value > 0:
            reasons.append(f"weighted_rule: {value:.4f} <= 0")

    floor_thr = inp.s_star - delta
    floor = {"threshold": floor_thr, "s_star": inp.s_star, "passed": s_new >= floor_thr}
    if not floor["passed"]:
        reasons.append(f"floor: S' {s_new:.4f} < S*-delta {floor_thr:.4f}")

    guards: dict[str, Any] = {"critical_safety": {"incumbent": crit_t, "arm": crit, "passed": crit <= crit_t}}
    if crit > crit_t:
        reasons.append(f"guard: critical safety violations increased {crit_t} -> {crit}")
    if a.critic is not None and not a.critic.passed:
        guards["critic"] = {"passed": False, "reasons": list(a.critic.reasons)}
        reasons.append("guard: critic rejected")
    if inp.incumbent_commit is not None:
        ok = a.base_commit == inp.incumbent_commit
        guards["base_commit"] = {"expected": inp.incumbent_commit, "actual": a.base_commit, "passed": ok}
        if not ok:
            reasons.append("guard: arm not based on the incumbent commit")
    if guard is not None:
        dom = list(guard(a, a.eval))
        guards["domain"] = {"passed": not dom, "reasons": dom}
        reasons.extend(f"guard: domain: {r}" for r in dom)
    guards_ok = all(g.get("passed", True) for g in guards.values())

    return ArmTrace(arm=a.arm, evaluated=True, components=tuple(e.component for e in a.edits), score=s_new,
                    cost=c_new, delta_s=d_s, delta_c=d_c, novelty=nu, missing_rate=stats.missing_rate(idx, keys),
                    critical=crit, branch=branch, rule=rule, ci=ci, floor=floor, guards=guards,
                    admissible=bool(rule["passed"]) and floor["passed"] and guards_ok, reasons=tuple(reasons))
