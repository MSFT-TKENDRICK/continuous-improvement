"""Trivial stand-ins for RRSI (M7) and OES (M4) used until integration wires the
real modules. Deterministic and pure; good enough for wiring tests and the
``fake`` profile, NOT a faithful Alg. 2 (no cost rule / novelty / bootstrap)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ci_lab.campaign.records import cases, critical_count, mean_score, tokens
from ci_lab.contracts import TEXT_COMPONENTS, ArmResult, EvalResult

DEFAULT_HYPER: dict[str, Any] = {
    "arms": 2,                # N arms per round
    "k": 1,                   # trials per case
    "max_rounds": 8,          # T
    "aa_repeats": 5,          # A/A repeats R (C8 default/minimum 5)
    "max_parallel_arms": 2,   # bounded arm concurrency (asyncio)
    "seed": 0,                # interleaving order of incumbent vs arms
    "budget": 1,              # edits per arm b_t (fixed here; M7 anneals)
    "budget_tokens": None,    # campaign token budget (None = unlimited)
    "max_arm_attempts": 2,    # arm workflow failures before the arm is marked failed
    "holdout_looks": 1,       # planned held-out looks L (global, per dataset hash)
    "draft_prs": True,
    "strategies": ["agent"],  # arm strategies rotated by the trivial schedule (M7 allocates)
    "arm_budget_tokens": None,  # ArmContext.budget_tokens for optimizer strategies
    "heartbeat_s": 30.0,      # status marker heartbeat while long steps run (<= 60, C36)
    "guard_trials": None,     # paired guard-off/on repetitions (None: >= 3 trials/case if stochastic, B4)
    "guard_stochastic": None,  # None: stochastic iff the Copilot profile
    "guard_margin": 0.0,      # B1 task-completion non-inferiority margin (C5)
}


def schedule(round_no: int, hyper: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    n = int(hyper.get("arms", 2))
    strategies = list(hyper.get("strategies") or ["agent"])
    out = []
    for i in range(n):
        strategy = strategies[i % len(strategies)]
        component = "guard" if strategy == "guard" else TEXT_COMPONENTS[(round_no + i) % len(TEXT_COMPONENTS)]
        out.append({"arm": f"v{i + 1}", "component": component, "strategy": strategy,
                    "budget": int(hyper.get("budget", 1))})
    return out


def select(incumbent: EvalResult, arms: Mapping[str, ArmResult], delta: float,
           hyper: Mapping[str, Any]) -> dict[str, Any]:
    s_inc = mean_score(incumbent)
    crit_inc = critical_count(incumbent)
    trace: list[dict[str, Any]] = []
    best: tuple[float, str] | None = None
    for name in sorted(arms):
        arm = arms[name]
        if arm.status != "evaluated" or arm.eval is None:
            trace.append({"arm": name, "admissible": False, "reason": f"status={arm.status}"})
            continue
        d_s = mean_score(arm.eval) - s_inc
        reason = None
        if critical_count(arm.eval) > crit_inc:
            reason = "safety_regression"
        elif d_s <= delta:
            reason = "delta_s_within_noise"
        trace.append({"arm": name, "admissible": reason is None, "delta_s": round(d_s, 6),
                      "delta_c": tokens(arm.eval) - tokens(incumbent), "reason": reason})
        if reason is None and (best is None or d_s > best[0]):
            best = (d_s, name)
    return {"decision": "ship" if best else "do_not_ship", "winner": best[1] if best else None,
            "incumbent_score": s_inc, "delta": delta, "trace": trace}


def calibrate_delta(results: Sequence[EvalResult], hyper: Mapping[str, Any]) -> float:
    means = [mean_score(r) for r in results]
    spread = max((abs(a - b) for i, a in enumerate(means) for b in means[i + 1:]), default=0.0)
    floor = 1.0 / max(1, min((cases(r) for r in results), default=1))
    return max(spread, floor)


def confirm_test(h0: EvalResult, final: EvalResult, hyper: Mapping[str, Any]) -> dict[str, Any]:
    d_s = mean_score(final) - mean_score(h0)
    safe = critical_count(final) <= critical_count(h0)
    ship = d_s > 0 and safe
    return {"decision": "ship" if ship else "do_not_ship", "delta_s": d_s, "safety_non_inferior": safe}


def build_envelope(kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Minimal OES-0.1.0-shaped envelope; M4 replaces this with validated builders."""
    design = {"round": "abn", "calibration": "ab", "confirm": "ab"}[kind]
    return {
        "oesVersion": "0.1.0",
        "objectType": "experiment",
        "id": record["eid"],
        "status": "decided",
        "design": {"type": design, "multipleTestingPolicy":
                   "exploratory-rrsi-selection" if kind == "round" else "preregistered"},
        "decision": record.get("decision"),
        "extensions": {"org.ci.rrsi": {k: v for k, v in record.items() if k != "eid"}},
    }
