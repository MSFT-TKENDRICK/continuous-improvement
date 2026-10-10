"""Builders shared by the RRSI tests (exposed through the ``mk`` fixture)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest

from ci_lab.contracts import (
    ArmResult,
    Edit,
    EvalResult,
    EvaluatorPin,
    TaskScore,
    Violation,
)
from ci_lab.rrsi.history import HistoryRecord
from ci_lab.rrsi.params import profile

PIN = EvaluatorPin(evaluator_tree="evaltree", judge_model="judge", judge_provider="prov")
CRIT = Violation(rule_id="change.unverified_identity", severity="critical", detail="x")
MAJOR = Violation(rule_id="pii.early", severity="major", detail="y")


def ev(scores: Mapping[str, Sequence[float | None]] | Sequence[float | None], *, tokens: int | Sequence[int] = 100,
       crit: Mapping[str, int] | None = None, tree: str = "tree", pin: EvaluatorPin = PIN,
       split: str = "evolve") -> EvalResult:
    """``scores``: case -> per-trial scores, or a flat list (cases c0..cN, one trial each)."""
    if not isinstance(scores, Mapping):
        scores = {f"c{i}": [s] for i, s in enumerate(scores)}
    out, n = [], 0
    for case, trials in scores.items():
        for t, s in enumerate(trials):
            tok = tokens if isinstance(tokens, int) else tokens[n]
            viol = (CRIT,) * (crit or {}).get(case, 0) if t == 0 else ()
            out.append(TaskScore(case_id=case, trial=t, suite="s", score=s, violations=viol + (MAJOR,),
                                 tokens_in=tok // 2, tokens_out=tok - tok // 2))
            n += 1
    return EvalResult(harness_tree=tree, split=split, pin=pin, scores=out)  # type: ignore[arg-type]


def edit(component: str = "prompt", hyp: str = "h") -> Edit:
    return Edit(component=component, hypothesis=hyp, files=(f"{component}.md",), commit=f"c-{component}")


def arm(name: str, result: EvalResult | None, *, components: Sequence[str] = ("prompt",), base: str = "inc",
        status: str = "evaluated", strategy: str = "agent") -> ArmResult:
    return ArmResult(arm=name, base_commit=base, head_commit=f"head-{name}", harness_tree=f"tree-{name}",
                     edits=[edit(c) for c in components], eval=result, status=status,  # type: ignore[arg-type]
                     strategy=strategy)


def rec(round_no: int, arm_name: str, components: Sequence[str], delta_s: float | None,
        accepted: bool = False, strategy: str = "agent") -> HistoryRecord:
    return HistoryRecord(round=round_no, arm=arm_name, edits=tuple(edit(c) for c in components), score=0.5, cost=100.0,
                         delta_s=delta_s, delta_c=0.0, accepted=accepted, strategy=strategy)


def hp(**kw):
    """Smoke profile with dyadic, hand-computable cost knobs (beta0=0.25, beta1=1)."""
    base = {"k": 1, "n_bootstrap": 400, "w_s": 0.0, "w_c": 15.0, "w_n": 0.5,
            "beta0": 0.25, "beta1": 1.0}
    base.update(kw)
    return profile("smoke", **base)


@pytest.fixture
def mk() -> SimpleNamespace:
    return SimpleNamespace(ev=ev, edit=edit, arm=arm, rec=rec, hp=hp, PIN=PIN, CRIT=CRIT)
