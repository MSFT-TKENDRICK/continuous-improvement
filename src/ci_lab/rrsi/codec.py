"""JSON codecs for the shared contract dataclasses used by RRSI records."""

from __future__ import annotations

import math
from typing import Any

from ci_lab.contracts import ArmResult, CriticVerdict, Edit, EvalResult, EvaluatorPin, TaskScore, Violation


def finite(x: float | None) -> float | None:
    """JSON-safe float: non-finite values become ``None``."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def violation_to_dict(v: Violation) -> dict[str, Any]:
    return {"rule_id": v.rule_id, "severity": v.severity, "detail": v.detail}


def violation_from_dict(d: dict[str, Any]) -> Violation:
    return Violation(rule_id=d["rule_id"], severity=d["severity"], detail=d.get("detail", ""))


def edit_to_dict(e: Edit) -> dict[str, Any]:
    return {"component": e.component, "hypothesis": e.hypothesis, "files": list(e.files), "commit": e.commit}


def edit_from_dict(d: dict[str, Any]) -> Edit:
    return Edit(component=d["component"], hypothesis=d.get("hypothesis", ""), files=tuple(d.get("files", ())),
                commit=d.get("commit", ""))


def task_score_to_dict(s: TaskScore) -> dict[str, Any]:
    return {"case_id": s.case_id, "trial": s.trial, "suite": s.suite, "score": s.score,
            "violations": [violation_to_dict(v) for v in s.violations], "tokens_in": s.tokens_in,
            "tokens_out": s.tokens_out, "served_model": s.served_model}


def task_score_from_dict(d: dict[str, Any]) -> TaskScore:
    return TaskScore(case_id=d["case_id"], trial=int(d.get("trial", 0)), suite=d.get("suite", ""),
                     score=d.get("score"), violations=tuple(violation_from_dict(v) for v in d.get("violations", ())),
                     tokens_in=int(d.get("tokens_in", 0)), tokens_out=int(d.get("tokens_out", 0)),
                     served_model=d.get("served_model"))


def pin_to_dict(p: EvaluatorPin) -> dict[str, Any]:
    return {"evaluator_tree": p.evaluator_tree, "judge_model": p.judge_model, "judge_provider": p.judge_provider,
            "served_judge_models": list(p.served_judge_models)}


def pin_from_dict(d: dict[str, Any]) -> EvaluatorPin:
    return EvaluatorPin(evaluator_tree=d["evaluator_tree"], judge_model=d["judge_model"],
                        judge_provider=d["judge_provider"], served_judge_models=tuple(d.get("served_judge_models", ())))


def eval_to_dict(r: EvalResult) -> dict[str, Any]:
    return {"harness_tree": r.harness_tree, "split": r.split, "pin": pin_to_dict(r.pin),
            "scores": [task_score_to_dict(s) for s in r.scores]}


def eval_from_dict(d: dict[str, Any]) -> EvalResult:
    return EvalResult(harness_tree=d["harness_tree"], split=d["split"], pin=pin_from_dict(d["pin"]),
                      scores=[task_score_from_dict(s) for s in d.get("scores", ())])


def arm_to_dict(a: ArmResult) -> dict[str, Any]:
    critic = None if a.critic is None else {"passed": a.critic.passed, "reasons": list(a.critic.reasons),
                                            "repairs": a.critic.repairs}
    return {"arm": a.arm, "base_commit": a.base_commit, "head_commit": a.head_commit, "harness_tree": a.harness_tree,
            "edits": [edit_to_dict(e) for e in a.edits], "critic": critic,
            "eval": None if a.eval is None else eval_to_dict(a.eval), "status": a.status}


def arm_from_dict(d: dict[str, Any]) -> ArmResult:
    c = d.get("critic")
    return ArmResult(arm=d["arm"], base_commit=d["base_commit"], head_commit=d.get("head_commit"),
                     harness_tree=d.get("harness_tree"), edits=[edit_from_dict(e) for e in d.get("edits", ())],
                     critic=None if c is None else CriticVerdict(passed=c["passed"], reasons=list(c.get("reasons", ())),
                                                                 repairs=int(c.get("repairs", 0))),
                     eval=None if d.get("eval") is None else eval_from_dict(d["eval"]),
                     status=d.get("status", "pending"))
