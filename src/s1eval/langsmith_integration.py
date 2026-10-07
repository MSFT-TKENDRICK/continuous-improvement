"""LangSmith integration (opt-in; nothing is uploaded unless you call langsmith yourself).

Two paths, matching the LangSmith docs:

1. **UI decision-model evaluator** (online/offline, no code): LangSmith can only create these in
   the UI. ``s1eval export-langsmith`` prints the TypeSafe questions JSON for the evaluator's
   "Advanced" editor. Point it at TypeSafe (Jev), the LangSmith gateway (semif-qwen3.5-4b), or a
   self-hosted TypeSafe-compatible base URL (LangSmith appends /v1/systemone). LangSmith cloud
   cannot reach ``127.0.0.1``; a local ``s1eval serve`` needs a tunnel you control + a token.

2. **SDK evaluator** (``make_evaluator``) for ``langsmith.evaluate(...)``: returns
   ``{"results": [...]}`` with one feedback key per question plus ``pass`` / ``needs_review``.
   Feedback mapping mirrors LangSmith's decision evaluators: noul -> score P(true),
   choice -> value, score -> expected score. ``reference_outputs`` are deliberately not
   accepted, so gold labels can never leak into judge state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .rubric import Rubric
from .state import project_observable

ToObservable = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def default_to_observable(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    if isinstance(inputs.get("agent_policy"), str):
        obs["agent_policy"] = inputs["agent_policy"]
    if isinstance(inputs.get("conversation"), list):
        obs["conversation"] = inputs["conversation"]
    elif isinstance(inputs.get("question"), str):
        obs["conversation"] = [{"role": "user", "content": inputs["question"]}]
    if isinstance(outputs.get("tool_calls"), list):
        obs["tool_calls"] = [{k: tc.get(k) for k in ("name", "arguments", "result")} for tc in outputs["tool_calls"]]
    obs["final_response"] = str(outputs.get("final_response", outputs.get("output", "")))
    return obs


def make_evaluator(backend, rubric: Rubric, to_observable: ToObservable = default_to_observable):
    def s1eval_judge(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        state = project_observable({"observable": to_observable(inputs, outputs)})
        d = backend.decide(state, rubric.questions)
        results: list[dict[str, Any]] = []
        for k, a in d.answers.items():
            if not a.ok:
                results.append({"key": k, "score": None, "comment": f"judge {a.status}: {a.diagnostics.get('reason', '')}"})
            elif a.type == "noul":
                results.append({"key": k, "score": a.noul})
            elif a.type == "choice":
                results.append({"key": k, "value": a.choice, "comment": f"confidence={a.confidence:.2f}"})
            else:
                results.append({"key": k, "score": a.score, "comment": f"level={a.level} confidence={a.confidence:.2f}"})
        comp = rubric.composite(d.answers)
        results.append({
            "key": "pass",
            "score": None if comp["pass"] is None else int(comp["pass"]),
            "comment": "; ".join(comp["reasons"]) or ("failed: " + ", ".join(comp["failed"]) if comp["failed"] else "ok"),
        })
        results.append({"key": "needs_review", "score": int(comp["needs_review"])})
        return {"results": results}

    return s1eval_judge


def ui_setup_text(rubric: Rubric) -> str:
    return (
        "LangSmith > Evaluators > + Evaluator > Decision model\n"
        "  Endpoint: TypeSafe-compatible (base URL WITHOUT /v1/systemone), e.g.\n"
        "    https://api.typesafe.ai                (model jev-latest, TypeSafe key)\n"
        "    LangSmith gateway decision model       (semif-qwen3.5-4b / typesafe/jev-1.13.0)\n"
        "  State: map only observable run fields (inputs.conversation, outputs.tool_calls,\n"
        "         outputs.final_response). Do NOT map reference outputs / expected behaviour.\n"
        "  Questions > Advanced: paste the JSON below.\n"
        "  The composite pass rule is not expressible in the UI; compute it from feedback or use\n"
        "  the SDK evaluator (s1eval.langsmith_integration.make_evaluator).\n\n"
        + rubric.to_langsmith_questions()
    )
