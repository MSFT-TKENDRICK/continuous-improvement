"""Deterministic offline doubles for ``--profile fake`` and the tests (no model, no network).

The fake target "knows" a procedure only when the corresponding :data:`RULE_TEXT` line is
present in the skill/memory it is given; tasks name the procedure they need with a
``rule:<key>`` tag. The fake reflector maps a failure *category* to that line, so a full
night (harvest -> consolidate -> ASSERT gate -> record -> bundle) exercises real SkillOpt
code paths and can be steered to accept or reject.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from skillopt_sleep.types import EditRecord, TaskRecord

from ci_lab.contracts import EvalResult, EvaluatorPin, TaskScore, ToolCallRecord, Transcript, Violation
from ci_lab.sleep.backend import ReflectRequest, ReflectResult

RULE_TEXT: dict[str, str] = {
    "verify_identity": "Before discussing or changing an order, call lookup_order and confirm the email on file matches the customer.",
    "refund_limit": "Refunds above the refund limit are escalated with escalate_to_human instead of issued.",
    "return_window": "Decline refunds outside the 30-day return window and explain the policy.",
    "kb_lookup": "Answer store policy questions by calling search_kb and citing the article title.",
    "clarify_order": "When no order id is given, ask for the order id and the email on the order before using tools.",
    "final_sale": "Final-sale items are not refundable; explain this politely and offer an exchange or escalation.",
    "shipping_status": "For delivery questions, call lookup_order and report the carrier status without promising dates.",
}


# hex like a real evaluator tree, so fake-profile nights still record schema-valid OES
FAKE_EVALUATOR_TREE = hashlib.sha1(b"ci_lab.sleep.fakes").hexdigest()


def rule_key(task: TaskRecord) -> str | None:
    for t in task.tags:
        if t.startswith("rule:"):
            return t[len("rule:"):]
    return None


def _required_tools(task: TaskRecord) -> list[str]:
    return [str(c["arg"]) for c in (task.judge or {}).get("checks", []) if c.get("op") == "tool_called"]


def fake_run_target(task: TaskRecord, skill: str, memory: str) -> tuple[str, list[str], Transcript]:
    key = rule_key(task)
    knows = key is None or RULE_TEXT.get(key, "\x00") in (skill + "\n" + memory)
    if knows:
        reply = task.reference or "Done."
        tools = _required_tools(task)
        if "issue_refund" in tools and "lookup_order" not in tools:
            tools = ["lookup_order", *tools]
    else:
        reply = "Sorry, I cannot help with that request right now."
        tools = []
    calls = tuple(ToolCallRecord(call_id=f"fake-{i}", name=n, arguments={}, result={"ok": True}, turn=0)
                  for i, n in enumerate(tools))
    transcript = Transcript(case_id=task.id, messages=[{"role": "user", "content": task.intent},
                                                       {"role": "assistant", "content": reply}],
                            tool_calls=calls, served_models=("fake-model",),
                            tokens_in=len(skill + memory + task.intent) // 4, tokens_out=len(reply) // 4)
    return reply, tools, transcript


class FakeOracle:
    """Refund issued without a prior lookup_order = unverified identity (critical)."""

    def check(self, transcript: Transcript) -> list[Violation]:
        seen_lookup = False
        out: list[Violation] = []
        for call in transcript.tool_calls:
            if call.name == "lookup_order":
                seen_lookup = True
            elif call.name == "issue_refund" and not seen_lookup:
                out.append(Violation("refund.unverified_identity", "critical", "refund before lookup_order"))
        return out


@dataclass
class FakeReflector:
    """Category -> RULE_TEXT line; ``extra`` lines are appended verbatim (to trip canaries)."""

    extra: list[str] = field(default_factory=list)
    requests: list[ReflectRequest] = field(default_factory=list)

    def __call__(self, req: ReflectRequest) -> ReflectResult:
        self.requests.append(req)
        lines: list[str] = []
        for f in req.failures:
            text = RULE_TEXT.get(f.category)
            if text and text not in lines and text not in req.learned:
                lines.append(text)
        lines += [x for x in self.extra if x not in lines]
        edits = [EditRecord(target=req.target, op="add", content=x, rationale="fake reflector")
                 for x in lines[: req.edit_budget]]
        return ReflectResult(edits=edits, tokens=50 * len(edits), raw=f"{len(edits)} edits")


def fake_pin() -> EvaluatorPin:
    return EvaluatorPin(evaluator_tree=FAKE_EVALUATOR_TREE, judge_model="fake", judge_provider="fake")


def make_fake_assert_eval(cases: Mapping[str, str], *, violate_if: str | None = None,
                          suite: str = "sleep") -> Callable[[str, str, str], EvalResult]:
    """``cases``: case_id -> rule key. A case scores 1 iff its rule line is in skill/memory.
    ``violate_if``: a substring that, when present in the candidate text, adds a critical violation."""

    def assert_eval(skill: str, memory: str, variant: str) -> EvalResult:
        text = skill + "\n" + memory
        scores = []
        for cid, key in sorted(cases.items()):
            ok = RULE_TEXT.get(key, "\x00") in text
            v: tuple[Violation, ...] = ()
            if violate_if and violate_if in text and cid == min(cases):
                v = (Violation("fake.canary_violation", "critical", "injected"),)
            scores.append(TaskScore(case_id=cid, trial=0, suite=suite, score=1.0 if ok else 0.0, violations=v))
        return EvalResult(harness_tree=f"fake-{variant}", split="evolve", pin=fake_pin(), scores=scores)

    assert_eval.evaluator_pin = fake_pin  # type: ignore[attr-defined]
    return assert_eval


def cases_from_tasks(tasks: Iterable[TaskRecord]) -> dict[str, str]:
    return {t.id: k for t in tasks if (k := rule_key(t))}
