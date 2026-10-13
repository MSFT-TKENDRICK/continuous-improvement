import asyncio
import json

from ci_lab.contracts import FailureRecord
from ci_lab.optim.lm import make_lm
from ci_lab.optim.scoring import CaseOutcome, DomainEvolveScorer
from ci_lab.optim.skillopt import (
    SkillOptConfig,
    calls_needed,
    fit_cases,
    optimize_skill,
)

SKILL = "harness/skills/harness-editing/SKILL.md"
MEMORY = "harness/skills/harness-editing/memory.md"
KW = {"c1": "refund", "c2": "tracking", "c3": "escalate", "c4": "polite", "c5": "verify", "c6": "apologise"}
GOOD_RULE = "Always: " + ", ".join(KW.values()) + "."


def test_skillopt_defaults_to_harness_project():
    assert SkillOptConfig().project == "harness"


class SkillScorer:
    def __init__(self, key=SKILL):
        self.key, self.requests = key, []

    def __call__(self, cand, cases):
        self.requests.append(list(cases))
        text = cand[self.key].lower()
        return [CaseOutcome(c, 1.0 if KW[c] in text else 0.0,
                            None if KW[c] in text else FailureRecord(c, "s", "missing", ("R-" + c,), {}),
                            tokens=7) for c in cases]

    @property
    def calls(self):
        return sum(map(len, self.requests))


class SpyLM:
    def __init__(self, lm):
        self.inner, self.prompts, self.history = lm, [], []

    def __call__(self, prompt=None, **kw):
        self.prompts.append(prompt)
        return self.inner(prompt, **kw)


def edits_lm(content, n=6):
    arr = json.dumps([{"op": "add", "content": content, "rationale": "fix failures"}])
    return SpyLM(make_lm("fake", "reflector", fake_answers=[{"text": arr}] * n))


def test_calls_needed_and_fit():
    assert calls_needed(4, 2, False) == 10
    assert calls_needed(4, 2, True) == 16
    tr, va = fit_cases(list(KW), 10, val_fraction=0.34, memory=False, salt="s")
    assert calls_needed(len(tr), len(va), False) <= 10 and len(tr) + len(va) == 6
    tr, va = fit_cases(list(KW), 7, val_fraction=0.34, memory=False, salt="s")
    assert calls_needed(len(tr), len(va), False) <= 7 and len(tr) + len(va) < 6
    assert fit_cases(list(KW), 3, val_fraction=0.34, memory=False, salt="s") == ([], [])


def test_accepts_improving_rule_and_reports_cost():
    scorer, lm = SkillScorer(), edits_lm(GOOD_RULE)
    res = asyncio.run(optimize_skill((SKILL, "# Skill\nBe helpful.\n"), scorer, list(KW), reflection_lm=lm,
                                     edit_budget=1, config=SkillOptConfig(max_metric_calls=40)))
    assert res.diagnostics["accepted"] is True
    assert GOOD_RULE in res.changed[SKILL]
    assert res.best_score > res.seed_score
    assert res.cost.metric_calls == scorer.calls <= 40
    assert res.cost.reflection_calls == 1 and res.cost.reflection_tokens > 0
    prompt = lm.prompts[0]
    assert "why-wrong: score=0.000; suite=s; category=missing; violated_rules=R-" in prompt
    assert "at most 1 bounded edits" in prompt


def test_gate_rejects_useless_rule():
    scorer = SkillScorer()
    res = asyncio.run(optimize_skill((SKILL, "Be helpful."), scorer, list(KW), reflection_lm=edits_lm("Be nice."),
                                     config=SkillOptConfig(max_metric_calls=40)))
    assert res.changed == {} and res.diagnostics["accepted"] is False
    assert res.note.startswith("gate")


def test_budget_too_small_is_noop():
    scorer = SkillScorer()
    res = asyncio.run(optimize_skill((SKILL, "x"), scorer, list(KW), reflection_lm=edits_lm(GOOD_RULE),
                                     budget_tokens=4000 * 3))
    assert scorer.calls == 0 and res.changed == {} and "too small" in res.note


def test_memory_and_evolve_only_domain(worktree, domain, tmp_path):
    scorer = DomainEvolveScorer(domain, worktree, tmp_path / "scratch", experiment_id="e", variant="a")
    lm = edits_lm("Offer a refund, share tracking, escalate when stuck, stay polite.")
    skill = (SKILL, (worktree / SKILL).read_text())
    memory = (MEMORY, (worktree / MEMORY).read_text())

    async def go():
        return await optimize_skill(skill, scorer, scorer.evolve_cases(), reflection_lm=lm, memory=memory,
                                    edit_budget=2, config=SkillOptConfig(max_metric_calls=60))

    res = asyncio.run(go())
    assert set(domain.splits_called) == {"evolve"}
    assert res.diagnostics["accepted"] is True and SKILL in res.changed
    assert res.cost.scorer_tokens == scorer.tokens_spent
