import asyncio

import pytest

from ci_lab.contracts import FailureRecord
from ci_lab.optim.gepa import (
    ComponentAdapter,
    GepaConfig,
    metric_cap,
    optimize_texts,
)
from ci_lab.optim.lm import make_lm
from ci_lab.optim.scoring import CaseOutcome, DomainEvolveScorer, EvolveGuard, MetricBudget

PROMPT = "src/order_support/harness/prompts/system.md"
KW = {"c1": "refund", "c2": "tracking", "c3": "escalate", "c4": "polite", "c5": "verify", "c6": "apologise"}


class RecordingScorer:
    def __init__(self):
        self.requests: list[list[str]] = []

    def __call__(self, cand, cases):
        self.requests.append(list(cases))
        text = " ".join(cand.values()).lower()
        return [CaseOutcome(c, 1.0 if KW[c] in text else 0.0,
                            None if KW[c] in text else FailureRecord(c, "s", "missing", ("R-" + c,), {"h": 0.0},
                                                                     excerpt=f"no {KW[c]}"),
                            tokens=11) for c in cases]

    @property
    def calls(self):
        return sum(len(r) for r in self.requests)


class SpyLM:
    def __init__(self, lm):
        self.inner, self.prompts, self.history = lm, [], []

    def __call__(self, prompt=None, **kw):
        self.prompts.append(prompt)
        return self.inner(prompt, **kw)


def fake_lm(text, n=20):
    return make_lm("fake", "reflector", fake_answers=[{"new_instruction": f"```\n{text}\n```"}] * n)


def run(coro):
    return asyncio.run(coro)


def test_metric_cap():
    assert metric_cap(60, None, 4000) == 60
    assert metric_cap(60, 40_000, 4000) == 10
    assert metric_cap(60, -5, 4000) == 0


def test_gepa_improves_and_reports_cost():
    scorer = RecordingScorer()
    spy = SpyLM(fake_lm(" ".join(KW.values())))
    res = run(optimize_texts({PROMPT: "You help."}, scorer, list(KW), reflection_lm=spy,
                             config=GepaConfig(max_metric_calls=30)))
    assert res.changed == {PROMPT: " ".join(KW.values())}
    assert res.best_score == 1.0 and res.seed_score == 0.0
    c = res.cost
    assert c.metric_calls == scorer.calls <= 30 and c.metric_budget == 30
    assert c.scorer_tokens == 11 * scorer.calls
    assert c.reflection_calls >= 1 and c.reflection_tokens > 0
    assert c.total_tokens == c.scorer_tokens + c.reflection_tokens
    # reflection saw typed failure feedback only
    prompt = spy.prompts[0]
    assert "violated_rules=R-" in prompt and "category=missing" in prompt


@pytest.mark.parametrize("cap", [7, 9, 13, 20])
def test_budget_cap_is_hard(cap):
    scorer = RecordingScorer()
    res = run(optimize_texts({PROMPT: "x"}, scorer, list(KW), reflection_lm=fake_lm("still nothing"),
                             config=GepaConfig(max_metric_calls=cap)))
    assert scorer.calls <= cap
    assert res.cost.metric_calls == scorer.calls


def test_budget_tokens_bound_metric_calls_and_noop_when_too_small():
    scorer = RecordingScorer()
    res = run(optimize_texts({PROMPT: "x"}, scorer, list(KW), reflection_lm=fake_lm("y"),
                             budget_tokens=3 * 4000, config=GepaConfig(max_metric_calls=100)))
    assert res.cost.metric_budget == 3 and "one GEPA iteration" in res.note
    assert scorer.calls == 0 and res.changed == {}


def test_only_evolve_cases_reach_domain(worktree, domain, tmp_path):
    scorer = DomainEvolveScorer(domain, worktree, tmp_path / "scratch", experiment_id="e", variant="a")

    async def go():
        return await optimize_texts({PROMPT: "hi"}, scorer, scorer.evolve_cases(),
                                    reflection_lm=fake_lm("refund tracking escalate polite"),
                                    config=GepaConfig(max_metric_calls=24))

    res = run(go())
    assert set(domain.splits_called) == {"evolve"}
    assert res.changed[PROMPT] == "refund tracking escalate polite"
    assert res.cost.scorer_tokens == scorer.tokens_spent > 0


def test_adapter_reflective_dataset_and_refusal():
    inc = FailureRecord("c2", "s", "inc", ("R9",), {}, "")
    g = EvolveGuard(lambda cand, cases: [CaseOutcome(c, 0.0) for c in cases], ["c1", "c2"], MetricBudget(2),
                    incumbent_failures=[inc])
    a = ComponentAdapter(g)
    eb = a.evaluate([{"case_id": "c1"}, {"case_id": "c2"}], {"p": "t"}, capture_traces=True)
    ds = a.make_reflective_dataset({"p": "t"}, eb, ["p"])
    assert [r["Inputs"]["case_id"] for r in ds["p"]] == ["c1", "c2"]
    assert "violated_rules=R9" in ds["p"][1]["Feedback"]  # incumbent FailureRecord fallback
    assert ds["p"][0]["Feedback"].startswith("score=0.000")
    eb2 = a.evaluate([{"case_id": "c1"}], {"p": "u"}, capture_traces=True)  # budget spent
    assert eb2.scores == [0.0] and eb2.num_metric_calls == 0
    assert a.make_reflective_dataset({"p": "u"}, eb2, ["p"]) == {"p": []}


def test_async_scorer_runs_on_caller_loop():
    loops = []

    async def scorer(cand, cases):
        loops.append(asyncio.get_running_loop())
        return RecordingScorer()(cand, cases)

    async def go():
        res = await optimize_texts({PROMPT: "x"}, scorer, list(KW), reflection_lm=fake_lm("refund"),
                                   config=GepaConfig(max_metric_calls=12))
        return res, asyncio.get_running_loop()

    res, loop = run(go())
    assert loops and all(lp is loop for lp in loops)


def test_reflection_lm_propagates_trace_headers_only_to_http_engines(monkeypatch):
    from ci_lab import obs
    from ci_lab.optim.gepa import DspyReflectionLM

    class HttpLM:
        _engine_spec = "litellm"

        def __init__(self):
            self.kw = []

        def __call__(self, **kw):
            self.kw.append(kw)
            return ["ok"]

    monkeypatch.setattr(obs, "carrier", lambda: {"traceparent": "00-abc-def-01"})
    http = HttpLM()
    assert DspyReflectionLM(http)("p") == "ok"
    assert http.kw == [{"prompt": "p", "extra_headers": {"traceparent": "00-abc-def-01"}}]
    assert isinstance(DspyReflectionLM(fake_lm("z"))("p"), str)  # DummyEngine: client kwargs withheld
