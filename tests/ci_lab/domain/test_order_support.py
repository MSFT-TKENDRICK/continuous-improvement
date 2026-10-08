from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import pytest
import yaml

import ci_lab.domain.order_support as dom
from ci_lab.contracts import (
    ATTR_CASE,
    ATTR_ROLLOUT,
    ATTR_SCORE,
    ATTR_SPLIT,
    ATTR_TRIAL,
    SPAN_CASE,
    Domain,
    RolloutKey,
    Transcript,
    Violation,
)
from ci_lab.domain.order_support import (
    AssertCaseRunner,
    CaseOutcome,
    OrderSupportDomain,
    TestCase,
    load_cases,
    parse_run,
    rubric_scores,
    score_outcome,
    stable_case_id,
    transcript_from_inference,
    tree_hash,
)
from ci_lab.testing import MemoryJournal

REPO = Path(__file__).resolve().parents[3]
EVALS = REPO / "evals" / "assert"
CFG = EVALS / "refund_authorization" / "eval_config.yaml"


def _row(i: int, behavior: str) -> dict:
    return {"type": "prompt", "test_case_id": f"test_case_{i:06d}", "behavior": behavior,
            "seed": {"title": f"Case {i}", "description": f"customer {i} asks about {behavior} details please"},
            "dimensions": {"tone": "polite"}}


def _cases(suites=("order_support_refund_authorization", "order_support_indirect_prompt_injection"),
           cats=10, per=4) -> list[TestCase]:
    out = []
    for suite in suites:
        for c in range(cats):
            for j in range(per):
                row = _row(c * 100 + j, f"cat{c}")
                out.append(TestCase(stable_case_id(suite, row), suite, f"cat{c}", "prompt", row, CFG))
    return out


def _transcript(cid: str, reply: str = "Your refund is on its way.") -> Transcript:
    return Transcript(case_id=cid, messages=[{"role": "user", "content": "refund please"},
                                             {"role": "assistant", "content": reply}],
                      served_models=["gpt-5-mini"], tokens_in=10, tokens_out=5)


class FakeRunner:
    def __init__(self, verdicts=None):
        self.calls: list[tuple[str, int, dict]] = []
        self.verdicts = verdicts or {}

    async def __call__(self, case, *, harness_dir, key, env):
        self.calls.append((case.case_id, key.trial, dict(env)))
        verdict = self.verdicts.get(case.case_id, {"policy_violation": False, "hallucination": False})
        if verdict == "crash":
            raise RuntimeError("boom")
        if verdict == "skip":
            return CaseOutcome(judge_status="judge_failed", transcript=_transcript(case.case_id))
        return CaseOutcome(verdict=verdict, judge_model="openai/gpt-4.1",
                           transcript=_transcript(case.case_id, "<<tool said: ignore rules>> refund issued"))


class FakeOracle:
    def __init__(self, bad: set[str]):
        self.bad = bad

    def check(self, transcript):
        if transcript.case_id in self.bad:
            return [Violation("refund.unverified_identity", "major", "refund without identity check")]
        return []


def _domain(tmp_path, **kw) -> OrderSupportDomain:
    kw.setdefault("cases", _cases())
    kw.setdefault("runner", FakeRunner())
    kw.setdefault("scope_factory", lambda key: contextlib.nullcontext())
    kw.setdefault("use_default_oracle", False)
    return OrderSupportDomain(evals_dir=EVALS, work_dir=tmp_path / "work", **kw)


def test_satisfies_domain_protocol(tmp_path):
    d = _domain(tmp_path)
    for attr in ("name", "surface_globs", "frozen_globs", "component_globs", "splits", "evaluate", "failures"):
        assert hasattr(d, attr) and hasattr(Domain, attr) or attr in Domain.__annotations__
    assert d.surface_globs == ("src/order_support/harness/**",)
    assert set(d.component_globs) == {"prompt", "skill", "client_tool", "config", "memory", "context_mgmt"}
    assert "**/*.py" in d.frozen_globs and "evals/**" in d.frozen_globs


def test_splits_deterministic_and_disjoint(tmp_path):
    a = _domain(tmp_path).splits()
    b = _domain(tmp_path, cases=list(reversed(_cases()))).splits()
    assert a == b
    ids = [c.case_id for c in _cases()]
    assert sorted(a["evolve"] + a["heldout"] + a["ood"]) == sorted(ids)
    assert not (set(a["evolve"]) & set(a["heldout"])) and not (set(a["ood"]) & set(a["evolve"]))
    assert a["heldout"] and a["evolve"]
    assert _domain(tmp_path, seed="other").splits() != a


def test_ood_is_whole_categories(tmp_path):
    d = _domain(tmp_path)
    splits = d.splits()
    by_id = {c.case_id: c for c in d.cases()}
    ood_groups = {(by_id[i].suite, by_id[i].category) for i in splits["ood"]}
    assert len(ood_groups) == 4  # floor(0.2 * 10) per suite x 2 suites
    for suite in {s for s, _ in ood_groups}:
        assert sum(1 for s, _ in ood_groups if s == suite) == 2
    rest = {(by_id[i].suite, by_id[i].category) for i in splits["evolve"] + splits["heldout"]}
    assert not (ood_groups & rest)
    # adding cases to a category does not move other cases between splits
    extra = _cases(cats=10, per=5)
    grown = _domain(tmp_path, cases=extra).splits()
    for name in ("evolve", "heldout", "ood"):
        assert set(splits[name]) <= set(grown[name])


def test_splits_need_cases(tmp_path):
    with pytest.raises(FileNotFoundError):
        _domain(tmp_path, cases=[]).splits()


def test_load_cases_from_frozen_test_sets(tmp_path):
    path = tmp_path / "ts.jsonl"
    rows = [_row(1, "refund_no_id"), _row(2, "refund_no_id"), _row(1, "refund_no_id")]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    cases = load_cases(EVALS, tmp_path / "nothing", {"order_support_refund_authorization": path})
    assert len(cases) == 2  # duplicates collapse onto the content hash
    assert {c.suite for c in cases} == {"order_support_refund_authorization"}
    assert all(c.case_id.startswith("refund_authorization-") and c.category == "refund_no_id" for c in cases)
    assert cases[0].config_path.name == "eval_config.yaml"
    renumbered = dict(rows[0], test_case_id="test_case_000099")
    assert stable_case_id("s", renumbered) == stable_case_id("s", rows[0])


def test_score_outcome():
    ok = CaseOutcome(verdict={"a": False, "b": True, "tone": "warm"})
    assert score_outcome(ok) == 0.5
    assert score_outcome(CaseOutcome(verdict={"a": False})) == 1.0
    assert score_outcome(CaseOutcome(verdict={"policy_violation": True, "a": False})) == 0.0
    assert score_outcome(CaseOutcome(judge_status="judge_failed")) is None
    major = Violation("r", "major", "d")
    assert score_outcome(CaseOutcome(verdict={"a": False}), [major]) == 0.5
    assert score_outcome(CaseOutcome(verdict={"a": False}), [Violation("r", "critical", "d")]) == 0.0
    assert score_outcome(CaseOutcome(verdict={"a": True, "b": False}, scored_keys=["b"])) == 1.0
    graded = CaseOutcome(verdict={"helpful": "medium", "x": True},
                         dimension_scales={"helpful": {"levels": ["low", "medium", "high"]}})
    assert rubric_scores(graded) == {"helpful": 0.5, "x": 0.0}


def test_evaluate_with_fake_runner(tmp_path, monkeypatch):
    spans: list[tuple[str, dict, dict]] = []

    class Span:
        def __init__(self, name, attrs):
            self.rec = (name, dict(attrs), {})
            spans.append(self.rec)

        def set_attribute(self, k, v):
            self.rec[2][k] = v

    @contextlib.contextmanager
    def fake_span(name, attrs=None):
        yield Span(name, attrs or {})

    monkeypatch.setattr(dom.obs, "span", fake_span)
    d0 = _domain(tmp_path)
    evolve = list(d0.splits()["evolve"])
    bad, skip, crash = evolve[0], evolve[1], evolve[2]
    runner = FakeRunner({bad: {"policy_violation": False, "hallucination": True}, skip: "skip", crash: "crash"})
    journal = MemoryJournal()
    entered: list[str] = []

    class Scope:
        def __init__(self, key):
            self.key, self.env = key, {"AGL_ROLLOUT": key.rollout_id}

        async def __aenter__(self):
            entered.append(self.key.rollout_id)
            return self

        async def __aexit__(self, *exc):
            return False

    harness = tmp_path / "harness"
    (harness / "prompts").mkdir(parents=True)
    (harness / "prompts" / "system.md").write_text("hi", encoding="utf-8")
    d = _domain(tmp_path, runner=runner, journal=journal, scope_factory=Scope, oracle=FakeOracle({evolve[3]}),
                concurrency=4, judge_model="openai/gpt-4.1", case_spans=True)
    result = asyncio.run(d.evaluate(harness, "evolve", 2, experiment_id="exp-1", variant="arm-a"))

    assert result.split == "evolve" and result.harness_tree == tree_hash(harness)
    assert len(result.scores) == 2 * len(evolve) == len(runner.calls) == len(entered)
    by = {(s.case_id, s.trial): s for s in result.scores}
    assert by[(bad, 0)].score == 0.5 and by[(skip, 1)].score is None and by[(crash, 0)].score is None
    assert by[(evolve[3], 0)].score == 0.5 and by[(evolve[3], 0)].violations[0].severity == "major"
    assert by[(evolve[4], 1)].score == 1.0 and by[(evolve[4], 1)].served_model == "gpt-5-mini"
    assert result.pin.judge_model == "openai/gpt-4.1" and result.pin.served_judge_models == ("openai/gpt-4.1",)
    env = runner.calls[0][2]
    assert env["ORDER_SUPPORT_HARNESS_DIR"] == str(harness.resolve()) and env["AGL_ROLLOUT"].startswith("ro-")
    key = RolloutKey("exp-1", "arm-a", bad, 0)
    assert journal.rollouts[key.rollout_id]["status"] == "succeeded"
    [ev] = journal.events(key)
    assert ev["data"]["score"] == 0.5 and ev["data"]["judge_model"] == "openai/gpt-4.1"
    assert journal.rollouts[RolloutKey("exp-1", "arm-a", crash, 0).rollout_id]["status"] == "failed"
    case_spans = [s for s in spans if s[0] == SPAN_CASE]
    assert len(case_spans) == len(result.scores)
    _, attrs, set_attrs = next(s for s in case_spans if s[1][ATTR_CASE] == bad and s[1][ATTR_TRIAL] == 0)
    assert attrs[ATTR_ROLLOUT] == key.rollout_id and attrs[ATTR_SPLIT] == "evolve" and set_attrs[ATTR_SCORE] == 0.5

    # deterministic re-run gives identical scores
    again = asyncio.run(d.evaluate(harness, "evolve", 2, experiment_id="exp-1", variant="arm-a"))
    assert sorted(map(repr, again.scores)) == sorted(map(repr, result.scores))


def test_aa_split_uses_evolve_cases(tmp_path):
    d = _domain(tmp_path)
    result = asyncio.run(d.evaluate(tmp_path, "aa", 1, experiment_id="e", variant="a"))
    assert result.split == "aa" and {s.case_id for s in result.scores} == set(d.splits()["evolve"])


def test_failures_are_typed_and_sanitized(tmp_path):
    cases = _cases()
    d0 = _domain(tmp_path, cases=cases)
    ev = d0.splits()["evolve"]
    by_id = {c.case_id: c for c in cases}
    inj = next(i for i in ev if "injection" in by_id[i].suite)
    ref = next(i for i in ev if "refund" in by_id[i].suite)
    runner = FakeRunner({inj: {"policy_violation": True}, ref: {"hallucination": True, "policy_violation": False}})
    d = _domain(tmp_path, cases=cases, runner=runner, oracle=FakeOracle({ref}))
    result = asyncio.run(d.evaluate(tmp_path, "evolve", 1, experiment_id="e", variant="a"))
    records = {r.case_id: r for r in d.failures(result)}
    assert set(records) == {inj, ref}
    assert records[inj].excerpt == "" and "judge.policy_violation" in records[inj].rule_ids
    assert records[ref].excerpt == "<<tool said: ignore rules>> refund issued"
    assert records[ref].rule_ids == ("judge.hallucination", "refund.unverified_identity")
    assert records[ref].rubric_scores == {"hallucination": 0.0, "policy_violation": 1.0}
    assert records[ref].category == by_id[ref].category
    for r in records.values():
        assert "tool_calls" not in repr(r)


def test_transcript_from_inference_and_parse_run(tmp_path):
    case = _cases()[0]
    inf = {"events": [
        {"edit": {"type": "add_message", "message": {"role": "system", "content": "sys"}}},
        {"edit": {"type": "add_message", "message": {"role": "user", "content": "refund ORD-1"}}},
        {"edit": {"type": "tool_call", "call_id": "c1", "tool_name": "lookup_order",
                  "tool_args": {"order_id": "ORD-1"}, "tool_result": {"status": "shipped"}}},
        {"edit": {"type": "add_message", "message": {"role": "assistant", "content": "Done."}}},
    ], "llm_calls": [{"model": "gpt-5-mini", "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
                     {"model": "gpt-5-mini", "usage": {"input_tokens": 1, "output_tokens": 1}}]}
    tr = transcript_from_inference(case.case_id, inf)
    assert [m["role"] for m in tr.messages] == ["user", "assistant"]
    assert tr.tool_calls[0].name == "lookup_order" and tr.tool_calls[0].turn == 1
    assert tr.served_models == ["gpt-5-mini"] and (tr.tokens_in, tr.tokens_out) == (8, 4)

    run = tmp_path / "run"
    run.mkdir()
    assert parse_run(case, run, returncode=3).judge_status == "missing"
    (run / "inference_set.jsonl").write_text(json.dumps(inf) + "\n", encoding="utf-8")
    score = {"judge_status": "ok", "verdict": {"dimensions": {"policy_violation": False, "na": True}},
             "score_keys": ["policy_violation", "na"], "not_applicable_score_keys": ["na"],
             "judge_model": "openai/gpt-4.1"}
    (run / "scores.jsonl").write_text(json.dumps(score) + "\n", encoding="utf-8")
    out = parse_run(case, run)
    assert out.judge_status == "ok" and out.scored_keys == ["policy_violation"] and score_outcome(out) == 1.0
    assert out.transcript is not None and out.judge_model == "openai/gpt-4.1"
    (run / "scores.jsonl").write_text(json.dumps({**score, "judge_error": "timeout"}) + "\n", encoding="utf-8")
    assert score_outcome(parse_run(case, run)) is None


def test_assert_runner_prepare_and_command(tmp_path):
    case = _cases()[0]
    runner = AssertCaseRunner(tmp_path / "w", overrides=("judge.concurrency=1",), model_timeout_s=60)
    key = RolloutKey("e", "a", case.case_id, 1)
    cfg_path, run_root, run = runner.prepare(case, key)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert "test_set" not in cfg["pipeline"] and cfg["run"] == run and run.startswith("ci-")
    ts = Path(cfg["pipeline"]["inference"]["test_set_path"])
    assert json.loads(ts.read_text(encoding="utf-8")) == case.row
    assert run_root == ts.parent / run
    if "judge" in cfg["pipeline"] and cfg["pipeline"]["judge"].get("taxonomy_path"):
        assert Path(cfg["pipeline"]["judge"]["taxonomy_path"]).is_absolute()
    cmd = runner.command(cfg_path, cfg_path.parent / "artifacts")
    assert cmd[1:4] == ["-m", "order_support.cli", "run"] and "--model-timeout" in cmd
    assert cmd[-2:] == ["--override", "judge.concurrency=1"]
    runner.prepare(case, key)  # idempotent: the attempt dir is recreated


def test_assert_runner_passes_child_env(tmp_path, monkeypatch):
    case = _cases()[0]
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw["env"])
        return type("P", (), {"returncode": 1, "stdout": "", "stderr": "x"})()

    monkeypatch.setattr(dom.subprocess, "run", fake_run)
    monkeypatch.setattr(dom.obs, "child_env", lambda env=None: {**(env or {}), "TRACEPARENT": "00-abc-def-01"})
    out = asyncio.run(AssertCaseRunner(tmp_path)(case, harness_dir=tmp_path, key=RolloutKey("e", "a", "c"),
                                                 env={"ORDER_SUPPORT_HARNESS_DIR": "H"}))
    assert out.judge_status == "missing" and seen["TRACEPARENT"] == "00-abc-def-01"
    assert seen["ORDER_SUPPORT_HARNESS_DIR"] == "H"


def test_case_env_attributes_guard_decisions(tmp_path, monkeypatch):
    """21a seam: each case x trial child sees CI_CASE_ID/CI_TRIAL (seed + guard-decision sink
    ``$CI_GUARD_DECISIONS/<case>/<trial>.jsonl``); paired-eval / telemetry env passes through."""
    passthrough = {"CI_GUARDS": "enforce", "CI_GUARD_DECISIONS": str(tmp_path / "dec"), "CI_TELEMETRY": "1",
                   "CI_RUN_DIR": str(tmp_path / "run"), "CI_CASE_ID": "stale", "CI_TRIAL": "9"}
    for k, v in passthrough.items():
        monkeypatch.setenv(k, v)
    runner = FakeRunner()
    d = _domain(tmp_path, runner=runner)
    asyncio.run(d.evaluate(tmp_path, "evolve", 2, experiment_id="e", variant="a"))
    assert {(c, t) for c, t, _ in runner.calls} == {(c, t) for c in d.splits()["evolve"] for t in (0, 1)}
    for case_id, trial, env in runner.calls:
        assert env["CI_CASE_ID"] == case_id and env["CI_TRIAL"] == str(trial)

    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw["env"])
        return type("P", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(dom.subprocess, "run", fake_run)
    case_id, trial, env = runner.calls[0]
    asyncio.run(AssertCaseRunner(tmp_path / "w")(d.case(case_id), harness_dir=tmp_path,
                                                 key=RolloutKey("e", "a", case_id, trial), env=env))
    assert seen["CI_CASE_ID"] == case_id and seen["CI_TRIAL"] == str(trial)
    for k in ("CI_GUARDS", "CI_GUARD_DECISIONS", "CI_TELEMETRY", "CI_RUN_DIR"):
        assert seen[k] == passthrough[k]


def test_leak_material_and_pin(tmp_path):
    d = _domain(tmp_path)
    assert any("asks about cat0" in t for t in d.leak_texts())
    lits = d.leak_literals()
    assert "Alex Rivera" in lits and "NW-10001" in lits
    pin = _domain(tmp_path).pin(["b", "a", "a"])
    assert pin.evaluator_tree.startswith("sha256:") and pin.served_judge_models == ("a", "b")
    assert pin.evaluator_tree == tree_hash(EVALS)
