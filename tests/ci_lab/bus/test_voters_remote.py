from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.voters.local import run_voters
from ci_lab.bus.voters.remote import AgentVoter, AssertVoter, S1RubricVoter, s1_ballot
from ci_lab.contracts import EvalResult, EvaluatorPin, TaskScore
from ci_lab.judge import admission
from ci_lab.judge.backends import BackendError, ScriptedBackend
from ci_lab.judge.s1types import Answer
from ci_lab.tools.submit import VerdictSubmission

POOLS = {"s1": 1, "llm": 2}


def _run(kit, voters, rubric, artifact=b"hello", timeout_s=5.0):
    return asyncio.run(run_voters(voters, kit.proposal(artifact), artifact, rubric, kit.spec(),
                                  pools=ResourcePools(POOLS), timeout_s=timeout_s))


class FakeDomain:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def pin(self):
        return EvaluatorPin("tree1", "judge-m", "p")

    async def evaluate(self, harness_dir, split, k, *, experiment_id, variant):
        text = (harness_dir / "out" / "answer.txt").read_text()
        self.calls.append((split, text))
        good = 1.0 if text == "hello" else 0.0
        return EvalResult("h", split, self.pin(), [TaskScore("c1", 0, "suiteA", good),
                                                   TaskScore("c2", 0, "suiteA", None),
                                                   TaskScore("c3", 0, "suiteB", 0.95)])


def test_assert_voter_scores_thresholds_and_caches(kit, tmp_path):
    a = lambda cid, suite, m, split="heldout": kit.crit(cid, "assert", {"suite": suite, "split": split, "min_score": m})
    rubric = kit.rubric(a("a1", "suiteA", 0.4), a("a2", "suiteB", 0.96), a("a3", "suiteZ", 0.5),
                        kit.crit("d1"))
    dom = FakeDomain()
    v = AssertVoter(dom, cache_dir=tmp_path)
    votes = _run(kit, [v], rubric)
    assert [(x.criterion, x.passed, x.score) for x in votes] == [
        ("a1", True, 0.5), ("a2", False, 0.95), ("a3", None, None)]
    assert "suiteZ" in votes[2].reasons[0] and votes[0].measure == "assert"
    assert len(dom.calls) == 2  # suiteA+suiteB share one evaluation; unknown suite retries once
    _run(kit, [v], rubric)
    assert len(dom.calls) == 3  # in-memory hits for a1/a2; a3 still unknown
    fresh = AssertVoter(dom, cache_dir=tmp_path, criteria=["a1", "a2"])
    assert [x.score for x in _run(kit, [fresh], rubric)] == [0.5, 0.95] and fresh.evaluations == 0
    repinned = AssertVoter(dom, cache_dir=tmp_path, pin="tree2|judge-m", criteria=["a1"])
    assert [x.passed for x in _run(kit, [repinned], rubric, b"bad")] == [False] and repinned.evaluations == 1
    assert dom.calls[-1] == ("heldout", "bad")


def _answer(state, name, q):
    if "boom" in state:
        raise BackendError("server down")
    if q.type == "choice":
        return Answer.from_choice_distribution({"yes": 0.7, "no": 0.3})
    if "refuse" in str(q.instructions):
        return Answer.non_answer("noul", "refusal")
    return Answer.from_noul_probability(0.8)


def _s1_rubric(kit):
    return kit.rubric(kit.crit("s.a", "s1", {"question": "is it ok?", "type": "noul"}),
                      kit.crit("s.b", "s1", {"question": "which?", "type": "choice", "options": ["yes", "no"]},
                               threshold=0.75),
                      kit.crit("s.c", "s1", {"question": "refuse me", "type": "noul"}),
                      kit.crit("l.a", "llm", {"question": "deep?", "type": "noul"}), kit.crit("d1"))


def test_s1_voter_maps_answers_and_abstains(kit, monkeypatch):
    leases: list[str] = []

    @asynccontextmanager
    async def fake_hold(url, **_):
        leases.append(url)
        yield None

    monkeypatch.setattr(admission, "hold_async", fake_hold)
    backend = ScriptedBackend(_answer)
    s1 = S1RubricVoter("s1/llamacpp/tiny", backend=backend, api_base="http://127.0.0.1:9")
    llm = S1RubricVoter("s1/scripted/x", backend=ScriptedBackend(_answer), name="s1-llm", measure="llm", pool=None)
    votes = _run(kit, [s1, llm], _s1_rubric(kit))
    got = [(v.voter, v.criterion, v.passed, v.score, v.confidence) for v in votes]
    assert got == [("s1", "s.a", True, 0.8, pytest.approx(0.6)), ("s1", "s.b", False, 0.7, pytest.approx(0.4)),
                   ("s1", "s.c", None, None, None), ("s1-llm", "l.a", True, 0.8, pytest.approx(0.6))]
    assert leases == ["http://127.0.0.1:9"]  # local model leased; scripted model not
    state, questions = backend.requests[0]
    assert "hello" in state and [q.instructions for q in questions.values()] == ["is it ok?", "which?", "refuse me"]
    down = _run(kit, [S1RubricVoter("s1/scripted/x", backend=ScriptedBackend(_answer))], _s1_rubric(kit), b"boom")
    assert [(v.passed, v.reasons[0][:16]) for v in down] == [(None, "s1 backend error")] * 3


def test_s1_ballot_choice_without_probabilities(kit):
    c = kit.crit("s.b", "s1", {"question": "q", "type": "choice", "options": ["yes", "no"]})
    assert s1_ballot(Answer(type="choice", choice="yes"), c).passed is True
    assert s1_ballot(Answer(type="choice", choice="no"), c).score == 0.0


def test_agent_voter_with_injected_runner(kit):
    seen: dict[str, str] = {}

    async def runner(key, run_dir, tools):
        seen.update(brief=tools["read_brief"]("brief"), proposal=tools["read_brief"]("proposal"),
                    files=tools["list_files"](), file=tools["read_file"]("out/answer.txt"), key=key)
        ok = "hello" in seen["file"]
        return VerdictSubmission(verdict="accept" if ok else "reject", reasons=[] if ok else ["no greeting"])

    rubric = kit.rubric(kit.crit("l.a", "llm", {"question": "Does it greet?", "type": "noul"}), kit.crit("d1"))
    votes = _run(kit, [AgentVoter(runner=runner)], rubric)
    assert [(v.voter, v.criterion, v.passed, v.measure) for v in votes] == [("critic-agent", "l.a", True, "llm")]
    assert "Does it greet?" in seen["brief"] and "hello" in seen["proposal"] and seen["files"] == "out/answer.txt"
    assert seen["key"] == "critic"
    bad = _run(kit, [AgentVoter(runner=runner)], rubric, b"bye")
    assert (bad[0].passed, bad[0].reasons) == (False, ("no greeting",))
    with pytest.raises(ValueError):
        AgentVoter()


def test_agent_voter_default_runner_uses_meta_agent(kit):
    class FakeAgent:
        def __init__(self, bindings):
            self.bindings = bindings

        async def run(self, message, session=None):
            self.bindings["submit_verdict"](verdict="reject", reasons=["vague"])

    def builder(spec, *, client, bindings, **_):
        assert set(bindings) >= {"read_brief", "list_documents", "list_files", "read_file", "submit_verdict"}
        return FakeAgent(bindings)

    rubric = kit.rubric(kit.crit("l.a", "llm", {"question": "Does it greet?", "type": "noul"}))
    votes = _run(kit, [AgentVoter(client=object(), builder=builder)], rubric)
    assert [(v.passed, v.reasons) for v in votes] == [(False, ("vague",))]
