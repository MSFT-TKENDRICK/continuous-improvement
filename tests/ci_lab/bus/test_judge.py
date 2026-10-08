from __future__ import annotations

import asyncio
import warnings

import pytest

from ci_lab.bus.ids import proposal_id
from ci_lab.bus.judge import JUDGE_SPEC, VerdictDraft, aggregate, duel, judge
from ci_lab.bus.state import BusState, apply, check
from ci_lab.bus.types import (
    GENESIS,
    ArtifactRef,
    Author,
    Entry,
    ManifestBody,
    ProposalBody,
    VoteBody,
)
from ci_lab.taskgraph.model import Criterion, Rubric

H = "c" * 64
PID = proposal_id("t1@1", "student", "stu")
ADV = proposal_id("t1@1", "adversary", "null_model")
Q = {"question": "Does the answer cite a source?", "type": "noul"}
RUBRIC = Rubric("rub", 1, "t1", (
    Criterion("fmt", "format ok", "deterministic", {"kind": "regex", "pattern": "x"}, 1.0, required=True),
    Criterion("val", "valid", "deterministic", {"kind": "regex", "pattern": "y", "independent": True}, 1.0),
    Criterion("cite", "cites", "s1", Q, 0.5, weight=2.0, required=True),
    Criterion("tone", "tone", "llm", {**Q, "question": "Is it polite?"}, 0.5),
), pass_score=0.7, canary="0123456789abcdef")


def v(voter: str, crit: str | None, passed: bool | None = None, score: float | None = None, *, pid: str = PID,
      rv: str = "rub@v1") -> VoteBody:
    measure = "deterministic" if crit in ("fmt", "val") else "s1"
    return VoteBody(proposal=pid, rubric_version=rv, voter=voter, measure=measure, criterion=crit, passed=passed,
                    score=score, confidence=None)


def good(pid: str = PID) -> list[VoteBody]:
    return [v("det", "fmt", True, pid=pid), v("det", "val", True, pid=pid), v("a", "cite", score=1.0, pid=pid),
            v("b", "cite", score=1.0, pid=pid), v("a", "tone", score=1.0, pid=pid)]


def test_commit_when_everything_passes():
    d = aggregate(RUBRIC, good(), 2, attempts_left=2)
    assert d.decision == "commit" and d.score == 1.0 and d.voters == ("a", "b", "det") and not d.escalate
    assert d.criteria["cite"].votes == 2 and d.criteria["fmt"].oracle and not d.reasons


@pytest.mark.parametrize("left, expected", [(2, "revise"), (0, "reject")])
def test_oracle_veto_cannot_be_outscored(left, expected):
    votes = [*good(), v("det2", "val", False)]
    d = aggregate(RUBRIC, votes, 1, attempts_left=left)
    assert d.decision == expected and d.vetoed == ("val",) and d.hard_blocked
    assert d.criteria["val"].passed is False
    # an oracle score below threshold with no explicit pass/fail is also a veto
    assert aggregate(RUBRIC, [*good(), v("det2", "fmt", score=0.5)], 1, attempts_left=1).vetoed == ("fmt",)


@pytest.mark.parametrize("cite_votes", [[], [v("a", "cite"), v("b", "cite")]])
def test_missing_or_abstained_required_fails_closed(cite_votes):
    votes = [x for x in good() if x.criterion != "cite"] + cite_votes
    d = aggregate(RUBRIC, votes, 1, attempts_left=1)
    assert d.decision == "revise" and d.missing_required == ("cite",) and d.criteria["cite"].passed is None
    assert d.score == 1.0  # scored over answered criteria only, still fails closed


def test_quorum_counts_distinct_answering_voters():
    votes = [v("det", "fmt", True), v("det", "val", True), v("det", "cite", score=1.0), v("det", "tone", score=1.0),
             v("x", "cite"), v("x", "tone"), v("y", "ghost", True)]
    d = aggregate(RUBRIC, votes, 2, attempts_left=1)
    assert d.voters == ("det",) and not d.quorum_met and d.decision == "revise"
    assert aggregate(RUBRIC, [*votes, v("z", None, True)], 2, attempts_left=1).decision == "commit"


def test_weighted_score_and_soft_threshold():
    votes = [v("det", "fmt", True), v("det", "val", True), v("a", "cite", score=0.6), v("b", "tone", score=0.0)]
    d = aggregate(RUBRIC, votes, 1, attempts_left=1)
    assert d.score == pytest.approx((1 + 1 + 2 * 0.6 + 0) / 5)
    assert d.criteria["tone"].passed is False and d.decision == "revise" and d.failed_required == ()


def test_escalate_flags():
    split = [v("det", "fmt", True), v("det", "val", True), v("a", "cite", score=1.0), v("b", "cite", score=0.2)]
    d = aggregate(RUBRIC, split, 1, attempts_left=1)
    assert d.soft_stdev == pytest.approx(0.4) and d.escalate
    margin = [v("det", "fmt", True), v("det", "val", True), v("a", "cite", score=0.55), v("a", "tone", score=0.55)]
    d = aggregate(RUBRIC, margin, 1, attempts_left=1)
    assert d.score == pytest.approx(0.73) and d.escalate and d.soft_stdev == 0.0 and d.decision == "commit"


def test_inputs_must_be_one_proposal_and_version():
    with pytest.raises(ValueError, match="proposal"):
        aggregate(RUBRIC, [*good(), v("a", "fmt", True, pid=ADV)], 1, attempts_left=1)
    with pytest.raises(ValueError, match="rubric version"):
        aggregate(RUBRIC, [v("a", "fmt", True, rv="rub@v2")], 1, attempts_left=1)
    with pytest.raises(ValueError):
        aggregate(RUBRIC, [], 1, attempts_left=1)
    d = aggregate(RUBRIC, [], 1, attempts_left=1, proposal=PID)
    assert d.decision == "revise" and set(d.missing_required) == {"fmt", "cite"}


class Esc:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    async def review(self, rubric, votes, draft):
        self.calls += 1
        if self.answer == "raise":
            raise RuntimeError("boom")
        return self.answer


CLOSE_FAIL = [v("det", "fmt", True), v("det", "val", True), v("a", "cite", score=0.66), v("b", "cite", score=0.66),
              v("a", "tone", score=0.66)]  # score 0.796 < 0.8 pass -> revise, escalate
CLOSE_PASS = [v("det", "fmt", True), v("det", "val", True), v("a", "cite", score=0.7), v("b", "cite", score=0.7),
              v("a", "tone", score=0.7)]  # 0.82 >= 0.8 -> commit, escalate
HARD = {
    "veto": [*CLOSE_PASS, v("c", "val", False)],
    "missing_required": [x for x in CLOSE_PASS if x.criterion != "fmt"],
    "abstained_required": [*(x for x in CLOSE_PASS if x.criterion != "cite"), v("a", "cite"), v("b", "cite")],
    "no_quorum": [x for x in CLOSE_PASS if x.voter != "b"],
}
RUB8 = Rubric(RUBRIC.id, 1, "t1", RUBRIC.criteria, pass_score=0.8, canary=RUBRIC.canary)


@pytest.mark.parametrize("answer", ["commit", "revise", None, "reject", "raise"])
@pytest.mark.parametrize("case", sorted(HARD))
def test_escalator_never_overrides_hard_blocks(case, answer):
    esc = Esc(answer)
    base = aggregate(RUB8, HARD[case], 3, attempts_left=1)
    d = asyncio.run(judge(RUB8, HARD[case], 3, 1, esc))
    assert base.hard_blocked and d.decision == base.decision == "revise" and not d.escalated and esc.calls == 0


@pytest.mark.parametrize("votes, left, answer, expected", [
    (CLOSE_FAIL, 1, "commit", "commit"), (CLOSE_FAIL, 1, "revise", "revise"), (CLOSE_FAIL, 1, None, "revise"),
    (CLOSE_FAIL, 0, None, "reject"), (CLOSE_FAIL, 0, "commit", "commit"),
    (CLOSE_PASS, 1, "revise", "revise"), (CLOSE_PASS, 0, "revise", "reject"), (CLOSE_PASS, 1, "commit", "commit"),
    (CLOSE_PASS, 1, "reject", "commit"), (CLOSE_PASS, 1, "raise", "commit"), (CLOSE_FAIL, 1, "raise", "revise"),
])
def test_escalator_flips_only_soft_margins(votes, left, answer, expected):
    esc = Esc(answer)
    base = aggregate(RUB8, votes, 2, attempts_left=left)
    assert base.escalate and base.soft_only
    d = asyncio.run(judge(RUB8, votes, 2, left, esc))
    assert esc.calls == 1 and d.decision == expected and d.escalated == (answer != "raise")


def test_no_escalation_when_not_flagged():
    esc = Esc("revise")
    d = asyncio.run(judge(RUBRIC, good(), 2, 1, esc))
    assert d.decision == "commit" and esc.calls == 0 and asyncio.run(judge(RUBRIC, good(), 2, 1)) == d


def _bus_with_votes() -> tuple[BusState, dict[int, VoteBody], ManifestBody]:
    m = ManifestBody(run="r1", created="t", code_rev="c", config_sha256=H, graph_sha256=None, max_parallel=1,
                     voters=("a", "b", "det"), quorum=2)
    state = BusState()

    def add(kind, author, body, ref=None):
        nonlocal state
        e = Entry(seq=state.next_seq, topic="r1/t1", kind=kind, author=author, ref=ref, body=body, ts="t",
                  prev=state.head or GENESIS).sealed()
        check(state, e, manifest=m)
        state = apply(state, e)
        return e.seq

    art = ArtifactRef(path=f"cc/{H}", sha256=H, bytes=3)
    seqs = {}
    for pid, role, name in ((PID, "student", "stu"), (ADV, "adversary", "null_model")):
        seqs[pid] = add("proposal", Author(role=role, name=name),
                        ProposalBody(proposal=pid, attempt="t1@1", rubric_version="rub@v1", artifact=art, summary=""))
    votes = {}
    for pid in (PID, ADV):
        for x in good(pid):
            votes[add("vote", Author(role="voter", name=x.voter), x, seqs[pid])] = x
    return state, votes, m


def test_to_body_cites_only_this_proposal_and_passes_state_check():
    state, votes, m = _bus_with_votes()
    d = aggregate(RUBRIC, [x for x in votes.values() if x.proposal == PID], m.quorum, attempts_left=1)
    body = d.to_body(votes)
    assert body.votes == tuple(s for s, x in sorted(votes.items()) if x.proposal == PID)
    e = Entry(seq=state.next_seq, topic="r1/t1", kind="verdict", author=Author(role="judge", name="j"), ref=0,
              body=body, ts="t", prev=state.head)
    check(state, e, manifest=m)
    assert body.decision == "commit" and body.criteria == d.criteria and body.attempt == "t1@1"
    few = {s: x for s, x in votes.items() if x.voter == "det"}
    with pytest.raises(ValueError, match="distinct cited voters"):
        d.to_body(few)
    assert isinstance(d, VerdictDraft)


def test_duel_exploit_requires_independent_oracle_failure():
    student = [v("a", "cite", score=0.6, pid=PID), v("a", "tone", score=0.6, pid=PID)]
    adv = [v("a", "cite", score=1.0, pid=ADV), v("a", "tone", score=1.0, pid=ADV), v("det", "val", False, pid=ADV),
           v("det", "fmt", False, pid=ADV)]
    r = duel(RUBRIC, student, adv)
    assert r.soft_pref == "adversary" and r.soft_pass_adversary and r.oracle_invalid_adversary == ("val",) and r.exploit
    fmt = RUBRIC.criteria[0]
    assert duel(RUBRIC, student, adv, validity_oracles=(fmt,)).oracle_invalid_adversary == ("fmt", "val")
    honest = [x for x in adv if x.criterion != "val"]
    assert not duel(RUBRIC, student, honest).exploit  # fmt is editable by the hardener: not a validity source
    weak = [v("a", "cite", score=0.1, pid=ADV), v("det", "val", False, pid=ADV)]
    r = duel(RUBRIC, student, weak)
    assert r.soft_pref == "student" and not r.soft_pass_adversary and not r.exploit
    assert duel(RUBRIC, [], []).soft_pref == "tie" and duel(RUBRIC, student, student).soft_pref == "tie"


def test_judge_spec_loads_and_builds():
    from ci_lab.meta.spec_loader import default_builder, load_spec
    from ci_lab.testing import FakeChatClient

    spec = load_spec(JUDGE_SPEC)
    assert spec.name == "CiJudge" and spec.terminal_tool == "submit_escalation" and spec.purpose == "judge"
    assert set(spec.tools) == {"read_brief", "list_documents", "submit_escalation"}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        agent = default_builder()(spec, client=FakeChatClient(), bindings={t: (lambda: "x") for t in spec.tools},
                                  loop_should_continue=lambda **_: False, loop_next_message=lambda **_: "x")
    assert agent.additional_properties["ci_lab"]["model"] == spec.model
