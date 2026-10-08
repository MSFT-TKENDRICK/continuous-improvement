from __future__ import annotations

import json
import random
from typing import Any

import pytest

from ci_lab.bus.ids import proposal_id
from ci_lab.bus.state import WRITERS, BusInvariantError, BusState, apply, check
from ci_lab.bus.types import (
    GENESIS,
    KINDS,
    ROLES,
    AbortBody,
    ArtifactRef,
    Author,
    CommitBody,
    Entry,
    ExploitBody,
    IntentBody,
    ManifestBody,
    NoteBody,
    OutcomeBody,
    ProposalBody,
    RejectBody,
    RubricPatchBody,
    StudentCorrection,
    VerdictBody,
    VoteBody,
    canonical_json,
)

H = "b" * 64
ART = ArtifactRef(path="x/" + H, sha256=H, bytes=1)
RV = "rub@v1"
ORCH, JUDGE, HARD = Author(role="orchestrator", name="orch"), Author(role="judge", name="j"), Author(
    role="hardener", name="h")
STU, ADV = Author(role="student", name="stu"), Author(role="adversary", name="null_model")
MANIFEST = ManifestBody(run="r1", created="now", code_rev="abc", config_sha256=H, graph_sha256=None,
                        max_parallel=2, voters=("v1", "v2", "v3"), quorum=2)


def voter(name: str) -> Author:
    return Author(role="voter", name=name)


def prop(attempt: str = "t1@1", author: Author = STU, art: ArtifactRef = ART) -> ProposalBody:
    return ProposalBody(proposal=proposal_id(attempt, author.role, author.name), attempt=attempt,
                        rubric_version=RV, artifact=art, summary="s")


def vote(pid: str, name: str, criterion: str | None = "c1", passed: bool | None = True, rv: str = RV) -> VoteBody:
    return VoteBody(proposal=pid, rubric_version=rv, voter=name, measure="s1", criterion=criterion, passed=passed,
                    score=None, confidence=None)


def verdict(p: ProposalBody, decision: str, votes: tuple[int, ...] = (),
            correction: StudentCorrection | None = None) -> VerdictBody:
    return VerdictBody(proposal=p.proposal, attempt=p.attempt, rubric_version=p.rubric_version,
                       decision=decision, score=0.5, criteria={}, votes=votes, correction=correction,  # type: ignore[arg-type]
                       escalated=False)


def patch(n: int | None) -> RubricPatchBody:
    return RubricPatchBody(rubric_id="rub", from_version="rub@v1", to_version="rub@v2", applies_from_attempt=n,
                           applies_from_epoch=None, diff_sha256=H, metrics={}, accepted=True)


class Log:
    def __init__(self, topic: str = "r1/t1", manifest: ManifestBody | None = MANIFEST) -> None:
        self.topic, self.manifest, self.state, self.log = topic, manifest, BusState(), []

    def entry(self, kind: str, author: Author, body: Any, ref: int | None = None) -> Entry:
        return Entry(seq=self.state.next_seq, topic=self.topic, kind=kind, author=author, ref=ref,  # type: ignore[arg-type]
                     body=body, ts="t", prev=self.state.head or GENESIS).sealed()

    def add(self, kind: str, author: Author, body: Any, ref: int | None = None) -> int:
        e = self.entry(kind, author, body, ref)
        check(self.state, e, manifest=self.manifest)
        self.state = apply(self.state, e)
        self.log.append(e)
        return e.seq

    def bad(self, code: str, kind: str, author: Author, body: Any, ref: int | None = None) -> None:
        with pytest.raises(BusInvariantError) as ei:
            self.add(kind, author, body, ref)
        assert ei.value.code == code, ei.value


def committed_task() -> tuple[Log, int, int]:
    log = Log()
    p = prop()
    ps = log.add("proposal", STU, p)
    v1 = log.add("vote", voter("v1"), vote(p.proposal, "v1"), ps)
    v2 = log.add("vote", voter("v2"), vote(p.proposal, "v2"), ps)
    vs = log.add("verdict", JUDGE, verdict(p, "commit", (v1, v2)), ps)
    return log, ps, vs


def test_i0_manifest() -> None:
    run = Log("r1/_run", manifest=None)
    run.bad("I0", "note", ORCH, NoteBody(text="x"))
    run.bad("I0", "manifest", ORCH, ManifestBody(**{**MANIFEST.__dict__, "run": "r2"}))
    run.add("manifest", ORCH, MANIFEST)
    assert run.state.manifest == MANIFEST
    run.bad("I0", "manifest", ORCH, MANIFEST)
    run.add("note", ORCH, NoteBody(text="plan"))
    Log().bad("I0", "manifest", ORCH, MANIFEST)


def test_i1_intent_outcome() -> None:
    log = Log()
    log.bad("I1", "outcome", ORCH, OutcomeBody(intent_seq=0, ok=True))
    i0 = log.add("intent", ORCH, IntentBody(action="attempt", key="k", attempt="t1@1"))
    log.bad("I1", "intent", ORCH, IntentBody(action="attempt", key="k", attempt="t1@1"))
    assert [e.seq for e in log.state.intents_without_outcome()] == [i0]
    n = log.add("note", ORCH, NoteBody(text="x"))
    log.bad("I1", "outcome", ORCH, OutcomeBody(intent_seq=n, ok=True), n)
    log.bad("I1", "outcome", ORCH, OutcomeBody(intent_seq=n, ok=True), i0)
    log.add("outcome", ORCH, OutcomeBody(intent_seq=i0, ok=False), i0)
    log.bad("I1", "outcome", ORCH, OutcomeBody(intent_seq=i0, ok=True), i0)
    assert log.state.outcome_for("k") is None and log.state.intents_without_outcome() == ()
    i1 = log.add("intent", ORCH, IntentBody(action="attempt", key="k", attempt="t1@1"))
    o1 = log.add("outcome", ORCH, OutcomeBody(intent_seq=i1, ok=True, detail={"r": 1}), i1)
    assert log.state.outcome_for("k").seq == o1  # type: ignore[union-attr]
    log.bad("I1", "intent", ORCH, IntentBody(action="attempt", key="k", attempt=None))


def test_i2_votes() -> None:
    log = Log()
    p = prop()
    n = log.add("note", ORCH, NoteBody(text="x"))
    log.bad("I2", "vote", voter("v1"), vote(p.proposal, "v1"), n)
    ps = log.add("proposal", STU, p)
    log.bad("I2", "vote", voter("v1"), vote(p.proposal, "v1", rv="rub@v2"), ps)
    log.bad("I2", "vote", voter("v1"), vote(proposal_id("t1@2", "student", "stu"), "v1"), ps)
    log.add("vote", voter("v1"), vote(p.proposal, "v1"), ps)
    log.bad("I2", "vote", voter("v1"), vote(p.proposal, "v1", passed=False), ps)
    log.add("vote", voter("v1"), vote(p.proposal, "v1", criterion="c2"), ps)
    log.add("vote", voter("v1"), vote(p.proposal, "v1", criterion=None), ps)
    assert len(log.state.votes_for(ps)) == 3


def test_i3_verdict_quorum_over_distinct_voters() -> None:
    log = Log()
    p, q = prop(), prop("t1@1", ADV)
    ps, qs = log.add("proposal", STU, p), log.add("proposal", ADV, q)
    a = log.add("vote", voter("v1"), vote(p.proposal, "v1"), ps)
    b = log.add("vote", voter("v1"), vote(p.proposal, "v1", criterion="c2"), ps)
    c = log.add("vote", voter("v2"), vote(p.proposal, "v2", passed=None), ps)
    x = log.add("vote", voter("v2"), vote(q.proposal, "v2"), qs)
    log.bad("I3", "verdict", JUDGE, verdict(p, "commit", (a, b)), ps)  # one distinct voter
    log.bad("I3", "verdict", JUDGE, verdict(p, "commit", (a, b, c)), ps)  # abstention does not count
    log.bad("I3", "verdict", JUDGE, verdict(p, "commit", (a, x)), ps)  # vote on another proposal
    log.bad("I3", "verdict", JUDGE, verdict(p, "revise", (a, 99)), ps)
    log.bad("I3", "verdict", JUDGE, verdict(q, "revise", (a,)), ps)
    log.bad("I3", "verdict", JUDGE, verdict(p, "revise"), a)
    log.add("verdict", JUDGE, verdict(p, "revise", (a,)), ps)
    d = log.add("vote", voter("v2"), vote(p.proposal, "v2", criterion="c2"), ps)
    nomani = Log(manifest=None)
    nomani.state = log.state
    nomani.bad("I3", "verdict", JUDGE, verdict(p, "commit", (a, d)), ps)
    log.add("verdict", JUDGE, verdict(p, "commit", (a, d)), ps)


def test_run_manifest_in_same_topic_used_for_quorum() -> None:
    log = Log("r1/_run", manifest=None)
    log.add("manifest", ORCH, ManifestBody(**{**MANIFEST.__dict__, "quorum": 1}))
    p = prop()
    ps = log.add("proposal", STU, p)
    a = log.add("vote", voter("v1"), vote(p.proposal, "v1"), ps)
    log.add("verdict", JUDGE, verdict(p, "commit", (a,)), ps)


def test_i4_commit_and_terminal() -> None:
    log, ps, vs = committed_task()
    p = log.state.proposals[ps].body
    other = ArtifactRef(path="y", sha256="c" * 64, bytes=2)
    log.bad("I4", "commit", JUDGE, CommitBody(proposal=p.proposal, verdict_seq=vs, artifact=other), vs)  # type: ignore[union-attr]
    log.bad("I4", "commit", JUDGE, CommitBody(proposal=p.proposal, verdict_seq=ps, artifact=ART), vs)  # type: ignore[union-attr]
    log.bad("I4", "commit", JUDGE, CommitBody(proposal=p.proposal, verdict_seq=ps, artifact=ART), ps)  # type: ignore[union-attr]
    cs = log.add("commit", JUDGE, CommitBody(proposal=p.proposal, verdict_seq=vs, artifact=ART), vs)  # type: ignore[union-attr]
    assert log.state.commit.seq == cs and log.state.terminal.seq == cs  # type: ignore[union-attr]
    log.bad("I4", "commit", JUDGE, CommitBody(proposal=p.proposal, verdict_seq=vs, artifact=ART), vs)  # type: ignore[union-attr]
    log.bad("I4", "vote", voter("v3"), vote(p.proposal, "v3"), ps)  # type: ignore[union-attr]
    log.bad("I4", "abort", ORCH, AbortBody(attempt=None, reason="late"))
    log.bad("I4", "rubric_patch", HARD, patch(5))
    log.add("note", ORCH, NoteBody(text="after"))


def test_i4_revise_verdict_cannot_commit() -> None:
    log = Log()
    p = prop()
    ps = log.add("proposal", STU, p)
    vs = log.add("verdict", JUDGE, verdict(p, "revise"), ps)
    log.bad("I4", "commit", JUDGE, CommitBody(proposal=p.proposal, verdict_seq=vs, artifact=ART), vs)


def test_i4_adversary_proposal_never_committed() -> None:
    log = Log()
    q = prop(author=ADV)
    qs = log.add("proposal", ADV, q)
    a = log.add("vote", voter("v1"), vote(q.proposal, "v1"), qs)
    b = log.add("vote", voter("v2"), vote(q.proposal, "v2"), qs)
    vs = log.add("verdict", JUDGE, verdict(q, "commit", (a, b)), qs)
    log.bad("I4", "commit", JUDGE, CommitBody(proposal=q.proposal, verdict_seq=vs, artifact=ART), vs)
    assert log.state.commit is None


@pytest.mark.parametrize("terminal", ["reject", "abort"])
def test_reject_and_abort_are_terminal(terminal: str) -> None:
    log = Log()
    if terminal == "reject":
        log.add("reject", JUDGE, RejectBody(attempt="t1@3", reason="exhausted"))
    else:
        log.add("abort", ORCH, AbortBody(attempt=None, reason="dependency_blocked"))
    log.bad("I4", "proposal", STU, prop())
    log.add("note", ORCH, NoteBody(text="ok"))


def test_i5_rubric_patch_never_targets_proposed_attempt() -> None:
    log = Log()
    log.add("rubric_patch", HARD, patch(1))
    log.add("proposal", STU, prop("t1@1"))
    log.add("proposal", ADV, prop("t1@2", ADV))
    assert log.state.max_proposed_attempt() == 2
    log.bad("I5", "rubric_patch", HARD, patch(1))
    log.bad("I5", "rubric_patch", HARD, patch(2))
    log.add("rubric_patch", HARD, patch(3))
    log.add("rubric_patch", HARD, patch(None))


EXPECTED_WRITERS = {
    "manifest": {"orchestrator"}, "intent": {"orchestrator"}, "outcome": {"orchestrator"},
    "abort": {"orchestrator"}, "exploit": {"orchestrator"}, "note": {"orchestrator"},
    "verdict": {"judge"}, "commit": {"judge"}, "reject": {"judge"}, "vote": {"voter"},
    "proposal": {"student", "adversary"}, "rubric_patch": {"hardener"},
}
SAMPLE_BODIES: dict[str, Any] = {
    "manifest": MANIFEST, "intent": IntentBody(action="a", key="k", attempt=None),
    "outcome": OutcomeBody(intent_seq=0, ok=True), "proposal": prop(), "vote": vote(prop().proposal, "v1"),
    "verdict": verdict(prop(), "revise"), "commit": CommitBody(proposal=prop().proposal, verdict_seq=0, artifact=ART),
    "reject": RejectBody(attempt="t1@1", reason="r"), "abort": AbortBody(attempt=None, reason="r"),
    "exploit": ExploitBody(attempt="t1@1", adversary_proposal=prop(author=ADV).proposal, student_proposal=None,
                           gamer="null_model", soft_pref="tie", soft_pass_adversary=False),
    "rubric_patch": patch(None), "note": NoteBody(text="n"),
}


def test_i6_role_kind_authorization() -> None:
    assert {k: set(v) for k, v in WRITERS.items()} == EXPECTED_WRITERS
    for kind in KINDS:
        for role in ROLES:
            if role in EXPECTED_WRITERS[kind]:
                continue
            log = Log()
            log.add("note", ORCH, NoteBody(text="x"))
            log.bad("I6", kind, Author(role=role, name="x"), SAMPLE_BODIES[kind], 0)  # type: ignore[arg-type]


def test_i6_proposal_identity() -> None:
    log = Log()
    log.bad("I6", "proposal", STU, prop(author=ADV))  # student cannot pose as adversary
    log.bad("I6", "proposal", ADV, prop(author=STU))  # nor adversary as student
    log.add("proposal", STU, prop())
    log.bad("I6", "proposal", STU, prop())


def test_seq_and_topic_structure() -> None:
    log = Log()
    log.add("note", ORCH, NoteBody(text="x"))
    e = log.entry("note", ORCH, NoteBody(text="y"))
    with pytest.raises(BusInvariantError, match="SEQ"):
        apply(log.state, Entry(**{**e.__dict__, "seq": 5}))
    with pytest.raises(BusInvariantError, match="SEQ"):
        check(log.state, Entry(**{**e.__dict__, "topic": "r1/t2"}))


def test_helpers() -> None:
    log, ps, vs = committed_task()
    assert list(log.state.proposals) == [ps] and list(log.state.verdicts) == [vs]
    assert log.state.latest_correction() is None
    p2 = prop("t1@2")
    s2 = log.add("proposal", STU, p2)
    corr = StudentCorrection(text="verify the output", attempt="t1@2")
    log.add("verdict", JUDGE, verdict(p2, "revise", correction=corr), s2)
    q = prop("t1@2", ADV)
    qs = log.add("proposal", ADV, q)
    log.add("verdict", JUDGE, verdict(q, "revise", correction=StudentCorrection(text="leak", attempt="t1@2")), qs)
    assert log.state.latest_correction() == corr
    assert log.state.proposal_by_id(p2.proposal).seq == s2  # type: ignore[union-attr]


# ---------------------------------------------------------------- randomized replay determinism

NAMES = ("a", "b")
VOTERS = ("v1", "v2", "v3")
KIND_WEIGHTS = {"intent": 3, "outcome": 3, "proposal": 3, "vote": 8, "verdict": 3, "commit": 2, "reject": 0.2,
                "abort": 0.2, "exploit": 1, "rubric_patch": 1, "note": 1}


def _random_entry(rng: random.Random, log: Log) -> Entry:
    st = log.state
    kind = rng.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]
    role = rng.choice(sorted(EXPECTED_WRITERS[kind]) if rng.random() < 0.85 else ROLES)
    author = Author(role=role, name=rng.choice(VOTERS if role == "voter" else NAMES))  # type: ignore[arg-type]
    attempt = f"t1@{rng.randint(1, 3)}"
    ref = rng.randrange(st.next_seq) if st.next_seq and rng.random() < 0.9 else None
    props, verdicts = list(st.proposals.values()), list(st.verdicts.values())
    body: Any
    if kind == "intent":
        body = IntentBody(action="attempt", key=rng.choice(("k1", "k2", "k3")), attempt=None)
    elif kind == "outcome":
        open_ = st.intents_without_outcome()
        ref = rng.choice(open_).seq if open_ and rng.random() < 0.8 else ref
        body = OutcomeBody(intent_seq=ref or 0, ok=rng.random() < 0.6)
    elif kind == "proposal":
        prole = author.role if author.role in ("student", "adversary") else "student"
        body = prop(attempt, Author(role=prole, name=author.name))
    elif kind == "vote" and props:
        p = rng.choice(props)
        ref = p.seq if rng.random() < 0.9 else ref
        body = vote(p.body.proposal, author.name if role == "voter" else "v1",  # type: ignore[union-attr]
                    rng.choice(("c1", "c2")), rng.choice((True, False, None)))
    elif kind == "verdict" and props:
        p = rng.choice(props)
        ref = p.seq
        cast = [v.seq for v in st.votes_for(p.seq)]
        body = verdict(p.body, rng.choice(("commit", "revise", "reject")),  # type: ignore[arg-type]
                       tuple(cast if rng.random() < 0.5 else rng.sample(cast, rng.randint(0, len(cast)))))
    elif kind == "commit" and verdicts:
        good = [v for v in verdicts if v.body.decision == "commit"]  # type: ignore[union-attr]
        v = rng.choice(good if good and rng.random() < 0.8 else verdicts)
        ref = v.seq
        body = CommitBody(proposal=v.body.proposal, verdict_seq=v.seq, artifact=ART)  # type: ignore[union-attr]
    elif kind == "reject":
        body = RejectBody(attempt=attempt, reason="r")
    elif kind == "abort":
        body = AbortBody(attempt=None, reason="r")
    elif kind == "exploit":
        body = SAMPLE_BODIES["exploit"]
    elif kind == "rubric_patch":
        body = patch(rng.randint(1, 4))
    else:
        kind, body = "note", NoteBody(text=f"n{rng.random()}")
        author = ORCH
    return log.entry(kind, author, body, ref)


def _snapshot(st: BusState) -> tuple[Any, ...]:
    return (st.entries, st.head, dict(st.proposals), dict(st.verdicts), st.commit, st.terminal,
            st.intents_without_outcome(), st.latest_correction(), st.max_proposed_attempt(),
            {s: st.votes_for(s) for s in st.proposals})


@pytest.mark.parametrize("seed", range(25))
def test_random_replay_is_deterministic(seed: int) -> None:
    rng = random.Random(seed)
    log = Log()
    rejected = 0
    for _ in range(120):
        e = _random_entry(rng, log)
        try:
            check(log.state, e, manifest=MANIFEST)
        except BusInvariantError:
            rejected += 1
            continue
        log.state = apply(log.state, e)
        log.log.append(e)
    assert log.log and rejected
    replay = BusState()
    for e in log.log:
        again = Entry.from_json(json.loads(canonical_json(e)))
        assert again == e and again.hash_ok() and again.prev == (replay.head or GENESIS)
        check(replay, again, manifest=MANIFEST)
        replay = apply(replay, again)
    assert replay == log.state
    assert _snapshot(replay) == _snapshot(log.state)
    commits = [e for e in replay.entries if e.kind == "commit"]
    assert len(commits) <= 1
    for c in commits:
        assert replay.entries[replay.entries[c.ref].ref].author.role == "student"  # type: ignore[index]
    if replay.terminal is not None:
        assert all(e.kind == "note" for e in replay.entries[replay.terminal.seq + 1:])
