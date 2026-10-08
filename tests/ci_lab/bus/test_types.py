from __future__ import annotations

import json
from typing import Any

import pytest

from ci_lab.bus.types import (
    BODY_TYPES,
    GENESIS,
    KINDS,
    ROLES,
    AbortBody,
    ArtifactRef,
    Author,
    BodyError,
    CommitBody,
    CriterionResult,
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
    entry_hash,
    visibility,
)

H = "a" * 64
ART = ArtifactRef(path="ab/" + H, sha256=H, bytes=12)
P = "t1@1/student:stu"
A = "t1@1/adversary:null_model"
RV = "rub@v1"

SAMPLES: dict[str, Any] = {
    "manifest": ManifestBody(run="r1", created="2026-01-01T00:00:00Z", code_rev="abc", config_sha256=H,
                             graph_sha256=None, max_parallel=4, voters=("v1", "v2"), quorum=2, extra={"k": "v"}),
    "intent": IntentBody(action="attempt", key="t1@1", attempt="t1@1", detail={"n": [1, 2.5, None]}),
    "outcome": OutcomeBody(intent_seq=0, ok=True, detail={}),
    "proposal": ProposalBody(proposal=P, attempt="t1@1", rubric_version=RV, artifact=ART, summary="s"),
    "vote": VoteBody(proposal=P, rubric_version=RV, voter="v1", measure="deterministic", criterion="c1",
                     passed=True, score=1, confidence=0.5, reasons=["ok"]),  # type: ignore[arg-type]
    "verdict": VerdictBody(proposal=P, attempt="t1@1", rubric_version=RV, decision="revise", score=0.25,
                           criteria={"c1": CriterionResult(passed=False, score=0.0, required=True, oracle=True,
                                                            votes=1)},
                           votes=(3,), correction=StudentCorrection(text="fix it", attempt="t1@1"), escalated=False),
    "commit": CommitBody(proposal=P, verdict_seq=4, artifact=ART),
    "reject": RejectBody(attempt="t1@3", reason="attempts exhausted"),
    "abort": AbortBody(attempt=None, reason="dependency_blocked"),
    "exploit": ExploitBody(attempt="t1@1", adversary_proposal=A, student_proposal=P, gamer="null_model",
                           soft_pref="adversary", soft_pass_adversary=True, oracle_invalid=("c1",)),
    "rubric_patch": RubricPatchBody(rubric_id="rub", from_version="rub@v1", to_version="rub@v2",
                                    applies_from_attempt=2, applies_from_epoch=None, diff_sha256=H,
                                    metrics={"held_out": 1}, accepted=True),  # type: ignore[dict-item]
    "note": NoteBody(text="torn_tail_repaired", data={"é": "ü"}),
}


def test_every_kind_has_a_sample_and_body_type() -> None:
    assert set(SAMPLES) == set(KINDS) == set(BODY_TYPES)
    for kind, body in SAMPLES.items():
        assert type(body) is BODY_TYPES[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_body_round_trip(kind: str) -> None:
    body = SAMPLES[kind]
    data = json.loads(canonical_json(body))
    again = BODY_TYPES[kind].from_json(data)  # type: ignore[index]
    assert again == body
    assert canonical_json(again) == canonical_json(body)


def test_normalization() -> None:
    vote = SAMPLES["vote"]
    assert vote.reasons == ("ok",) and isinstance(vote.score, float)
    with pytest.raises(TypeError):
        SAMPLES["manifest"].extra["k"] = "x"  # type: ignore[index]
    assert list(IntentBody(action="a", key="k", attempt=None, detail={"b": 1, "a": 2}).detail) == ["a", "b"]


@pytest.mark.parametrize("bad", [
    lambda d: d.update(extra_field=1),
    lambda d: d.pop("voter"),
    lambda d: d.update(score=1.5),
    lambda d: d.update(confidence=-0.1),
    lambda d: d.update(score="0.5"),
    lambda d: d.update(passed=1),
    lambda d: d.update(measure="vibes"),
    lambda d: d.update(reasons="ok"),
    lambda d: d.update(proposal="t1@1/student"),
    lambda d: d.update(rubric_version="rub"),
    lambda d: d.update(voter="a b"),
])
def test_vote_strictness(bad: Any) -> None:
    data = SAMPLES["vote"].to_json()
    bad(data)
    with pytest.raises(BodyError):
        VoteBody.from_json(data)


def test_body_specific_validation() -> None:
    with pytest.raises(BodyError):
        ProposalBody(proposal=P, attempt="t1@2", rubric_version=RV, artifact=ART, summary="s")
    with pytest.raises(BodyError):
        ProposalBody(proposal=P, attempt="t1@1", rubric_version=RV, artifact=ART, summary="x" * 401)
    with pytest.raises(BodyError):
        ArtifactRef(path="../x", sha256=H, bytes=1)
    with pytest.raises(BodyError):
        ArtifactRef(path="x", sha256="A" * 64, bytes=1)
    with pytest.raises(BodyError):
        ManifestBody(run="r1", created="", code_rev="", config_sha256=H, graph_sha256=None, max_parallel=1,
                     voters=("v", "v"), quorum=1)
    with pytest.raises(BodyError):
        ExploitBody(attempt="t1@1", adversary_proposal=P, student_proposal=None, gamer="g", soft_pref="tie",
                    soft_pass_adversary=False)
    with pytest.raises(BodyError):
        RubricPatchBody(rubric_id="rub", from_version="rub@v1", to_version="other@v2", applies_from_attempt=2,
                        applies_from_epoch=None, diff_sha256=H, metrics={}, accepted=True)
    with pytest.raises(BodyError):
        VerdictBody.from_json({**SAMPLES["verdict"].to_json(), "votes": [1, 1]})
    with pytest.raises(BodyError):
        StudentCorrection(text="x" * 801, attempt="t1@1")
    with pytest.raises(BodyError):
        NoteBody(text="x", data={"f": float("nan")})
    with pytest.raises(BodyError):
        NoteBody(text="x", data={"f": object()})
    with pytest.raises(BodyError):
        Author(role="boss", name="x")  # type: ignore[arg-type]


def _entry(kind: str = "note", seq: int = 1, **kw: Any) -> Entry:
    base: dict[str, Any] = {"seq": seq, "topic": "r1/t1", "kind": kind, "author": Author(role="orchestrator", name="o"),
                            "ref": None, "body": SAMPLES[kind], "ts": "2026-01-01T00:00:00Z", "prev": GENESIS}
    return Entry(**(base | kw))


def test_entry_round_trip_and_hash() -> None:
    e = _entry().sealed()
    assert e.hash_ok() and e.hash == entry_hash(e) == entry_hash(e.to_json())
    line = canonical_json(e)
    assert "\\u" not in line and '"é":"ü"' in line
    again = Entry.from_json(json.loads(line))
    assert again == e and again.hash_ok()
    assert entry_hash(_entry()) == e.hash  # hash excludes the hash field itself
    assert entry_hash(_entry(ts="2026-01-02T00:00:00Z")) != e.hash
    assert not _entry(hash=e.hash, ts="later").hash_ok()


def test_entry_strictness() -> None:
    with pytest.raises(BodyError):
        _entry(kind="note", body=SAMPLES["vote"])
    with pytest.raises(BodyError):
        _entry(ref=1)
    with pytest.raises(BodyError):
        _entry(topic="/abs")
    with pytest.raises(BodyError):
        _entry(prev="nothex")
    with pytest.raises(BodyError):
        _entry(seq=True)
    data = _entry().to_json()
    with pytest.raises(BodyError):
        Entry.from_json({**data, "visibility": ["student"]})
    with pytest.raises(BodyError):
        Entry.from_json({**data, "kind": "vote"})


EXPECTED_VIS = {
    "manifest": {"orchestrator", "hardener", "judge"},
    "intent": {"orchestrator", "hardener", "judge"},
    "outcome": {"orchestrator", "hardener", "judge"},
    "note": {"orchestrator", "hardener", "judge"},
    "proposal": {"orchestrator", "voter", "judge", "hardener"},
    "vote": {"orchestrator", "judge", "hardener"},
    "verdict": {"orchestrator", "judge", "hardener"},
    "commit": set(ROLES),
    "reject": {"orchestrator", "judge", "hardener", "planner"},
    "abort": {"orchestrator", "judge", "hardener", "planner"},
    "exploit": {"orchestrator", "judge", "hardener"},
    "rubric_patch": {"orchestrator", "judge", "hardener"},
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("role", ROLES)
def test_visibility_table(kind: str, role: str) -> None:
    vis = visibility(_entry(kind=kind, author=Author(role=role, name="x")))  # type: ignore[arg-type]
    assert vis == EXPECTED_VIS[kind]
    if kind != "commit":
        assert "student" not in vis and "adversary" not in vis
