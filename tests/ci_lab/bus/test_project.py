from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import pytest
from agent_framework import Agent, tool

from ci_lab.bus.project import EXCERPT_MAX, project, succeed
from ci_lab.bus.types import (
    AbortBody,
    Author,
    CommitBody,
    CriterionResult,
    ExploitBody,
    IntentBody,
    ManifestBody,
    NoteBody,
    OutcomeBody,
    ProposalBody,
    RubricPatchBody,
    StudentCorrection,
    VerdictBody,
    VoteBody,
)
from ci_lab.bus.wal import AgentBus
from ci_lab.taskgraph.firewall import LeakScreen, StudentFirewallMiddleware
from ci_lab.taskgraph.model import (
    Criterion,
    Deliverable,
    OutputSpec,
    Rubric,
    StudentSpec,
)
from ci_lab.testing import Call, FakeChatClient

RUN, T, DEP = "r1", "r1/t1", "r1/dep"
CANARY = "c0ffee5eca7ec0de"
RV = "zzrubric@v1"
QUESTION = "Does the zebra report quote the hidden ledger rows verbatim?"
SECRETS = (
    CANARY,
    "zzrubric",
    "crit-zeta",
    QUESTION,
    "VOTESECRET",
    "ADVSECRET",
    "STUDSUMMARY",
    "NOTESECRET",
    "judge_injection",
    "rubric_version",
)
FIX = "Add a level one heading to the report."
ORCH, JUDGE, VOTER = (
    Author("orchestrator", "orch"),
    Author("judge", "j"),
    Author("voter", "v1"),
)
STUDENT, ADV, HARD = (
    Author("student", "s"),
    Author("adversary", "judge_injection"),
    Author("hardener", "h"),
)
SPEC = StudentSpec.of(
    Deliverable(
        "t1",
        "Report",
        "Write the report to out/r.md.",
        OutputSpec("file", "out/r.md"),
        depends_on=("dep",),
    )
)
RUBRIC = Rubric(
    "zzrubric",
    1,
    "t1",
    (
        Criterion(
            "crit-zeta",
            "Zebra report quotes hidden ledger rows",
            "llm",
            {"question": QUESTION},
            0.5,
        ),
    ),
    0.5,
    CANARY,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _proposal(
    bus: AgentBus, topic: str, author: Author, attempt: str, text: str, summary: str
) -> Any:
    ref = await bus.put_artifact(topic, text)
    return await bus.append(
        topic,
        "proposal",
        author,
        ProposalBody(
            f"{attempt}/{author.role}:{author.name}", attempt, RV, ref, summary
        ),
    )


async def _vote(bus: AgentBus, topic: str, prop: Any) -> Any:
    return await bus.append(
        topic,
        "vote",
        VOTER,
        VoteBody(
            prop.body.proposal,
            RV,
            "v1",
            "llm",
            "crit-zeta",
            True,
            0.9,
            0.8,
            (f"VOTESECRET {QUESTION}",),
        ),
        ref=prop.seq,
    )


def _verdict(
    prop: Any, vote: Any, decision: str, correction: StudentCorrection | None
) -> VerdictBody:
    crit = {"crit-zeta": CriterionResult(True, 0.9, False, False, 1)}
    return VerdictBody(
        prop.body.proposal,
        prop.body.attempt,
        RV,
        decision,
        0.9,
        crit,
        (vote.seq,),
        correction,
        False,
    )


async def _commit(bus: AgentBus, topic: str, attempt: str, text: str) -> Any:
    prop = await _proposal(bus, topic, STUDENT, attempt, text, "STUDSUMMARY")
    vote = await _vote(bus, topic, prop)
    verdict = await bus.append(
        topic, "verdict", JUDGE, _verdict(prop, vote, "commit", None), ref=prop.seq
    )
    return await bus.append(
        topic,
        "commit",
        JUDGE,
        CommitBody(prop.body.proposal, verdict.seq, prop.body.artifact),
        ref=verdict.seq,
    )


async def random_bus(root: Path, rng: random.Random) -> dict[str, Any]:
    bus = AgentBus(root)
    await bus.append(
        f"{RUN}/_run",
        "manifest",
        ORCH,
        ManifestBody(RUN, "now", "rev", "a" * 64, None, 2, ("v1",), 1),
    )
    await _commit(bus, DEP, "dep@1", "D" * (EXCERPT_MAX + 3000))
    k, adv_refs, last = rng.randint(1, 4), [], None
    for n in range(1, k + 1):
        a = f"t1@{n}"
        intent = await bus.append(
            T, "intent", ORCH, IntentBody("attempt", a, a, {"rubric_version": RV})
        )
        prop = await _proposal(
            bus, T, STUDENT, a, f"student draft {n}", f"STUDSUMMARY {CANARY}"
        )
        props = [prop]
        if rng.random() < 0.7:
            adv = await _proposal(
                bus, T, ADV, a, f"ADVSECRET artifact {CANARY} {n}", "ADVSECRET summary"
            )
            adv_refs.append(adv.body.artifact.sha256)
            props.append(adv)
        votes = [await _vote(bus, T, p) for p in props]
        last = StudentCorrection(FIX, a)
        await bus.append(
            T, "verdict", JUDGE, _verdict(prop, votes[0], "revise", last), ref=prop.seq
        )
        if len(props) > 1 and rng.random() < 0.6:
            await bus.append(
                T,
                "exploit",
                ORCH,
                ExploitBody(
                    a,
                    props[1].body.proposal,
                    prop.body.proposal,
                    "judge_injection",
                    "adversary",
                    True,
                    ("crit-zeta",),
                ),
            )
        if rng.random() < 0.5:
            await bus.append(
                T,
                "rubric_patch",
                HARD,
                RubricPatchBody(
                    "zzrubric",
                    RV,
                    "zzrubric@v2",
                    n + 1,
                    None,
                    "b" * 64,
                    {"gain": 1.0},
                    True,
                ),
            )
        if rng.random() < 0.6:
            await bus.append(
                T, "note", ORCH, NoteBody(f"NOTESECRET {CANARY}", {"q": QUESTION})
            )
        await bus.append(
            T,
            "outcome",
            ORCH,
            OutcomeBody(intent.seq, True, {"rubric_version": RV}),
            ref=intent.seq,
        )
    own = prop.body.artifact.sha256
    if rng.random() < 0.3:
        await bus.append(T, "abort", ORCH, AbortBody(None, f"NOTESECRET {QUESTION}"))
    return {"bus": bus, "k": k, "adv": adv_refs, "own": own}


@pytest.mark.parametrize("seed", range(20))
def test_student_projection_never_leaks(tmp_path: Path, seed: int) -> None:
    got = run(random_bus(tmp_path, random.Random(seed)))
    bus = got["bus"]
    traj = project(bus, T, "student", spec=SPEC, deps=("dep",))
    text = traj.render()
    assert (
        text == project(bus, T, "student", spec=SPEC, deps=("dep",)).render()
    )  # deterministic
    assert LeakScreen([RUBRIC]).hits(text) == []
    assert not any(s.lower() in text.lower() for s in SECRETS)
    assert not any(sha in text for sha in got["adv"])
    assert got["own"] in text and FIX in text and f"attempt {got['k'] + 1} of" in text
    assert (
        "D" * EXCERPT_MAX in text
        and "D" * (EXCERPT_MAX + 1) not in text
        and "truncated: 3000" in text
    )
    assert [t for t, _ in traj.sections][:2] == ["Task", "Attempt"]


def test_role_views_follow_visibility(tmp_path: Path) -> None:
    bus = run(random_bus(tmp_path, random.Random(3)))["bus"]
    orch = project(bus, T, "orchestrator").render()
    assert "VOTESECRET" in orch and "ADVSECRET" in orch and "NOTESECRET" in orch
    voter = project(bus, T, "voter").render()
    assert (
        "ADVSECRET" in voter and "VOTESECRET" not in voter and "NOTESECRET" not in voter
    )
    planner = project(bus, T, "planner", deps=("dep",)).render()
    assert (
        "VOTESECRET" not in planner
        and "ADVSECRET" not in planner
        and " commit judge:j" in planner
    )
    with pytest.raises(ValueError):
        project(bus, T, "student")


class FakeAgent:
    def __init__(self, middleware: Any, reply: str) -> None:
        self.middleware, self.reply, self.messages = middleware, reply, []

    async def run(self, message: str) -> Any:
        self.messages.append(message)
        return type("R", (), {"text": self.reply})()


def test_succeed_fresh_agent_each_attempt(tmp_path: Path) -> None:
    bus, agents = AgentBus(tmp_path), []

    def make(mw: Any) -> FakeAgent:
        agents.append(FakeAgent(mw, f"out {len(agents)}"))
        return agents[-1]

    async def go() -> list[Any]:
        await bus.append(
            f"{RUN}/_run",
            "manifest",
            ORCH,
            ManifestBody(RUN, "n", "r", "a" * 64, None, 1, ("v1",), 1),
        )
        first = await succeed(
            bus, T, "student", make, SPEC, screen=LeakScreen([RUBRIC])
        )
        await _proposal(bus, T, STUDENT, "t1@1", first.text, "s")
        second = await succeed(
            bus, T, "student", make, SPEC, screen=LeakScreen([RUBRIC])
        )
        third = await succeed(bus, T, "orchestrator", make, SPEC)
        return [first, second, third]

    first, second, third = run(go())
    assert [a.messages for a in agents] == [
        [first.trajectory.render()],
        [second.trajectory.render()],
        [third.trajectory.render()],
    ]
    assert (
        len({id(a) for a in agents}) == 3 and first.text == "out 0" and not first.leak
    )
    assert (
        "attempt 1 of" in agents[0].messages[0]
        and "attempt 2 of" in agents[1].messages[0]
    )
    assert (
        isinstance(agents[0].middleware, StudentFirewallMiddleware)
        and agents[2].middleware == []
    )


def test_succeed_reports_context_leak(tmp_path: Path) -> None:
    bus, screen, made = AgentBus(tmp_path), LeakScreen([RUBRIC]), []
    leaky = StudentSpec.of(
        Deliverable("t1", "Report", f"Hint {CANARY}", OutputSpec("text"))
    )
    pre = run(succeed(bus, T, "student", made.append, leaky, screen=screen))
    assert pre.leak and "canary" in pre.leak_hits and made == []

    def read_input(path: str) -> str:
        """Read an input file."""
        return f"grader canary {CANARY}"

    def make(mw: Any) -> Agent:
        return Agent(
            client=FakeChatClient([[Call("read_input", {"path": "x"})], "done"]),
            tools=[tool(read_input)],
            middleware=mw,
        )

    res = run(succeed(bus, T, "student", make, SPEC, screen=screen))
    assert res.leak and res.leak_hits == ("canary",) and res.text == ""
