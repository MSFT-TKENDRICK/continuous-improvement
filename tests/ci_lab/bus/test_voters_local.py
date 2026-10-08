from __future__ import annotations

import asyncio
import sys
import time
import types
from contextlib import asynccontextmanager

import pytest

from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.types import VoteBody
from ci_lab.bus.voters.local import (
    Ballot,
    CallableVoter,
    CriticChecksVoter,
    DeterministicCheckVoter,
    RulesVoter,
    Voter,
    run_voters,
)
from ci_lab.rules import build_bundle
from ci_lab.rulespec import RuleSpec
from ci_lab.tools.critic_checks import CriticConfig

POOLS = {"cpu": 1, "llm": 3}


def _run(kit, voters, artifact=b"hello", *, rubric=None, pools=None, timeout_s=1.0):
    rubric = rubric or kit.rubric(kit.crit("c1"), kit.crit("c2"))
    pools = pools or ResourcePools(POOLS)
    return asyncio.run(run_voters(voters, kit.proposal(artifact), artifact, rubric, kit.spec(),
                                  pools=pools, timeout_s=timeout_s))


def test_callable_overall_and_per_criterion(kit):
    overall = CallableVoter("whole", lambda p, a, r, s: a == b"hello")
    per = CallableVoter("per", lambda p, a, r, s: {"c1": Ballot(True, 0.9, 0.8, ("ok",))},
                        criteria=["c1", "c2"])

    async def afn(p, a, r, s):
        return False

    asyncv = CallableVoter("async", afn, measure="llm")
    votes = _run(kit, [overall, per, asyncv])
    assert all(isinstance(v, Voter) for v in (overall, per, asyncv))
    got = [(v.voter, v.criterion, v.passed, v.score) for v in votes]
    assert got == [("whole", None, True, 1.0), ("per", "c1", True, 0.9), ("per", "c2", None, None),
                   ("async", None, False, 0.0)]
    assert {v.rubric_version for v in votes} == {"r1@v1"}
    assert votes[2].reasons == ("no ballot",) and votes[3].measure == "llm"


def test_errors_and_timeouts_abstain(kit):
    def boom(p, a, r, s):
        raise RuntimeError("kaput")

    async def slow(p, a, r, s):
        await asyncio.sleep(5)

    votes = _run(kit, [CallableVoter("boom", boom, criteria=["c1", "c2"]), CallableVoter("slow", slow)],
                 timeout_s=0.1)
    assert [(v.voter, v.criterion, v.passed, v.answered) for v in votes] == [
        ("boom", "c1", None, False), ("boom", "c2", None, False), ("slow", None, None, False)]
    assert "RuntimeError: kaput" in votes[0].reasons[0] and "timeout" in votes[2].reasons[0]


def test_timeout_starts_after_pool_admission(kit):
    pools = ResourcePools(POOLS)

    async def work(p, a, r, s):
        await asyncio.sleep(0.05)
        return True

    async def main():
        held = asyncio.Event()

        async def hog():
            async with pools.acquire("cpu"):
                held.set()
                await asyncio.sleep(0.4)

        h = asyncio.create_task(hog())
        await held.wait()
        t0 = time.monotonic()
        votes = await run_voters([CallableVoter("v", work, pool="cpu")], kit.proposal(b"x"), b"x",
                                 kit.rubric(kit.crit("c1")), kit.spec(), pools=pools, timeout_s=0.2)
        await h
        return votes, time.monotonic() - t0

    votes, waited = asyncio.run(main())
    assert waited >= 0.3 and votes[0].passed is True  # queued longer than timeout_s, still voted
    assert pools.stats()["cpu"].granted == 2 and pools.stats()["cpu"].in_use == 0


def test_admission_hook_precedes_timeout(kit):
    order: list[str] = []

    class Gated(CallableVoter):
        @asynccontextmanager
        async def admission(self):
            order.append("admit")
            await asyncio.sleep(0.3)
            yield
            order.append("release")

    async def work(p, a, r, s):
        order.append("vote")
        return True

    votes = _run(kit, [Gated("g", work, pool="llm")], timeout_s=0.1)
    assert votes[0].passed is True and order == ["admit", "vote", "release"]


def test_concurrency_bounded_by_pool(kit):
    async def work(p, a, r, s):
        await asyncio.sleep(0.15)
        return True

    pools = ResourcePools(POOLS)
    t0 = time.monotonic()
    votes = _run(kit, [CallableVoter(f"v{i}", work, pool="llm") for i in range(3)], pools=pools)
    assert time.monotonic() - t0 < 0.4 and all(v.passed for v in votes)
    assert pools.stats()["llm"].peak_in_use == 3
    _run(kit, [CallableVoter(f"v{i}", work, pool="cpu") for i in range(3)], pools=pools)
    assert pools.stats()["cpu"].peak_in_use == 1


def test_off_target_and_duplicate_votes_are_dropped(kit):
    class Sloppy(CallableVoter):
        async def vote(self, proposal, artifact, rubric, spec):
            mk = lambda c, ok, voter="sloppy": VoteBody(
                proposal=proposal.proposal, rubric_version=proposal.rubric_version, voter=voter,
                measure="deterministic", criterion=c, passed=ok, score=None, confidence=None)
            return [mk("c1", True), mk("c1", False), mk("zz", True), mk("c2", True, "other")]

    votes = _run(kit, [Sloppy("sloppy", lambda *a: None, criteria=["c1", "c2"])])
    assert [(v.criterion, v.passed, v.reasons) for v in votes] == [("c1", True, ()), ("c2", None, ("no vote",))]


def test_run_voters_rejects_bad_config(kit):
    v = CallableVoter("v", lambda *a: True)
    with pytest.raises(ValueError, match="rubric"):
        asyncio.run(run_voters([v], kit.proposal(b"x", "r1@v2"), b"x", kit.rubric(), kit.spec(),
                               pools=ResourcePools(POOLS), timeout_s=1))
    with pytest.raises(ValueError, match="duplicate"):
        _run(kit, [v, CallableVoter("v", lambda *a: True)])
    with pytest.raises(ValueError, match="unknown resource pool"):
        _run(kit, [CallableVoter("w", lambda *a: True, pool="gpu")])
    assert _run(kit, [CallableVoter("w", lambda *a: True, criteria=["nope"])]) == []

# ---------------------------------------------------------------- deterministic / critic / rules


def _det(kit, *checks, artifact=b"hello", voter=None, threshold=0.5):
    crits = [kit.crit(f"k{i}", check=chk, threshold=threshold) for i, chk in enumerate(checks)]
    rubric = kit.rubric(*crits, kit.crit("soft", "s1", {"question": "q?", "type": "noul"}))
    votes = _run(kit, [voter or DeterministicCheckVoter()], artifact, rubric=rubric, timeout_s=30)
    assert [v.criterion for v in votes] == [c.id for c in crits]
    return [(v.passed, v.score) for v in votes], votes


def test_regex_json_schema_file_exists(kit):
    schema = {"type": "object", "required": ["a"]}
    got, votes = _det(kit, {"kind": "regex", "pattern": "^hel"}, {"kind": "regex", "pattern": "nope"},
                      {"kind": "regex", "pattern": "nope", "negate": True}, {"kind": "json_schema", "schema": schema},
                      {"kind": "file_exists", "path": "out/answer.txt"}, {"kind": "file_exists", "path": "x.txt"},
                      {"kind": "telepathy"})
    assert got == [(True, 1.0), (False, 0.0), (True, 1.0), (False, 0.0), (True, 1.0), (False, 0.0), (None, None)]
    assert "invalid JSON" in votes[3].reasons[0] and "unknown check kind" in votes[6].reasons[0]
    got, votes = _det(kit, {"kind": "json_schema", "schema": schema}, {"kind": "json_schema", "schema": schema},
                      artifact=b'{"a": 1}')
    assert got == [(True, 1.0), (True, 1.0)]
    got, votes = _det(kit, {"kind": "json_schema", "schema": schema}, artifact=b'{"b": 1}')
    assert got == [(False, 0.0)] and "'a' is a required property" in votes[0].reasons


def test_command_check(kit):
    read = "import pathlib,sys; sys.exit(pathlib.Path('out/answer.txt').read_text() != 'hello')"
    got, votes = _det(kit, {"kind": "command", "argv": ["python", "-c", read]},
                      {"kind": "command", "argv": ["python", "-c", "print('bad'); raise SystemExit(3)"]},
                      {"kind": "command", "argv": ["cmd", "/c", "echo", "hi"]},
                      {"kind": "command", "argv": ["python", "-c", "import time; time.sleep(30)"], "timeout_s": 0.5})
    assert got == [(True, 1.0), (False, 0.0), (None, None), (False, 0.0)]
    assert votes[1].reasons == ("exit 3: bad",) and "not in allowlist" in votes[2].reasons[0]
    assert "timed out" in votes[3].reasons[0]
    got, _ = _det(kit, {"kind": "command", "argv": ["python", "-c", "pass"]},
                  voter=DeterministicCheckVoter(allowlist={"git"}))
    assert got == [(None, None)]


def test_python_check(kit, monkeypatch):
    mod = types.ModuleType("ci_lab._vote_fixture")
    mod.yes = lambda a, s: a.startswith(b"hel") and s.id == "t1"
    mod.why = lambda a, s: (False, "too short")
    mod.score = lambda a, s: 0.7

    async def boom(a, s):
        raise RuntimeError("nope")

    mod.boom, mod.weird = boom, (lambda a, s: "maybe")
    monkeypatch.setitem(sys.modules, "ci_lab._vote_fixture", mod)
    names = ["yes", "why", "score", "boom", "weird"]
    got, votes = _det(kit, *({"kind": "python", "callable": f"ci_lab._vote_fixture.{n}"} for n in names),
                      {"kind": "python", "callable": "os.system"}, threshold=0.8)
    assert got == [(True, 1.0), (False, 0.0), (False, 0.7), (None, None), (None, None), (None, None)]
    assert votes[1].reasons == ("too short",) and "RuntimeError: nope" in votes[3].reasons[0]
    assert "not under ci_lab" in votes[5].reasons[0]


def test_critic_checks_and_rules_voters(kit):
    cfg = CriticConfig(surface_globs=["out/*"], component_globs={})
    rule = RuleSpec.model_validate({"id": "t.nosecret", "version": 1, "rung": "R3", "on": "response", "target": "*",
                                    "action": "warn", "template": "response.blocked", "slots": {},
                                    "require": {"kind": "not", "of": {"kind": "text", "matches": "SECRET"}}})
    minor = rule.model_copy(update={"id": "t.minor", "severity": "minor"})
    voters = [CriticChecksVoter(cfg), RulesVoter(build_bundle([rule], [])),
              RulesVoter(build_bundle([minor], []), name="soft-rules")]
    ok = _run(kit, voters, b"plain answer")
    assert [(v.voter, v.criterion, v.passed) for v in ok] == [
        ("critic-checks", None, True), ("rules", None, True), ("soft-rules", None, True)]
    bad = _run(kit, voters, b"the SECRET rubric")
    assert [v.passed for v in bad] == [False, False, True]
    assert any("denylist" in r for r in bad[0].reasons) and bad[1].reasons[0].startswith("t.nosecret [major]")
    assert bad[2].reasons[0].startswith("t.minor [minor]")
