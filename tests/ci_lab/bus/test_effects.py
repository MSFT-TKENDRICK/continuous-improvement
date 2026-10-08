from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ci_lab.bus.effects import effect, in_flight
from ci_lab.bus.types import Author, IntentBody
from ci_lab.bus.wal import AgentBus

T = "r1/t1"
ORCH = Author(role="orchestrator", name="orch")


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


async def do(bus: AgentBus, ran: list[str], key: str = "k", *, fail: bool = False, reconcile=None,  # type: ignore[no-untyped-def]
             via_method: bool = False):
    cm = bus.effect(T, "push", key, ORCH, reconcile=reconcile) if via_method else effect(
        bus, T, "push", key, ORCH, {"branch": "b"}, reconcile=reconcile)
    async with cm as eff:
        if not eff.skipped:
            ran.append(key)
            await asyncio.sleep(0.02)
            if fail:
                raise ValueError("boom")
            eff.set_result({"sha": "abc"})
    return eff


def kinds(bus: AgentBus) -> list[str]:
    return [e.kind for e in bus.read(T)]


def test_runs_once_then_skips(tmp_path: Path) -> None:
    bus, ran = AgentBus(tmp_path), []
    first = run(do(bus, ran))
    again = run(do(AgentBus(tmp_path), ran, via_method=True))
    assert ran == ["k"] and not first.skipped and again.skipped and dict(again.result) == {"sha": "abc"}
    assert again.intent == first.intent and kinds(bus) == ["intent", "outcome"]
    assert dict(bus.read(T)[0].body.detail) == {"branch": "b"} and in_flight(bus, T) == ()  # type: ignore[union-attr]
    with pytest.raises(RuntimeError):
        again.set_result({})


def test_failure_records_outcome_then_retry(tmp_path: Path) -> None:
    bus, ran = AgentBus(tmp_path), []
    with pytest.raises(ValueError, match="boom"):
        run(do(bus, ran, fail=True))
    failed = bus.read(T)[1].body
    assert not failed.ok and failed.detail["error"] == "ValueError: boom"  # type: ignore[union-attr]
    assert not run(do(bus, ran)).skipped and ran == ["k", "k"]
    assert kinds(bus) == ["intent", "outcome", "intent", "outcome"] and bus.read(T)[3].body.ok  # type: ignore[union-attr]


def crash_after_intent(bus: AgentBus) -> int:
    e = run(bus.append(T, "intent", ORCH, IntentBody(action="push", key="k", attempt=None)))
    assert in_flight(bus, T) == (e,)
    return e.seq


def test_crash_between_intent_and_outcome_reruns_reusing_intent(tmp_path: Path) -> None:
    bus, ran, seen = AgentBus(tmp_path), [], []

    async def nothing(key: str) -> None:
        seen.append(key)

    seq = crash_after_intent(bus)
    eff = run(do(bus, ran, reconcile=nothing))
    assert ran == seen == ["k"] and eff.intent.seq == seq  # type: ignore[union-attr]
    assert kinds(bus) == ["intent", "outcome"] and bus.read(T)[1].ref == seq and in_flight(bus, T) == ()


def test_crash_between_intent_and_outcome_reconciles(tmp_path: Path) -> None:
    bus, ran = AgentBus(tmp_path), []

    async def found(key: str) -> dict[str, str]:
        return {"sha": "already"}

    crash_after_intent(bus)
    eff = run(do(bus, ran, reconcile=found))
    assert ran == [] and eff.skipped and eff.reconciled and dict(eff.result) == {"sha": "already"}
    assert kinds(bus) == ["intent", "outcome"] and dict(run(do(bus, ran)).result) == {"sha": "already"}


def test_concurrent_same_key_asyncio_runs_once(tmp_path: Path) -> None:
    bus, ran = AgentBus(tmp_path), []

    async def many() -> list:  # type: ignore[type-arg]
        return await asyncio.gather(*(do(bus, ran, k) for k in ["k"] * 8 + ["j"] * 4))

    effs = run(many())
    assert sorted(ran) == ["j", "k"] and sum(not e.skipped for e in effs) == 2
    assert all(dict(e.result) == {"sha": "abc"} for e in effs) and kinds(bus).count("intent") == 2


_CHILD = """
import asyncio, sys
from ci_lab.bus.types import Author
from ci_lab.bus.wal import AgentBus
async def main():
    async with AgentBus(sys.argv[1]).effect("r1/t1", "push", "k", Author(role="orchestrator", name="o")) as eff:
        if not eff.skipped:
            with open(sys.argv[2], "a") as fh:
                fh.write("ran\\n")
            await asyncio.sleep(1.0)
            eff.set_result({"n": 1})
    assert dict(eff.result) == {"n": 1}
asyncio.run(main())
"""


def test_concurrent_same_key_processes_run_once(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    procs = [subprocess.Popen([sys.executable, "-c", _CHILD, str(tmp_path / "bus"), str(marker)]) for _ in range(3)]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    assert marker.read_text() == "ran\n"
    assert [e.kind for e in AgentBus(tmp_path / "bus").read(T)] == ["intent", "outcome"]
