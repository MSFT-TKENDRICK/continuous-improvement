from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ci_lab.agl.journal import FileRolloutJournal
from ci_lab.agl.scope import RolloutScope, current_rollout
from ci_lab.contracts import RolloutKey, op_id
from ci_lab.testing import MemoryJournal

KEY = RolloutKey("camp-r00", "base", "case-1", trial=0)


def _body(scope: RolloutScope) -> None:
    scope.record_model_request({"model": "m"})
    scope.record_model_request({"model": "m"})
    scope.score("assert.policy", 0.75, suite="harness_policy", violations=[])
    scope.reward(0.75, source="assert", reason="judged")


def test_sync_scope_records_and_succeeds(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    assert current_rollout.get() is None
    with RolloutScope(j, KEY, {"intent": "change"}) as scope:
        assert current_rollout.get() is scope
        _body(scope)
    assert current_rollout.get() is None
    rec = j.load(KEY.rollout_id)
    assert rec is not None and rec.status == "succeeded" and rec.input == {"intent": "change"}
    types = [e["event_type"] for e in rec.events]
    assert types == ["model_request", "model_request", "ci.score", "reward", "ci.metric"]
    reward = rec.events[-2]
    assert reward["data"] == {"value": 0.75, "message": None, "source": "assert", "reason": "judged"}
    assert reward["event_id"] == op_id(KEY.rollout_id, "0", "reward", "reward")
    assert rec.events[2]["data"] == {"name": "assert.policy", "value": 0.75,
                                     "suite": "harness_policy", "violations": []}
    assert rec.events[-1]["data"]["llm_calls"] == 2


def test_async_scope_and_contextvar(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)

    async def main() -> None:
        async with RolloutScope(j, KEY) as scope:
            assert current_rollout.get() is scope
            await asyncio.sleep(0)
            _body(scope)
        assert current_rollout.get() is None

    asyncio.run(main())
    rec = j.load(KEY.rollout_id)
    assert rec is not None and rec.status == "succeeded" and len(rec.events) == 5


def test_exception_and_fail_mark_failed(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    with pytest.raises(RuntimeError), RolloutScope(j, KEY):
        raise RuntimeError("secret detail")
    rec = j.load(KEY.rollout_id)
    assert rec is not None and rec.status == "failed"
    error = next(e for e in rec.events if e["event_type"] == "ci.error")
    assert error["data"] == {"type": "RuntimeError"}
    k2 = RolloutKey("camp-r00", "base", "case-2")
    with RolloutScope(j, k2) as scope:
        scope.fail()
    assert j.load(k2.rollout_id).status == "failed"  # type: ignore[union-attr]


def test_base_exception_leaves_rollout_running(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    with pytest.raises(KeyboardInterrupt), RolloutScope(j, KEY) as scope:
        scope.reward(1.0)
        raise KeyboardInterrupt
    assert current_rollout.get() is None
    rec = j.load(KEY.rollout_id)
    assert rec is not None and not rec.terminal and rec.status == "running"
    with RolloutScope(j, KEY) as scope:  # resume completes it without duplicating the reward
        scope.reward(1.0)
    rec = j.load(KEY.rollout_id)
    assert rec is not None and rec.status == "succeeded" and len(rec.events) == 2


@pytest.mark.parametrize("make", [lambda p: FileRolloutJournal(p), lambda p: MemoryJournal()])
def test_reexecution_is_idempotent(tmp_path: Path, make) -> None:  # type: ignore[no-untyped-def]
    j = make(tmp_path)
    for _ in range(3):  # e.g. checkpoint resume re-running the same case
        with RolloutScope(j, KEY, {"intent": "x"}) as scope:
            _body(scope)
    assert len(j.events(KEY)) == 5
    ids = [e["event_id"] for e in j.events(KEY)]
    assert len(set(ids)) == 5
    if isinstance(j, FileRolloutJournal):
        assert len(j.path(KEY.rollout_id).read_text().splitlines()) == 7  # start + 5 events + finish


def test_new_attempt_has_distinct_ids(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path)
    with RolloutScope(j, KEY) as s0:
        s0.reward(0.0)
    with RolloutScope(j, RolloutKey("camp-r00", "base", "case-1", attempt=1)) as s1:
        s1.reward(1.0)
    rec = j.load(KEY.rollout_id)
    assert rec is not None and rec.attempts == ["0", "1"] and len(rec.events) == 4
    assert rec.status == "succeeded" and rec.finished_attempt == "0"  # terminal state recorded once


def test_scope_proxy_base_url(tmp_path: Path) -> None:
    scope = RolloutScope(FileRolloutJournal(tmp_path), KEY)
    assert scope.proxy_base_url("http://h:1", "train") == (
        f"http://h:1/proxy/rollout/{KEY.rollout_id}/attempt/0/mode/train/openai/v1")
