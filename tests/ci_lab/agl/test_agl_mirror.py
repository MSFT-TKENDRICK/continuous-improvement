from __future__ import annotations

import logging
from pathlib import Path

from ci_lab.agl.client import AglClient
from ci_lab.agl.journal import FileRolloutJournal
from ci_lab.agl.mirror import MODEL_REQUEST_FIELDS, MirroringJournal, model_request_data, model_request_recorder
from ci_lab.agl.scope import RolloutScope
from ci_lab.contracts import RolloutKey, op_id

KEY = RolloutKey("camp-r00", "base", "case-1", trial=0)


def _mirror(tmp_path: Path, fake) -> MirroringJournal:  # type: ignore[no-untyped-def]
    return MirroringJournal(FileRolloutJournal(tmp_path, fsync=False),
                            AglClient("http://agl.test", fake.key, transport=fake.transport()))


def _run(j: MirroringJournal, key: RolloutKey = KEY) -> None:
    j.start(key, {"intent": "refund"})
    j.event(key, "reward", {"value": 1.0}, event_id="e1")
    j.event(key, "reward", {"value": 1.0}, event_id="e1")  # duplicate: not re-posted
    j.event(key, "ci.score", {"name": "s", "value": 0.5}, event_id="e2")
    j.finish(key, "succeeded")


def test_mirrors_journal_first(tmp_path: Path, fake_agl) -> None:  # type: ignore[no-untyped-def]
    j = _mirror(tmp_path, fake_agl)
    _run(j)
    assert fake_agl.state(KEY.rollout_id) == "succeeded"
    assert [e.data["ci_event_id"] for e in fake_agl.all_events(KEY.rollout_id)] == ["e1", "e2"]
    r = fake_agl.rollouts[KEY.rollout_id]
    assert r.input == {"intent": "refund", "data_id": "case-1"} and r.is_train is False
    assert r.metadata.model_dump()["variant"] == "base"  # type: ignore[union-attr]
    assert not j.dirty and not j.offline
    assert [e["event_id"] for e in j.events(KEY)] == ["e1", "e2"]  # journal has no ci_event_id stamp
    assert "ci_event_id" not in j.events(KEY)[0]["data"]


def test_offline_then_sync_without_duplicates(tmp_path: Path, fake_agl, caplog) -> None:  # type: ignore[no-untyped-def]
    j = _mirror(tmp_path, fake_agl)
    j.start(KEY, {})
    j.event(KEY, "reward", {"value": 1.0}, event_id="e1")
    fake_agl.down = True
    with caplog.at_level(logging.WARNING, logger="ci_lab.agl.mirror"):
        j.event(KEY, "ci.score", {"name": "s"}, event_id="e2")
    assert j.offline and j.dirty == {KEY.rollout_id}
    assert fake_agl.key not in caplog.text
    calls = len(fake_agl.requests)
    j.event(KEY, "ci.score", {"name": "t"}, event_id="e3")
    j.finish(KEY, "succeeded")
    assert len(fake_agl.requests) == calls  # offline: no stalls on a dead server
    assert [e["event_id"] for e in j.events(KEY)] == ["e1", "e2", "e3"]  # journal still written
    assert not j.sync().ok  # still down
    fake_agl.down = False
    report = j.sync()
    assert report.ok and report.events_posted == 2 and not j.dirty and not j.offline
    assert [e.data["ci_event_id"] for e in fake_agl.all_events(KEY.rollout_id)] == ["e1", "e2", "e3"]
    assert fake_agl.state(KEY.rollout_id) == "succeeded"
    assert j.sync([KEY.rollout_id]).events_posted == 0  # idempotent


def test_sync_all_rollouts_to_fresh_server(tmp_path: Path, fake_agl) -> None:  # type: ignore[no-untyped-def]
    j = MirroringJournal(FileRolloutJournal(tmp_path, fsync=False))  # client=None: journal only
    _run(j)
    k1 = RolloutKey("camp-r00", "base", "case-2", attempt=1)
    j.start(RolloutKey("camp-r00", "base", "case-2"), {})
    j.event(RolloutKey("camp-r00", "base", "case-2"), "reward", {"value": 0.0}, event_id="a0")
    j.start(k1, {})
    j.event(k1, "reward", {"value": 1.0}, event_id="a1")
    j.finish(k1, "failed")
    assert j.sync().rollouts == 0
    j.client = AglClient("http://agl.test", fake_agl.key, transport=fake_agl.transport())
    report = j.sync(all_rollouts=True)
    assert report.ok and report.created == 2 and report.events_posted == 4
    assert fake_agl.state(KEY.rollout_id) == "succeeded" and fake_agl.state(k1.rollout_id) == "failed"
    assert {a: [e.data["ci_event_id"] for e in evs] for a, evs in fake_agl.events[k1.rollout_id].items()} == {
        "0": ["a0"], "1": ["a1"]}
    assert fake_agl.rollouts[k1.rollout_id].status.last_attempt_id == "1"


def test_import_server_events_idempotent(tmp_path: Path, fake_agl) -> None:  # type: ignore[no-untyped-def]
    j = _mirror(tmp_path, fake_agl)
    j.start(KEY, {})
    j.event(KEY, "reward", {"value": 1.0}, event_id="e1")
    fake_agl.add_proxy_event(KEY.rollout_id, "0", {"model": "m", "usage": {"prompt_tokens": 3},
                                                   "routed_experts": [1, 2]})
    assert j.import_server_events(KEY) == 1
    assert j.import_server_events(KEY) == 0
    mr = [e for e in j.events(KEY) if e["event_type"] == "model_request"]
    assert len(mr) == 1 and "routed_experts" not in mr[0]["data"] and mr[0]["data"]["model"] == "m"
    assert MirroringJournal(j.journal).import_server_events(KEY) == 0  # no client


def test_model_request_data_field_set() -> None:
    data = model_request_data({"served_model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}],
                               "content": "hello", "input_tokens": 10, "output_tokens": 4,
                               "latency_ms": 12.5, "finish_reason": "stop", "turn": 3})
    assert set(MODEL_REQUEST_FIELDS) <= set(data)
    assert data["model"] == "gpt-5-mini" and data["status"] == "ok" and data["http_status"] == 200
    assert data["usage"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    assert data["request"] == {"messages": [{"role": "user", "content": "hi"}]}
    assert data["response"]["choices"][0]["message"]["content"] == "hello"
    assert data["server"] == {"model": "gpt-5-mini", "endpoint": "copilot", "version": None}
    assert data["ci"] == {"turn": 3, "served_model": "gpt-5-mini"}
    err = model_request_data({"model": "m", "error": "boom", "retry_count": 2})
    assert err["status"] == "error" and err["http_status"] == 500 and err["response"]["error"] == "boom"
    assert err["retry_count"] == 2 and err["usage"] is None
    passthrough = model_request_data({"model": "m", "response": {"model": "m-2024", "choices": [
        {"finish_reason": "length"}], "usage": {"prompt_tokens": 1}}})
    assert passthrough["finish_reason"] == "length" and passthrough["usage"] == {"prompt_tokens": 1}


def test_model_request_recorder(tmp_path: Path) -> None:
    j = FileRolloutJournal(tmp_path, fsync=False)
    rec = model_request_recorder()
    rec({"model": "m"})  # outside any scope: dropped
    with RolloutScope(j, KEY) as scope:
        rec({"model": "m", "name": "turn-1"})
        rec({"model": "m", "name": "turn-1"})  # same logical name: deduped
        rec({"model": "m"})
        model_request_recorder(scope)({"model": "m2"})
    evs = [e for e in j.events(KEY) if e["event_type"] == "model_request"]
    assert [e["data"]["model"] for e in evs] == ["m", "m", "m2"]
    assert evs[0]["event_id"] == op_id(KEY.rollout_id, "0", "model_request", "turn-1")
