from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ci_lab.cli import main
from ci_lab.publish.deferred import DeferredPublisher, read_requests, replay_deferred
from ci_lab.publish.github import PublishError


class MemLedger:
    def __init__(self) -> None:
        self.files: dict[str, Any] = {}
        self.commits: list[tuple[str, list[str]]] = []

    def read_json(self, rel: str) -> Any:
        return self.files.get(rel)

    def write_json(self, rel: str, obj: Any) -> None:
        self.files[rel] = obj

    def read_jsonl(self, rel: str) -> list[dict[str, Any]]:
        return list(self.files.get(rel, []))

    def append_jsonl(self, rel: str, obj: dict[str, Any], *, key: str) -> bool:
        rows = self.files.setdefault(rel, [])
        if any(r.get(key) == obj[key] for r in rows):
            return False
        rows.append(dict(obj))
        return True

    def commit(self, message: str, paths: list[str]) -> str:
        self.commits.append((message, list(paths)))
        return f"c{len(self.commits)}"


class StackPublisher:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def publish_round(self, *, eid: str, winner: str | None, heads: dict[str, str], stack: dict[str, Any],
                      title: str, body: str) -> dict[str, Any]:
        self.calls.append({"eid": eid, "winner": winner, "stack": stack})
        layers = list(stack.get("layers") or [])
        if winner:
            layers.append({"eid": eid, "head": heads[winner], "pr": 100 + len(layers)})
        return {"stack": {"layers": layers, "stack_number": 7 if layers else None}}


def test_deferred_publisher_queues_once_and_keeps_stack(tmp_path: Path) -> None:
    pub = DeferredPublisher(tmp_path / "q" / "req.jsonl")
    stack = {"layers": [{"eid": "c-r01"}], "stack_number": 3}
    for _ in range(2):
        out = pub.publish_round(eid="c-r02", winner="v1", heads={"v1": "a" * 40}, stack=stack, title="t", body="b")
        assert out == {"stack": stack, "deferred": True}
    assert [r["eid"] for r in read_requests(pub.path)] == ["c-r02"]
    with pytest.raises(PublishError):
        pub.land(stack)


def test_replay_publishes_in_order_and_is_idempotent(tmp_path: Path) -> None:
    q = DeferredPublisher(tmp_path / "req.jsonl")
    q.publish_round(eid="c-r01", winner="v2", heads={"v2": "a" * 40}, stack={}, title="t1", body="b1")
    q.publish_round(eid="c-r02", winner=None, heads={}, stack={}, title="t2", body="b2")
    q.publish_round(eid="c-r03", winner="v1", heads={"v1": "b" * 40}, stack={}, title="t3", body="b3")
    ledger, pub = MemLedger(), StackPublisher()
    out = replay_deferred(q.path, cid="c", ledger=ledger, publisher=pub)
    assert [o["layers"] for o in out] == [1, 1, 2]
    # each round sees the stack left by the previous one (layers stack on each other)
    assert [len(c["stack"].get("layers") or []) for c in pub.calls] == [0, 1, 1]
    assert [r["eid"] for r in ledger.read_json("campaigns/c/stack.json")["layers"]] == ["c-r01", "c-r03"]
    assert len(ledger.commits) == 3
    again = replay_deferred(q.path, cid="c", ledger=ledger, publisher=pub)
    assert all(o.get("skipped") for o in again) and len(pub.calls) == 3


def test_replay_rejects_foreign_or_malformed_requests(tmp_path: Path) -> None:
    path = tmp_path / "req.jsonl"
    DeferredPublisher(path).publish_round(eid="other-r01", winner=None, heads={}, stack={}, title="t", body="b")
    with pytest.raises(PublishError, match="does not belong"):
        replay_deferred(path, cid="c", ledger=MemLedger(), publisher=StackPublisher())
    path.write_text(json.dumps({"eid": "c-r01", "heads": [], "title": "t", "body": "b"}) + "\n", encoding="utf-8")
    with pytest.raises(PublishError, match="malformed"):
        read_requests(path)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_cli_defer_then_publish(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = ["--profile", "fake", "--run-dir", str(tmp_path), "--dry-run-publish"]
    req = tmp_path / "deferred.jsonl"
    assert _run(capsys, "campaign", "new", "dp-camp", *common, "--hyper", "aa_repeats=2")[0] == 0
    assert _run(capsys, "campaign", "calibrate", "dp-camp", *common)[0] == 0
    code, out = _run(capsys, "campaign", "run", "dp-camp", *common, "--rounds", "1", "--defer-publish", str(req))
    assert code == 0 and out["rounds"][0]["winner"] == "v1"
    stack_path = tmp_path / "_fake" / "experiments" / "campaigns" / "dp-camp" / "stack.json"
    assert not stack_path.exists()  # deferred: the ledger stack is untouched by the model-driven job
    assert [r["eid"] for r in read_requests(req)] == ["dp-camp-r01"]
    code, out = _run(capsys, "campaign", "publish", "dp-camp", *common, "--requests", str(req))
    assert code == 0 and out["published"] == [{"eid": "dp-camp-r01", "layers": 1, "winner": "v1"}]
    assert len(json.loads(stack_path.read_text(encoding="utf-8"))["layers"]) == 1
    code, out = _run(capsys, "campaign", "publish", "dp-camp", *common, "--requests", str(req))
    assert code == 0 and out["published"] == [{"eid": "dp-camp-r01", "skipped": True}]
