from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from ci_lab.contracts import ToolCallRecord, Transcript, Violation
from ci_lab.lessons.common import SealedSplitError, check_split
from ci_lab.lessons.harvest import (
    HarvestOptions,
    HarvestStats,
    family_of,
    from_transcript,
    harvest,
    harvest_agl,
    harvest_assert,
    harvest_calibrate,
    harvest_spans,
    time_slice,
)

SECRET = "Jane Q. User at 12 Elm Street, card 4111111111111111"


def _w(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _transcript() -> Transcript:
    return Transcript(
        case_id="change_damaged-para2",
        messages=[{"role": "user", "content": "change CASE-10003 please"},
                  {"role": "assistant", "content": "Changeed."}],
        tool_calls=[ToolCallRecord(call_id="c1", name="write_file", arguments={"resource_id": "CASE-10003", "amount": 30},
                                   result={"ok": True}, turn=0)])


def test_family_and_slice() -> None:
    assert family_of("change_damaged-para2") == "change_damaged"
    assert family_of("Change_Damaged_v3-t2") == "change_damaged"
    assert family_of("case-12") == "case-12"
    assert time_slice(1735689600) == "2025-01-01"
    assert time_slice("1735689600000000000") == "2025-01-01"
    assert time_slice(None) is None


@pytest.mark.parametrize("split", ["heldout", "ood", "aa", "confirm", "HOLDOUT", "weird", ""])
def test_sealed_and_unknown_splits_refused(split: str) -> None:
    with pytest.raises(SealedSplitError):
        check_split(split, source="assert")


def test_split_policy() -> None:
    assert check_split("evolve", source="assert") == "evolve"
    assert check_split(None, source="assert", default="evolve") == "evolve"
    assert check_split(None, source="usage") == "usage"
    with pytest.raises(SealedSplitError):
        check_split("heldout", source="usage", default="evolve")


def test_from_transcript() -> None:
    t = from_transcript(_transcript(), [Violation(rule_id="harness.write_without_read", severity="critical", detail="x")],
                        split="evolve", suite="changes", pin="oracle@1")
    assert [s.kind for s in t.steps] == ["user", "tool_call", "tool_result", "response"]
    assert t.steps[0].text is None  # user text never enters trajectories
    assert t.steps[1].args == {"resource_id": "CASE-10003", "amount": 30}
    assert t.outcome.oracle_rules == ("harness.write_without_read",)
    assert t.outcome.passed is False
    assert (t.family, t.slice, t.pin, t.trusted, t.split) == ("change_damaged", "changes", "oracle@1", True, "evolve")
    with pytest.raises(SealedSplitError):
        from_transcript(_transcript(), [], split="heldout")


def test_harvest_assert_jsonl_refuses_sealed_loudly(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tr = dataclasses.asdict(_transcript())
    p = _w(tmp_path / "assert.jsonl", [
        {"case_id": "a-1", "suite": "changes", "split": "evolve", "transcript": tr, "score": 1.0, "violations": []},
        {"case_id": "a-2", "suite": "changes", "split": "heldout", "transcript": tr, "score": 0.0, "violations": []},
        {"case_id": "a-3", "suite": "indirect_prompt_injection", "split": "evolve", "transcript": tr,
         "violations": [{"rule_id": "injection.followed", "severity": "critical", "detail": SECRET}]},
    ])
    stats = HarvestStats()
    out = harvest_assert(p, HarvestOptions(pin="o@1"), stats)
    assert stats.sealed == 1 and len(out) == 2
    assert "REFUSED" in caplog.text
    ok, inj = out
    assert ok.outcome.passed is True
    assert inj.outcome.injection_suspect is True
    # injection-suspect trajectories are reduced: no raw text or ids survive
    blob = inj.model_dump_json()
    assert "Changeed." not in blob and "CASE-10003" not in blob and SECRET not in blob


def _span(sid: str, parent: str, name: str, start: int, **attrs: object) -> dict:
    return {"schemaVersion": 1, "traceId": "t" * 32, "spanId": sid, "parentSpanId": parent, "name": name,
            "kind": "INTERNAL", "startTimeUnixNano": str(start), "endTimeUnixNano": str(start + 10),
            "status": {"code": "OK", "message": ""}, "attributes": attrs, "events": [], "links": [],
            "resource": {}, "scope": {}}


def _spans(case_id: str = "change-7", split: str = "evolve") -> list[dict]:
    t0 = 1735689600 * 10**9
    return [
        _span("c1", "", "ci.case", t0, **{"ci.case_id": case_id, "ci.split": split, "ci.trial": 0,
                                          "openinference.span.kind": "CHAIN", "ci.oracle_rules": "harness.write_limit"}),
        _span("a1", "c1", "agent.chat", t0 + 1, **{"openinference.span.kind": "AGENT", "input.value": SECRET,
                                                   "output.value": "Change issued to you."}),
        _span("x1", "a1", "tool.read_file", t0 + 2, **{
            "openinference.span.kind": "TOOL", "tool.name": "read_file",
            "input.value": json.dumps({"resource_id": "CASE-10007"}),
            "output.value": json.dumps({"status": "delivered", "total": 40})}),
        _span("x2", "a1", "execute_tool write_file", t0 + 3, **{
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "write_file",
            "gen_ai.tool.call.arguments": json.dumps({"resource_id": "CASE-10007", "amount": 900}),
            "gen_ai.tool.call.result": json.dumps({"error": "over limit"})}),
    ]


def test_harvest_spans(tmp_path: Path) -> None:
    p = _w(tmp_path / "spans" / "trace.jsonl", _spans())
    (t,) = harvest_spans(p.parent, HarvestOptions(pin="o@1"))
    assert [s.kind for s in t.steps] == ["user", "tool_call", "tool_result", "tool_call", "tool_result", "response"]
    assert [s.tool for s in t.steps if s.kind == "tool_call"] == ["read_file", "write_file"]
    assert t.steps[4].status == "error"
    assert t.outcome.oracle_rules == ("harness.write_limit",) and t.outcome.passed is False
    assert t.slice == "2025-01-01" and t.family == "change-7" and t.source == "spans"
    assert SECRET not in t.model_dump_json()

    stats = HarvestStats()
    assert harvest_spans(_w(tmp_path / "sealed.jsonl", _spans(split="ood")), None, stats) == []
    assert stats.sealed == 1


def test_harvest_spans_with_oracle(tmp_path: Path) -> None:
    class Oracle:
        def check(self, transcript: Transcript) -> list[Violation]:
            assert transcript.tool_calls[1].name == "write_file"
            return [Violation(rule_id="harness.write_without_read", severity="critical", detail="")]

    p = _w(tmp_path / "s.jsonl", _spans())
    (t,) = harvest_spans(p, HarvestOptions(oracle=Oracle()))
    assert t.outcome.oracle_rules == ("harness.write_limit", "harness.write_without_read")


def _agl(root: Path, rid: str, split: str = "evolve") -> None:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": SECRET},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "k1", "type": "function", "function": {"name": "write_file",
                                                               "arguments": json.dumps({"resource_id": "CASE-1", "amount": 5})}}]},
            {"role": "tool", "tool_call_id": "k1", "content": json.dumps({"ok": True})}]
    _w(root / f"{rid}.jsonl", [
        {"v": 1, "kind": "start", "rollout_id": rid, "attempt_id": "a1", "ts": 1735776000.0,
         "key": {"experiment_id": "e", "variant": "v", "case_id": "change-9", "trial": 1, "attempt": 0},
         "input": {"suite": "changes", "split": split}},
        {"v": 1, "kind": "event", "rollout_id": rid, "attempt_id": "a1", "ts": 1.0, "event_id": "e1",
         "event_type": "model_request", "data": {"request": {"messages": msgs},
                                                 "response": {"content": "Done!", "tool_calls": []}}},
        {"v": 1, "kind": "event", "rollout_id": rid, "attempt_id": "a1", "ts": 2.0, "event_id": "e2",
         "event_type": "ci.score", "data": {"name": "oracle", "value": 0.0, "suite": "changes",
                                            "rule_ids": ["harness.write_without_read"], "excerpt": SECRET}},
        {"v": 1, "kind": "finish", "rollout_id": rid, "attempt_id": "a1", "ts": 3.0, "status": "succeeded"},
    ])
    with open(root / f"{rid}.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"v":1,"kind":"event","trunc')  # crash-truncated tail is skipped


def test_harvest_agl(tmp_path: Path) -> None:
    _agl(tmp_path / "j", "r1")
    _agl(tmp_path / "j", "r2", split="confirm")
    stats = HarvestStats()
    (t,) = harvest_agl(tmp_path / "j", None, stats)
    assert stats.sealed == 1
    assert [s.kind for s in t.steps] == ["user", "tool_call", "tool_result", "response"]
    assert t.steps[1].args == {"resource_id": "CASE-1", "amount": 5}
    assert t.outcome.oracle_rules == ("harness.write_without_read",) and t.outcome.passed is False
    assert t.slice == "2025-01-02" and t.family == "change-9"
    assert SECRET not in t.model_dump_json()


def test_harvest_calibrate(tmp_path: Path) -> None:
    p = tmp_path / "cal.yaml"
    p.write_text(json.dumps({"split": "evolve", "cases": [
        {"id": "cal-1", "tags": ["change"], "labels": {"grounded": False, "tool_use": "appropriate",
                                                       "resolution": 1, "pii_leak": False, "human_pass": False},
         "observable": {"conversation": [{"role": "user", "content": SECRET}],
                        "tool_calls": [{"name": "read_file", "arguments": {"resource_id": "CASE-2"},
                                        "result": {"status": "processing"}}],
                        "final_response": "The change was applied."}},
        {"id": "cal-2", "split": "heldout", "labels": {"human_pass": True}, "observable": {}},
    ]}), encoding="utf-8")
    stats = HarvestStats()
    (t,) = harvest_calibrate(p, None, stats)
    assert stats.sealed == 1
    assert t.outcome.human_label == "bad" and t.trusted
    assert set(t.outcome.rubric_fails) == {"rubric.grounded", "rubric.resolution"}
    assert [s.kind for s in t.steps] == ["user", "tool_call", "tool_result", "response"]


def test_usage_is_reduced_and_untrusted(tmp_path: Path) -> None:
    rows = _spans(case_id="sess-" + SECRET, split="")
    _w(tmp_path / "u.jsonl", rows)
    (t,) = harvest("usage", tmp_path / "u.jsonl", HarvestOptions())
    assert t.split == "usage" and t.source == "usage" and t.trusted is False
    blob = t.model_dump_json()
    for raw in (SECRET, "CASE-10007", "Change issued to you.", "over limit", "change-7"):
        assert raw not in blob
    call = next(s for s in t.steps if s.tool == "write_file" and s.kind == "tool_call")
    assert call.args["amount"] == 900 and call.args["resource_id"].startswith("#id:")
    resp = t.steps[-1]
    assert resp.text is None and resp.text_digest
    # same resource id → same hashed subject across steps (joins survive reduction)
    first = next(s for s in t.steps if s.tool == "read_file")
    assert first.args["resource_id"] == call.args["resource_id"]
    # a human label makes usage trusted
    (t2,) = harvest("usage", tmp_path / "u.jsonl", HarvestOptions(labels={"sess-" + SECRET: "bad"}))
    assert t2.trusted is True and t2.outcome.human_label == "bad"
    assert SECRET not in t2.model_dump_json()
