from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ci_lab.lessons.cli import main
from ci_lab.rulespec import LessonCluster, PriorPred, RuleFile, RuleSpec

SECRET = "Jane Q. User at 12 Elm Street, card 4111111111111111"
DAY = 86400 * 10**9
T0 = 1735689600 * 10**9  # 2025-01-01


def _w(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _assert_rec(kind: str, n: int, split: str = "evolve") -> dict:
    oid = f"CASE-{10000 + n}"
    calls = []
    if kind == "good":
        calls.append({"call_id": "c0", "name": "read_file", "arguments": {"resource_id": oid},
                      "result": {"resource_id": oid, "status": "delivered"}, "turn": 0})
    calls.append({"call_id": "c1", "name": "write_file", "arguments": {"resource_id": oid, "amount": 30},
                  "result": {"ok": True}, "turn": 0})
    return {"case_id": f"{kind}-{n}", "suite": "changes", "split": split, "slice": f"2025-01-0{1 + n % 4}",
            "pin": "oracle@1", "passed": kind == "good",
            "transcript": {"messages": [{"role": "user", "content": f"change {oid} please"},
                                        {"role": "assistant", "content": "Edit applied."}],
                           "tool_calls": calls},
            "violations": [] if kind == "good" else
            [{"rule_id": "harness.write_without_read", "severity": "critical", "detail": "no lookup"}]}


def _span(sid: str, parent: str, name: str, start: int, **attrs: object) -> dict:
    return {"schemaVersion": 1, "traceId": f"{sid:0>32}", "spanId": sid, "parentSpanId": parent, "name": name,
            "kind": "INTERNAL", "startTimeUnixNano": str(start), "endTimeUnixNano": str(start + 10),
            "status": {"code": "OK", "message": ""}, "attributes": attrs, "events": [], "links": [],
            "resource": {}, "scope": {}}


def _usage(n: int) -> list[dict]:
    t0 = T0 + (n % 4) * DAY
    oid = f"CASE-{20000 + n}"
    p = f"s{n}"
    return [
        _span(p + "c", "", "ci.case", t0, **{"ci.case_id": f"sess-{n}-{SECRET}", "openinference.span.kind": "CHAIN",
                                            "ci.oracle_rules": "harness.write_limit"}),
        _span(p + "a", p + "c", "agent.chat", t0 + 1, **{"openinference.span.kind": "AGENT", "input.value": SECRET,
                                                        "output.value": f"Dear {SECRET}, edit applied."}),
        _span(p + "x", p + "a", "tool.write_file", t0 + 2, **{
            "openinference.span.kind": "TOOL", "tool.name": "write_file",
            "input.value": json.dumps({"resource_id": oid, "amount": 900, "note": SECRET}),
            "output.value": json.dumps({"error": f"over limit for {SECRET}"})}),
    ]


def _all_output_text(*dirs: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for d in dirs for p in d.rglob("*") if p.is_file())


def test_usage_end_to_end_never_leaks_untrusted_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = _w(tmp_path / "in" / "usage.jsonl", [r for n in range(16) for r in _usage(n)])
    out, run = tmp_path / "traj", tmp_path / "run"
    assert main(["lessons", "harvest", "--source", "usage", "--in", str(src), "--out", str(out)]) == 0
    assert main(["lessons", "mine", "--in", str(out), "--run", str(run)]) == 0
    lines = [json.loads(x) for x in (run / "lessons" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines and all(LessonCluster.model_validate(x["cluster"]).status == "backlog" for x in lines)
    assert all(x["trusted"] is False and x["features"].get("trusted", False) is False for x in lines)
    blob = _all_output_text(out, run) + capsys.readouterr().out
    for raw in (SECRET, "Jane", "Elm Street", "4111111111111111", "CASE-2000", "edit applied", "over limit for"):
        assert raw not in blob, raw


def test_assert_end_to_end_confirm_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_assert_rec("unverified", i) for i in range(16)] + [_assert_rec("good", 1000 + i) for i in range(200)]
    rows.append(_assert_rec("unverified", 99, split="heldout"))
    src = _w(tmp_path / "assert.jsonl", rows)
    out, run = tmp_path / "traj", tmp_path / "run"
    assert main(["lessons", "harvest", "--source", "assert", "--in", str(src), "--out", str(out)]) == 3  # loud
    report = json.loads((out / "harvest_report.json").read_text(encoding="utf-8"))
    assert report["sealed_refused"] == 1 and report["harvested"] == 216
    assert "unverified-99" not in (out / "trajectories.jsonl").read_text(encoding="utf-8")

    assert main(["lessons", "mine", "--in", str(out), "--run", str(run)]) == 0
    cand = run / "lessons" / "candidates.jsonl"
    (line,) = [json.loads(x) for x in cand.read_text(encoding="utf-8").splitlines()]
    c = LessonCluster.model_validate(line["cluster"])
    assert c.route == "R2" and c.status == "candidate" and not c.human_confirmed
    assert line["features"]["prior_tool"] == "read_file" and line["holdout_members"]
    assert (run / "lessons" / "mine_report.json").exists()

    # confirmation is a human act: refused under CI / agent contexts
    monkeypatch.setenv("CI_LAB_AGENT", "1")
    assert main(["lessons", "confirm", c.id, "--run", str(run), "--by", "bot", "--yes"]) == 4
    monkeypatch.delenv("CI_LAB_AGENT")
    assert main(["lessons", "confirm", "lc-nope", "--run", str(run), "--by", "alice", "--yes"]) == 2
    assert main(["lessons", "confirm", c.id, "--run", str(run), "--by", "alice", "--yes"]) == 0
    c2 = LessonCluster.model_validate(json.loads(cand.read_text(encoding="utf-8"))["cluster"])
    assert c2.human_confirmed and c2.status == "confirmed"
    # re-mining preserves the human decision
    assert main(["lessons", "mine", "--in", str(out), "--run", str(run)]) == 0
    assert LessonCluster.model_validate(json.loads(cand.read_text(encoding="utf-8"))["cluster"]).human_confirmed

    rule = RuleSpec(id="harness.requires_read", version=1, rung="R2", on="tool_call", target="write_file",
                    require=PriorPred(kind="prior", tool="read_file",
                                      same=[("current.args.resource_id", "prior.args.resource_id")]),
                    action="block", template="precondition.prior_call",
                    slots={"tool": "write_file", "prior_tool": "read_file", "subject": "resource_id"})
    rules = tmp_path / "rules.yaml"
    rules.write_text(yaml.safe_dump(RuleFile(rules=[rule]).model_dump(mode="json", exclude_none=True)),
                     encoding="utf-8")
    rep_path = tmp_path / "replay.json"
    capsys.readouterr()
    assert main(["lessons", "validate", "--rules", str(rules), "--in", str(out), "--run", str(run),
                 "--cluster", c.id, "--out", str(rep_path)]) == 0, capsys.readouterr().out
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    assert rep["verdict"] == "pass_to_closed_loop" and rep["recall"] == 1.0

    broad = rule.model_copy(update={"require": PriorPred(kind="prior", tool="no_such_tool")})
    rules.write_text(yaml.safe_dump(RuleFile(rules=[broad]).model_dump(mode="json", exclude_none=True)),
                     encoding="utf-8")
    assert main(["lessons", "validate", "--rules", str(rules), "--in", str(out), "--run", str(run),
                 "--cluster", c.id]) == 2
    assert main(["lessons", "validate", "--rules", str(rules), "--in", str(out), "--cluster", c.id]) == 2


def test_confirm_refuses_non_tty_without_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    (run / "lessons").mkdir(parents=True)
    (run / "lessons" / "candidates.jsonl").write_text(json.dumps({"id": "lc-x"}) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["lessons", "confirm", "lc-x", "--run", str(run), "--by", "alice"]) == 4
