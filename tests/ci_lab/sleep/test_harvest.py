from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci_lab.sleep.harvest import (
    HarvestError,
    SplitViolation,
    UnknownJudgeOp,
    assign_stable_splits,
    from_agl_exports,
    harvest,
    load_reviewed_tasks,
    read_jsonl_rows,
    validate_judge,
)
from ci_lab.sleep.night import TASKS_REL

REPO = Path(__file__).resolve().parents[3]


def agl_row(i: int, split: str = "evolve", **task) -> dict:
    body = {"id": f"case-{i}", "project": "harness-editing", "intent": f"Inspect fixture CASE-{1000 + i}.",
            "reference_kind": "rule", "judge": {"kind": "rule", "checks": [{"op": "tool_called", "arg": "read_file"}]},
            "tags": ["inspect_before_edit"], "context_excerpt": "TOOL RESULT: {secret stuff}",
            "tool_calls": [{"name": "read_file", "result": {"token": "secret"}}]}
    body.update(task)
    return {"dataset_split": split, "suite": "harness", "task": body}


def test_seed_tasks_file_is_valid():
    res = harvest(REPO / TASKS_REL)
    assert len(res.tasks) >= 8
    assert {t.split for t in res.tasks} == {"train", "val"}
    assert all(t.reference_kind in ("ci_rule", "ci_assert") for t in res.tasks)
    header = json.loads((REPO / TASKS_REL).read_text(encoding="utf-8").splitlines()[0])
    assert header["format"] == "skillopt_sleep.tasks.v1" and header["reviewed"] is True


def test_unreviewed_task_rejected(tmp_path, h):
    p = h.write_tasks(tmp_path / "t.jsonl", [h.task_row(0), h.task_row(1, reviewed=False)])
    with pytest.raises(HarvestError, match="not reviewed"):
        load_reviewed_tasks(p)


def test_unreviewed_header_rejected(tmp_path, h):
    p = h.write_tasks(tmp_path / "t.jsonl", [h.task_row(0)],
                      header={"format": "skillopt_sleep.tasks.v1", "reviewed": False})
    with pytest.raises(HarvestError, match="header"):
        load_reviewed_tasks(p)


def test_duplicate_ids_rejected(tmp_path, h):
    p = h.write_tasks(tmp_path / "t.jsonl", [h.task_row(0), h.task_row(0)])
    with pytest.raises(HarvestError, match="duplicate"):
        load_reviewed_tasks(p)


@pytest.mark.parametrize("op", ["llm", "exact", "TOOL_CALL", None])
def test_unknown_op_rejected_at_harvest(tmp_path, h, op):
    row = h.task_row(0, judge={"kind": "rule", "checks": [{"op": op, "arg": "x"}]})
    with pytest.raises(UnknownJudgeOp):
        load_reviewed_tasks(h.write_tasks(tmp_path / "t.jsonl", [row]))


@pytest.mark.parametrize("judge", [
    {"kind": "llm", "checks": [{"op": "contains", "arg": "x"}]},
    {"kind": "rule", "checks": []},
    {"kind": "rule", "checks": [{"op": "regex", "arg": "("}]},
    {"kind": "rule", "checks": [{"op": "max_chars", "arg": "10"}]},
    {"kind": "rule", "checks": [{"op": "contains", "arg": "  "}]},
])
def test_bad_judges_rejected(judge):
    with pytest.raises(HarvestError):
        validate_judge(judge)


def test_unknown_tool_rejected_when_tool_contract_is_supplied():
    judge = {"kind": "rule", "checks": [{"op": "tool_called", "arg": "delete_database"}]}
    with pytest.raises(HarvestError, match="unknown tool"):
        validate_judge(judge, known_tools=frozenset({"read_file", "write_file"}))


def test_agl_evolve_rows_are_harvested_and_stripped():
    tasks, excluded = from_agl_exports([agl_row(1), agl_row(2)])
    assert excluded == 0 and len(tasks) == 2
    t = tasks[0]
    assert t.id.startswith("agl-case-1-") and t.context_excerpt == "" and t.attempted_solution == ""
    assert "suite:harness" in t.tags and t.reference_kind == "ci_rule"
    assert "secret" not in json.dumps(t.to_dict() if hasattr(t, "to_dict") else t.__dict__)


@pytest.mark.parametrize("split", ["heldout", "ood", "aa", "train", "validation"])
def test_non_evolve_split_is_hard_error(split):
    with pytest.raises(SplitViolation):
        from_agl_exports([agl_row(1), agl_row(2, split=split)])


def test_missing_split_is_hard_error():
    row = agl_row(1)
    del row["dataset_split"]
    with pytest.raises(SplitViolation):
        from_agl_exports([row])


def test_flat_split_field_accepted_and_policed():
    flat = {**agl_row(1)["task"], "split": "evolve"}
    assert len(from_agl_exports([flat])[0]) == 1
    with pytest.raises(SplitViolation):
        from_agl_exports([{**flat, "split": "heldout"}])


def test_injection_suites_excluded():
    tasks, excluded = from_agl_exports([agl_row(1), {**agl_row(2), "suite": "indirect_prompt_injection"}])
    assert len(tasks) == 1 and excluded == 1


def test_dedupe_and_reviewed_win(tmp_path, h):
    rev = h.task_row(0, intent="Inspect fixture CASE-1001.", reference_kind="rule",
                     judge={"kind": "rule", "checks": [{"op": "tool_called", "arg": "read_file"}]})
    p = h.write_tasks(tmp_path / "t.jsonl", [rev, h.task_row(1)])
    rows = [agl_row(1), agl_row(1), agl_row(3)]
    res = harvest(p, rows)
    ids = [t.id for t in res.tasks]
    assert "t00" in ids and not any(i.startswith("agl-case-1-") for i in ids)
    assert res.duplicates == 2 and len(res.tasks) == 3


def test_stable_split_and_cap(tmp_path, h):
    rows = [agl_row(i, intent=f"question number {i}") for i in range(30)]
    a = harvest(None, rows, max_tasks=12)
    b = harvest(None, list(reversed(rows)), max_tasks=12)
    assert sorted((t.id, t.split) for t in a.tasks) == sorted((t.id, t.split) for t in b.tasks)
    assert len(a.tasks) == 12 and a.capped == 18
    assert {t.split for t in a.tasks} == {"train", "val"}


def test_split_guarantees_both_sides():
    from skillopt_sleep.types import TaskRecord

    two = [TaskRecord(id=f"x{i}", project="p", intent="i") for i in range(2)]
    assert sorted(t.split for t in assign_stable_splits(two, val_fraction=0.01)) == ["train", "val"]
    assert sorted(t.split for t in assign_stable_splits(two, val_fraction=0.99)) == ["train", "val"]


def test_secrets_redacted(tmp_path, h):
    row = h.task_row(0, intent="my key is ghp_" + "A" * 36 + " help")
    (t,) = load_reviewed_tasks(h.write_tasks(tmp_path / "t.jsonl", [row]))
    assert "ghp_" + "A" * 36 not in t.intent


def test_read_jsonl_rows(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(agl_row(1)) + "\n\n", encoding="utf-8")
    assert len(read_jsonl_rows([p])) == 1
    p.write_text("{nope\n", encoding="utf-8")
    with pytest.raises(HarvestError):
        read_jsonl_rows([p])
