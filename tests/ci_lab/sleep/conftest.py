from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SKILL = """---
name: harness-editing
description: Safe self-hosted harness editing procedures.
---
# Harness editing

Keep harness changes bounded and validate the frozen manifest.
"""


def task_row(i: int, rule: str = "verify_identity", **extra) -> dict:
    row = {
        "id": f"t{i:02d}", "project": "harness-editing",
        "intent": f"Please propose a bounded harness improvement for failing case {i}.",
        "reference": "I inspected the harness and proposed a bounded edit.",
        "reference_kind": "rule",
        "judge": {"kind": "rule", "checks": [{"op": "tool_called", "arg": "lookup_order"},
                                             {"op": "contains", "arg": "verified"}]},
        "tags": [rule, f"rule:{rule}", "suite:refunds"], "reviewed": True,
    }
    row.update(extra)
    return row


def write_tasks(path: Path, rows: list[dict], *, header: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = header or {"format": "skillopt_sleep.tasks.v1", "project": "harness-editing", "reviewed": True}
    path.write_text("".join(json.dumps(r) + "\n" for r in [head, *rows]), encoding="utf-8", newline="\n")
    return path


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def init_repo(repo: Path) -> None:
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")


@pytest.fixture
def h() -> SimpleNamespace:
    return SimpleNamespace(SKILL=SKILL, task_row=task_row, write_tasks=write_tasks, git=git, init_repo=init_repo)


@pytest.fixture
def sleep_repo(tmp_path: Path) -> Path:
    """A tiny git repo with the incumbent skill, night-0 state and 6 reviewed tasks."""
    repo = tmp_path / "repo"
    skill = repo / "harness/skills/harness-editing/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(SKILL, encoding="utf-8", newline="\n")
    triage = repo / "harness/skills/trace-triage/SKILL.md"
    triage.parent.mkdir(parents=True, exist_ok=True)
    triage.write_text(SKILL.replace("harness-editing", "trace-triage"), encoding="utf-8", newline="\n")
    legacy = repo / "src/order_support/harness/skills/order-support/SKILL.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(SKILL.replace("harness-editing", "order-support"), encoding="utf-8", newline="\n")
    state = repo / "experiments/sleep/state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"format": "ci_lab.sleep.state.v1", "night": 0, "history": []}, indent=2) + "\n",
                     encoding="utf-8", newline="\n")
    write_tasks(repo / "experiments/sleep/harness-editing.jsonl", [task_row(i) for i in range(6)])
    write_tasks(repo / "experiments/sleep/trace-triage.jsonl",
                [task_row(i, id=f"r{i:02d}", project="trace-triage") for i in range(6)],
                header={"format": "skillopt_sleep.tasks.v1", "project": "trace-triage", "reviewed": True})
    write_tasks(repo / "experiments/sleep/tasks.jsonl",
                [task_row(i, project="order-support") for i in range(6)],
                header={"format": "skillopt_sleep.tasks.v1", "project": "order-support", "reviewed": True})
    init_repo(repo)
    return repo
