"""``ci-lab harness validate|metrics``."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ci_lab.cli import main
from ci_lab.harness_tree import ENV_VAR, repo_harness_dir


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    dest = tmp_path / "candidate"
    shutil.copytree(repo_harness_dir(), dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def test_validate_repo_tree(capsys, monkeypatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert main(["harness", "validate"]) == 0
    assert "valid" in capsys.readouterr().out


def test_validate_reports_errors_as_json(tree: Path, capsys) -> None:
    (tree / "harness.yaml").write_text("format: ci_lab.harness.v1\n", encoding="utf-8")
    (tree / "notes.txt").write_text("stray\n", encoding="utf-8")
    assert main(["harness", "validate", "--dir", str(tree), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert not out["ok"] and out["root"] == str(tree)
    assert any("differs from the frozen manifest" in e for e in out["errors"])
    assert "notes.txt: not in any component" in out["errors"]


def test_env_selects_the_default_dir(tree: Path, capsys, monkeypatch) -> None:
    (tree / "agents" / "student.yaml").unlink()
    monkeypatch.setenv(ENV_VAR, str(tree))
    assert main(["harness", "validate"]) == 1
    assert "agents/student.yaml: required agent missing" in capsys.readouterr().err


def test_metrics_per_component(tree: Path, capsys) -> None:
    assert main(["harness", "metrics", "--dir", str(tree), "--json"]) == 0
    metrics = json.loads(capsys.readouterr().out)
    for component in ("prompt", "agent", "skill", "workflow", "loop", "client_tool", "mcp", "guard"):
        assert f"component.{component}.complexity" in metrics
    assert metrics["component.prompt.complexity"] > 0
    assert main(["harness", "metrics", "--dir", str(tree)]) == 0
    assert "component.agent.complexity\t" in capsys.readouterr().out


def test_metrics_missing_dir(tmp_path: Path, capsys) -> None:
    assert main(["harness", "metrics", "--dir", str(tmp_path / "nope")]) == 2
    assert "not found" in capsys.readouterr().err
