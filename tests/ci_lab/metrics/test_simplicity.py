"""ci_lab.metrics.simplicity: surface counts, composite complexity, simplicity score."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from ci_lab.metrics.simplicity import (
    COMPLEXITY_WEIGHTS,
    complexity,
    relative_change,
    simplicity_score,
    surface_metrics,
)


def _w(root: Path, rel: str, text: str | bytes) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def harness(tmp_path: Path) -> Path:
    h = tmp_path / "harness"
    _w(h, "harness.yaml", "format: 1\ncomponents:\n  prompt: [prompts/**]\n" * 20)
    _w(h, "prompts/system.md", "x" * 101)
    _w(h, "agents/triage.yaml",
       "kind: Prompt\nname: triage\ninstructions: " + "y" * 40 + "\ntools:\n  - lookup\n  - change\n")
    _w(h, "workflows/flow.yaml",
       "kind: Workflow\ntrigger:\n  actions:\n    - id: a\n    - id: b\n      actions:\n        - id: c\n")
    _w(h, "tools/lookup.yaml", "format: 1\ntools:\n  lookup: {}\n  change: {}\n  cancel: {}\n")
    _w(h, "mcp/servers.json", '{"servers": {"a": {}, "b": {}}}')
    _w(h, "skills/changes/SKILL.md", "line one\n\nline two\nline three\n")
    _w(h, "loop/policy.py",
       "# comment\n\ndef f(x):\n    if x and x > 1:\n        return [i for i in x if i]\n    return 0\n")
    return h


def test_counts(harness: Path) -> None:
    m = surface_metrics(harness)
    assert m["files"] == 7
    # 101 md chars -> 26; skill md 37 chars -> 10; instructions 40 chars -> 10
    skill_chars = len("line one\n\nline two\nline three\n")
    assert m["prompt_tokens"] == math.ceil(101 / 4) + math.ceil(skill_chars / 4) + 10
    assert m["agents"] == 1
    assert m["tools"] == 2 + 3
    assert m["workflow_steps"] == 3
    assert m["mcp_servers"] == 2
    assert m["skill_lines"] == 3
    assert m["py_loc"] == 4
    # 1 function + if + `and` + comprehension(1 + 1 if)
    assert m["py_cyclomatic"] == 5
    assert m["yaml_nodes"] > 0
    assert m["complexity"] == pytest.approx(complexity(m))


def test_complexity_formula() -> None:
    counts = {"prompt_tokens": 50, "yaml_nodes": 20, "agents": 1, "tools": 1, "workflow_steps": 1,
              "mcp_servers": 1, "skill_lines": 10, "py_loc": 10, "py_cyclomatic": 1}
    assert complexity(counts) == pytest.approx(1 + 1 + 5 + 2 + 1 + 3 + 1 + 1 + 1)
    assert set(COMPLEXITY_WEIGHTS) <= set(counts)


def test_harness_yaml_excluded_by_default(harness: Path) -> None:
    base = surface_metrics(harness)
    _w(harness, "harness.yaml", "format: 1\n" + "k: [1, 2, 3]\n" * 500)
    assert surface_metrics(harness) == base
    assert surface_metrics(harness, exclude=())["yaml_nodes"] > base["yaml_nodes"]


def test_exclude_globs(harness: Path) -> None:
    m = surface_metrics(harness, exclude=("harness.yaml", "loop/**"))
    assert m["py_loc"] == 0 and m["py_cyclomatic"] == 0


def test_ignores_pycache_dotfiles_binary(harness: Path) -> None:
    base = surface_metrics(harness)
    _w(harness, "loop/__pycache__/policy.py", "def g():\n    return 1\n")
    _w(harness, ".hidden.md", "z" * 1000)
    _w(harness, ".cache/x.md", "z" * 1000)
    _w(harness, "prompts/blob.md", b"\x00\x01\x02" * 100)
    assert surface_metrics(harness) == base


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinks_not_followed(harness: Path, tmp_path: Path) -> None:
    base = surface_metrics(harness)
    outside = tmp_path / "outside"
    _w(outside, "big.md", "q" * 4000)
    (harness / "linked").symlink_to(outside, target_is_directory=True)
    (harness / "prompts" / "l.md").symlink_to(outside / "big.md")
    assert surface_metrics(harness) == base


def test_unparseable_files_fail_closed(tmp_path: Path) -> None:
    h = tmp_path / "h"
    _w(h, "bad.yaml", "a: [1, 2\nb: : :\n\nc\n")
    _w(h, "bad.py", "def (:\n  pass\n\n")
    m = surface_metrics(h)
    assert m["yaml_nodes"] == 3
    assert m["py_loc"] == 2 and m["py_cyclomatic"] == 2


def test_component_globs(harness: Path) -> None:
    m = surface_metrics(harness, {"prompt": ["prompts/**"], "loop": ["harness/loop/**"], "none": ["nope/**"]})
    assert m["component.prompt.complexity"] == pytest.approx(math.ceil(101 / 4) / 50)
    assert m["component.loop.complexity"] == pytest.approx(4 / 10 + 5)
    assert m["component.none.complexity"] == 0.0


def test_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        surface_metrics(tmp_path / "missing")


def test_deterministic(harness: Path) -> None:
    assert surface_metrics(harness) == surface_metrics(harness)


def test_relative_change() -> None:
    assert relative_change(110, 100) == pytest.approx(0.1)
    assert relative_change(1, 0) == pytest.approx(1.0)
    assert relative_change(0.5, 0.2) == pytest.approx(0.3)


@pytest.mark.parametrize(("new", "score"), [
    (80, 1.0), (90, 1.0), (95, 0.75), (100, 0.5), (112.5, 0.25), (125, 0.0), (200, 0.0)])
def test_simplicity_score(new: float, score: float) -> None:
    assert simplicity_score({"complexity": new}, {"complexity": 100}) == pytest.approx(score)
