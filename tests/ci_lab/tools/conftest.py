from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = "harness"
LAYOUT = SimpleNamespace(
    harness=HARNESS,
    surface=(f"{HARNESS}/**",),
    component_globs={
        "prompt": (f"{HARNESS}/prompts/*.md", f"{HARNESS}/prompts/**/*.md"),
        "skill": (f"{HARNESS}/skills/**",),
        "client_tool": (f"{HARNESS}/tool_specs.yaml",),
        "config": (f"{HARNESS}/agent.yaml",),
        "memory": (f"{HARNESS}/skills/**/memory.md",),
        "context_mgmt": (f"{HARNESS}/agent.yaml",),
    },
    frozen=("**/*.py", "evals/**", "tests/**"),
    files={
        f"{HARNESS}/prompts/system.md": "You are a helpful harness agent.\nAlways inspect the target first.\n",
        f"{HARNESS}/skills/changes/SKILL.md": "# Changes\nReview repository state before changing files.\n",
        f"{HARNESS}/skills/changes/memory.md": "- nothing yet\n",
        f"{HARNESS}/tool_specs.yaml": "read_file:\n  description: Read a harness file by path.\n",
        f"{HARNESS}/agent.yaml": "name: HarnessAgent\nmodel:\n  id: gpt-5\ninstructions_file: prompts/system.md\n",
        f"{HARNESS}/helper.py": "print('code is frozen')\n",
        "evals/assert/x/eval_config.yaml": "suite: x\n",
    },
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def make_repo(root: Path, files: dict[str, str] | None = None) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    for key, value in (("user.name", "test"), ("user.email", "test@example.com"), ("commit.gpgsign", "false"),
                       ("core.autocrlf", "false")):
        git(root, "config", key, value)
    for rel, text in (files or LAYOUT.files).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root.resolve(), git(root, "rev-parse", "HEAD").strip()


@pytest.fixture
def layout() -> SimpleNamespace:
    return LAYOUT


@pytest.fixture
def gitrun() -> Callable[..., str]:
    return git


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    return make_repo(tmp_path / "wt")
