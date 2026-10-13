"""Shared offline fakes for optimizer/strategy tests."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ci_lab.contracts import EvalResult, EvaluatorPin, FailureRecord, TaskScore

PROMPT = "src/order_support/harness/prompts/system.md"
SKILL = "src/order_support/harness/skills/order-support/SKILL.md"
MEMORY = "src/order_support/harness/skills/order-support/memory.md"


class KeywordDomain:
    """Case ``c`` passes iff its keyword appears in the composed instructions
    (system.md + SKILL.md). Records every split it is asked to evaluate."""

    name = "fake"
    surface_globs = ("src/order_support/harness/**",)
    frozen_globs = ("src/order_support/harness/frozen/**",)
    component_globs = {"prompt": ("src/order_support/harness/prompts/*.md",)}

    def __init__(self, keywords=None, heldout=("h1", "h2")):
        self.keywords = keywords or {"c1": "refund", "c2": "tracking", "c3": "escalate", "c4": "polite"}
        self.heldout = tuple(heldout)
        self.splits_called: list[str] = []
        self.dirs: list[Path] = []

    def splits(self):
        return {"evolve": tuple(self.keywords), "heldout": self.heldout, "ood": (), "aa": ()}

    async def evaluate(self, harness_dir, split, k, *, experiment_id, variant):
        self.splits_called.append(split)
        self.dirs.append(Path(harness_dir))
        text = ""
        for rel in (PROMPT, SKILL):
            f = Path(harness_dir) / rel
            if f.exists():
                text += f.read_text(encoding="utf-8").lower()
        cases = self.splits()[split]
        scores = [TaskScore(c, t, "suite-a", 1.0 if self.keywords.get(c, "\0") in text else 0.0,
                            tokens_in=10, tokens_out=5) for c in cases for t in range(k)]
        return EvalResult("tree", split, EvaluatorPin("ev", "judge", "fake"), scores)

    def failures(self, result):
        return [FailureRecord(s.case_id, s.suite, "missing_policy", ("R1",), {"helpful": 0.0},
                              excerpt=f"agent forgot {self.keywords[s.case_id]}")
                for s in result.scores if s.trial == 0 and (s.score or 0) < 1]


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def worktree(tmp_path):
    wt = tmp_path / "wt"
    for rel, text in {PROMPT: "You are an order support agent.\n",
                      SKILL: "# Order support\nBe helpful.\n",
                      MEMORY: "- remember things\n",
                      "src/order_support/harness/agent.yaml": "name: OrderSupport\ninstructions: Base.\n"}.items():
        f = wt / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8", newline="\n")
    git(wt.parent, "init", "-q", str(wt))
    git(wt, "config", "user.email", "t@example.com")
    git(wt, "config", "user.name", "t")
    git(wt, "config", "core.autocrlf", "false")
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "base")
    return wt


@pytest.fixture
def domain():
    return KeywordDomain()
