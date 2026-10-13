"""`.github/CODEOWNERS` covers every ci-guardrails frozen path and the protected harness surfaces.

Expected owners come from the template marker `.github/template.yml`: the template's owner in the
template repository, the `--owners` recorded by `ci-lab template init` in a repository created from it.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

from ci_lab.template.marker import (
    TEMPLATE_OWNERS,
    TEMPLATE_REPO,
    expected_owners,
    read_marker,
)

REPO = Path(__file__).resolve().parents[3]
CODEOWNERS = REPO / ".github" / "CODEOWNERS"
POLICY = REPO / ".github" / "extensions" / "ci-guardrails" / "policy.mjs"
MARKER = read_marker(REPO)
OWNERS = list(expected_owners(MARKER))


def rules() -> list[tuple[str, list[str]]]:
    out = []
    for line in CODEOWNERS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pattern, *owners = line.split()
            out.append((pattern, owners))
    return out


def owned(path: str) -> list[str]:
    """Owners of a repo-relative path (last matching rule wins, as on GitHub)."""
    hit: list[str] = []
    for pattern, owners in rules():
        pat = pattern.lstrip("/")
        anchored = pattern.startswith("/") or "/" in pattern.rstrip("/").removeprefix("**/")
        cands = [pat] if anchored else [pat, f"**/{pat}"]
        if pat.startswith("**/"):
            cands.append(pat[3:])
        if any(fnmatch.fnmatchcase(path, c) for c in cands):
            hit = owners
    return hit


def frozen_list(name: str) -> list[str]:
    body = re.search(rf"export const {name} = Object\.freeze\(\[(.*?)\]\);", POLICY.read_text(encoding="utf-8"),
                     re.DOTALL)
    assert body, name
    return re.findall(r'"([^"]+)"', body.group(1))


def test_marker_names_the_owners():
    assert MARKER is not None, ".github/template.yml is missing"
    if MARKER["role"] == "template":
        assert MARKER["template_repository"] == TEMPLATE_REPO and OWNERS == list(TEMPLATE_OWNERS)
    else:
        assert MARKER["initialized"] is True and OWNERS == MARKER["owners"]


def test_every_rule_has_the_owner_and_anchors_exist():
    rs = rules()
    assert rs
    for pattern, owners in rs:
        assert owners == OWNERS, pattern
        if pattern.startswith("/"):
            base = pattern.lstrip("/").removesuffix("/**")
            assert (REPO / base).exists(), pattern


@pytest.mark.parametrize("path", [
    "lint/rules/workflows.yaml",
    "src/order_support/harness/guards/order_support.yaml",
    "src/ci_lab/rules/engine.py",
    "src/ci_lab/judge/provider.py",
    "src/order_support/oracle.py",
    "evals/assert/judge_replay/eval_config.yaml",
    ".github/workflows/sleep-nightly.yml",
    ".github/extensions/ci-guardrails/policy.mjs",
    ".github/CODEOWNERS",
    ".github/template.yml",
])
def test_protected_paths_are_owned(path):
    assert owned(path) == OWNERS


def test_frozen_guardrail_paths_are_owned():
    for f in frozen_list("FROZEN_PATHS"):
        target = f if (REPO / f).exists() else f"src/order_support/{f}"
        assert owned(target) == OWNERS, f
    for d in frozen_list("FROZEN_DIRS"):
        assert owned(f"{d}x.yaml") == OWNERS or owned(f"src/order_support/{d}x.yaml") == OWNERS, d


def test_unprotected_paths_are_not_owned():
    for path in ("README.md", "src/ci_lab/sleep/night.py", "docs/harness.md", "src/order_support/agent.py"):
        assert owned(path) == [], path
