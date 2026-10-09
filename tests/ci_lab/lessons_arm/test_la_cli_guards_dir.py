"""Lifecycle tools (promote / retire / CLI) default to the guard dir the agent actually loads."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ci_lab.domain.layout import repo_guards_dir
from ci_lab.lessons_arm.bundle import read_rule_file
from ci_lab.lessons_arm.cli import _all_rules, _guards
from ci_lab.lessons_arm.promote import PromotionError, promote
from ci_lab.lessons_arm.retire import RetirementCandidate, apply_ablation
from ci_lab.rulespec import GUARDS_DIR

HARNESS_ROOT = "harness"
SEED_RULES = Path(__file__).parents[1] / "guards" / "fixtures" / "rules.yaml"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    gdir = tmp_path / HARNESS_ROOT / "guards"
    gdir.mkdir(parents=True)
    shutil.copy(SEED_RULES, gdir / SEED_RULES.name)
    return tmp_path


def test_repo_guards_dir_uses_harness_layout(repo: Path, tmp_path_factory) -> None:
    assert repo_guards_dir(repo) == repo / HARNESS_ROOT / "guards"
    bare = tmp_path_factory.mktemp("bare")
    assert repo_guards_dir(bare) == bare / GUARDS_DIR


def test_cli_reads_rules_from_agent_guard_dir(repo: Path) -> None:
    assert _guards(repo) == repo / HARNESS_ROOT / "guards"
    seeds = {r.id for r in read_rule_file(SEED_RULES).rules}
    assert seeds and {r.id for r in _all_rules(_guards(repo))} == seeds


def test_promote_searches_agent_guard_dir(repo: Path, tmp_path: Path) -> None:
    decisions = tmp_path / "d.jsonl"
    decisions.write_text("", encoding="utf-8")
    with pytest.raises(PromotionError, match="no-such-rule"):
        promote("no-such-rule", decisions, None, repo=repo)


def test_apply_ablation_edits_agent_guard_dir(repo: Path) -> None:
    rule = read_rule_file(SEED_RULES).rules[0]
    committed: list[list[str]] = []

    def committer(wt: Path, files: list[str], msg: str) -> str:
        committed.append(files)
        return "deadbeef"

    edit = apply_ablation(RetirementCandidate(rule.id, rule.version, "dormant", 0, 0, 1.0), repo,
                          committer=committer)
    assert committed == [[f"{HARNESS_ROOT}/guards/{SEED_RULES.name}"]]
    assert edit.files == (f"{HARNESS_ROOT}/guards/{SEED_RULES.name}",)
    remaining = repo / HARNESS_ROOT / "guards" / SEED_RULES.name
    if remaining.exists():
        assert rule.id not in {r.id for r in read_rule_file(remaining).rules}
