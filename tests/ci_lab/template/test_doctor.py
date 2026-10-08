"""`ci-lab template doctor`: offline readiness checks for a repository created from the template."""

from __future__ import annotations

from pathlib import Path

import pytest

from ci_lab.cli import main as cli_main
from ci_lab.template import doctor as tdoc
from ci_lab.template import init as tinit
from ci_lab.template.codeowners import rewrite_owners
from ci_lab.template.marker import (
    FORMAT,
    MARKER_REL,
    TEMPLATE_OWNERS,
    TEMPLATE_REPO,
    expected_owners,
    read_marker,
    render_marker,
)

REPO = Path(__file__).resolve().parents[3]
SUITE = """\
name: s
pipeline:
  test_set:
    taxonomy_path: taxonomy.json
  judge:
    model:
      name: s1/llamacpp/qwen
"""
GATED = """\
on:
  schedule:
    - cron: "0 3 * * *"
jobs:
  opt_in:
    runs-on: ubuntu-latest
    steps:
      - env:
          CI_HARNESS_ENABLED: ${{ vars.CI_HARNESS_ENABLED }}
        run: echo "$CI_HARNESS_ENABLED"
  gate:
    environment: campaign-publish
    runs-on: ubuntu-latest
    steps:
      - env:
          ROUNDS: ${{ vars.CAMPAIGN_ROUNDS }}
        run: echo "$ROUNDS"
"""
DOC = "Settings: `CI_HARNESS_ENABLED`, `CAMPAIGN_ROUNDS`, `campaign-publish`.\n"


def _template_state(root: Path) -> None:
    """Write CODEOWNERS + marker as they are in the template, whether this checkout is the template or a
    repository created from it (then its owners are mapped back to the template owner)."""
    real = read_marker(REPO)
    text = (REPO / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    text = rewrite_owners(text, expected_owners(real), TEMPLATE_OWNERS).text
    (root / ".github").mkdir(parents=True, exist_ok=True)
    (root / ".github" / "CODEOWNERS").write_text(text, encoding="utf-8")
    (root / MARKER_REL).write_text(render_marker({"format": FORMAT, "role": "template", "initialized": False,
                                                  "template_repository": TEMPLATE_REPO,
                                                  "template_owners": list(TEMPLATE_OWNERS)}), encoding="utf-8")


def _derived(root: Path, *, init: bool = True) -> Path:
    (root / ".github" / "workflows").mkdir(parents=True)
    _template_state(root)
    (root / ".github" / "workflows" / "nightly.yml").write_text(GATED, encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "template.md").write_text(DOC, encoding="utf-8")
    suite = root / "evals" / "assert" / "s"
    suite.mkdir(parents=True)
    (suite / "eval_config.yaml").write_text(SUITE, encoding="utf-8")
    (suite / "test_set.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    if init:
        opts = tinit.Options(repo="acme/agent", owners=("@acme/agent-owners",))
        tinit.apply_plan(root, tinit.build_plan(root, opts))
    return root


def _status(checks: list[tdoc.Check], name: str) -> set[str]:
    return {c.status for c in checks if c.name == name}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _derived(tmp_path / "copy")


def test_initialized_copy_is_ready(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checks = tdoc.doctor(repo, skip=("lint",))
    assert all(c.status == "PASS" for c in checks), tdoc.format_checks(checks)
    assert cli_main(["template", "doctor", "--root", str(repo), "--skip-lint"]) == 0
    assert "doctor: ready" in capsys.readouterr().out


def test_uninitialized_copy_fails_with_fixes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _derived(tmp_path / "fresh", init=False)
    checks = tdoc.doctor(root, skip=("lint",))
    assert _status(checks, "marker") == {"FAIL"} and _status(checks, "codeowners") == {"FAIL"}
    assert all(c.fix for c in checks if c.status == "FAIL")
    assert cli_main(["template", "doctor", "--root", str(root), "--skip-lint"]) == 1
    out = capsys.readouterr().out
    assert "NOT READY" in out and "template init" in out


def test_missing_marker_fails(repo: Path) -> None:
    (repo / MARKER_REL).unlink()
    assert _status(tdoc.check_marker(repo), "marker") == {"FAIL"}


def test_broken_marker_fails(repo: Path) -> None:
    (repo / MARKER_REL).write_text("role: nonsense\n", encoding="utf-8")
    assert _status(tdoc.check_marker(repo), "marker") == {"FAIL"}


def test_template_owner_left_in_codeowners_fails(repo: Path) -> None:
    path = repo / ".github" / "CODEOWNERS"
    path.write_text(path.read_text(encoding="utf-8") + "/extra/ @MSFT-TKENDRICK\n", encoding="utf-8")
    checks = tdoc.check_codeowners(repo)
    assert any(c.status == "FAIL" and "template owner" in c.detail for c in checks)


def test_owner_drift_warns(repo: Path) -> None:
    path = repo / ".github" / "CODEOWNERS"
    path.write_text(path.read_text(encoding="utf-8") + "/extra/ @someone-else\n", encoding="utf-8")
    assert _status(tdoc.check_codeowners(repo), "codeowners") == {"WARN"}


def test_undocumented_setting_fails(repo: Path) -> None:
    wf = repo / ".github" / "workflows" / "nightly.yml"
    wf.write_text(wf.read_text(encoding="utf-8").replace("vars.CAMPAIGN_ROUNDS", "vars.NEW_KNOB")
                  + "# ${{ secrets.MY_TOKEN }}\n", encoding="utf-8")
    checks = tdoc.check_settings(repo)
    fail = next(c for c in checks if c.status == "FAIL")
    assert "NEW_KNOB" in fail.detail and "MY_TOKEN" in fail.detail


def test_ungated_schedule_fails(repo: Path) -> None:
    (repo / ".github" / "workflows" / "other.yml").write_text(
        "on:\n  schedule:\n    - cron: '0 1 * * *'\njobs: {}\n", encoding="utf-8")
    checks = tdoc.check_settings(repo)
    assert any(c.status == "FAIL" and "other.yml" in c.detail for c in checks)


def test_missing_or_empty_test_set_fails(repo: Path) -> None:
    ts = repo / "evals" / "assert" / "s" / "test_set.jsonl"
    ts.write_text("\n", encoding="utf-8")
    assert _status(tdoc.check_test_sets(repo), "test-sets") == {"FAIL"}
    ts.unlink()
    checks = tdoc.check_test_sets(repo)
    assert checks[0].status == "FAIL" and "order-support-evals run" in checks[0].fix


def test_unknown_judge_backend_fails(repo: Path) -> None:
    cfg = repo / "evals" / "assert" / "s" / "eval_config.yaml"
    cfg.write_text(SUITE.replace("s1/llamacpp/qwen", "s1/nope/qwen"), encoding="utf-8")
    assert _status(tdoc.check_judge(repo), "judge") == {"FAIL"}
    cfg.write_text(SUITE.replace("s1/llamacpp/qwen", "gpt-x"), encoding="utf-8")
    assert _status(tdoc.check_judge(repo), "judge") == {"WARN"}


def test_template_repo_settings_judge_and_test_sets_pass() -> None:
    """The template's own workflows are gated and documented; its suites have judges and frozen test sets."""
    for check in (tdoc.check_settings, tdoc.check_judge, tdoc.check_test_sets):
        checks = check(REPO)
        assert all(c.status == "PASS" for c in checks), tdoc.format_checks(checks)


def test_this_checkout_marker_matches_its_role() -> None:
    """The template fails the marker check (it is not initialized); an initialized copy passes it."""
    marker = read_marker(REPO)
    statuses = _status(tdoc.check_marker(REPO), "marker")
    if marker is None or marker["role"] == "template":
        assert statuses == {"FAIL"}
    else:
        assert "FAIL" not in statuses
