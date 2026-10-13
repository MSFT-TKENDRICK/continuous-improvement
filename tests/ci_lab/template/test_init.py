"""`ci-lab template init`: dry-run plan, apply, idempotency, refusal in the template, CODEOWNERS rewrite."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ci_lab.cli import main as cli_main
from ci_lab.template import init as tinit
from ci_lab.template.codeowners import OwnersError, parse_owners, rewrite_owners, rules
from ci_lab.template.marker import (
    FORMAT,
    MARKER_REL,
    TEMPLATE_OWNERS,
    TEMPLATE_REPO,
    MarkerError,
    expected_owners,
    parse_marker,
    read_marker,
    render_marker,
)

REPO = Path(__file__).resolve().parents[3]
TODAY = dt.date(2026, 1, 2)
OWNERS = ("@acme/agent-owners", "@alice")


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


def _fake_repo(root: Path) -> Path:
    """A minimal copy of the template: real CODEOWNERS + marker, plus template history to reset."""
    _template_state(root)
    (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8")
    camp = root / "experiments" / "campaigns" / "c1"
    camp.mkdir(parents=True)
    (camp / "stack.json").write_text('{"prs": [42]}\n', encoding="utf-8")
    (camp / "published.jsonl").write_text("{}\n", encoding="utf-8")
    sleep = root / "experiments" / "sleep"
    (sleep / "nights").mkdir(parents=True)
    (sleep / "nights" / "n1.json").write_text("{}\n", encoding="utf-8")
    (sleep / "harvest.pending.jsonl").write_text("{}\n", encoding="utf-8")
    (sleep / "state.json").write_text(json.dumps({**tinit.DEFAULT_SLEEP_STATE, "night": 7, "accepted_total": 3}),
                                      encoding="utf-8")
    (sleep / "tasks.jsonl").write_text(
        json.dumps({"format": tinit.TASKS_FORMAT, "project": "order-support", "reviewed": True}) + "\n"
        + json.dumps({"id": "legacy", "reviewed": True}) + "\n", encoding="utf-8")
    for rel in tinit.HARNESS_SLEEP_TASKS_RELS:
        path = root / rel
        path.write_text(
            json.dumps({"format": tinit.TASKS_FORMAT, "project": "harness", "reviewed": True}) + "\n"
            + json.dumps({"id": "t1", "reviewed": True}) + "\n", encoding="utf-8")
    (root / "experiments" / "holdout-looks.jsonl").write_text('{"dataset": "abc"}\n', encoding="utf-8")
    (root / "lessons").mkdir()
    (root / "lessons" / "registry.yaml").write_text("schema_version: 1\nlessons:\n  - id: L1\n", encoding="utf-8")
    suite = root / "evals" / "assert" / "s"
    suite.mkdir(parents=True)
    (suite / "test_set.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _opts(**kw) -> tinit.Options:
    return tinit.Options(**{"repo": "acme/agent", "owners": OWNERS, "today": TODAY, **kw})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _fake_repo(tmp_path / "copy")


# ---------------------------------------------------------------- plan / apply


def test_dry_run_plans_without_writing(repo: Path) -> None:
    before = _snapshot(repo)
    plan = tinit.build_plan(repo, _opts())
    ops = {(a.op, a.path) for a in plan.actions}
    assert ops == {
        ("write", ".github/CODEOWNERS"),
        ("delete", "experiments/campaigns/"),
        ("delete", "experiments/sleep/nights/"),
        ("delete", "experiments/sleep/harvest.pending.jsonl"),
        ("write", "experiments/sleep/state.json"),
        ("write", "lessons/registry.yaml"),
        ("write", MARKER_REL),
    }
    assert _snapshot(repo) == before
    assert plan.repo == "acme/agent" and plan.template_repository == TEMPLATE_REPO


def test_cli_defaults_to_dry_run(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = _snapshot(repo)
    rc = cli_main(["template", "init", "--root", str(repo), "--repo", "acme/agent", "--owners", *OWNERS])
    out = capsys.readouterr().out
    assert rc == 0 and "dry run: nothing written" in out and "experiments/campaigns/" in out
    assert _snapshot(repo) == before


def test_apply_resets_history_and_keeps_example(repo: Path) -> None:
    tinit.apply_plan(repo, tinit.build_plan(repo, _opts()))
    assert not (repo / "experiments" / "campaigns").exists()
    assert not (repo / "experiments" / "sleep" / "nights").exists()
    assert not (repo / "experiments" / "sleep" / "harvest.pending.jsonl").exists()
    assert json.loads((repo / "experiments" / "sleep" / "state.json").read_text()) == tinit.DEFAULT_SLEEP_STATE
    assert (repo / "lessons" / "registry.yaml").read_text() == tinit.EMPTY_REGISTRY
    # kept by default: reviewed example tasks, the held-out look ledger, suites and frozen test sets
    assert all("t1" in (repo / rel).read_text() for rel in tinit.HARNESS_SLEEP_TASKS_RELS)
    assert "legacy" in (repo / tinit.SLEEP_TASKS_REL).read_text()
    assert (repo / "experiments" / "holdout-looks.jsonl").is_file()
    assert (repo / "evals" / "assert" / "s" / "test_set.jsonl").is_file()

    marker = read_marker(repo)
    assert marker is not None
    assert marker["role"] == "derived" and marker["initialized"] is True
    assert marker["repository"] == "acme/agent" and marker["owners"] == list(OWNERS)
    assert marker["template_repository"] == TEMPLATE_REPO and marker["template_owners"] == list(TEMPLATE_OWNERS)
    assert marker["template_version"] == "1.2.3" and marker["initialized_on"] == "2026-01-02"
    assert marker["template_commit"] == "unknown" and marker["reset_state"] is False
    assert expected_owners(marker) == OWNERS


def test_apply_is_idempotent(repo: Path) -> None:
    tinit.apply_plan(repo, tinit.build_plan(repo, _opts()))
    after = _snapshot(repo)
    later = tinit.Options(repo="acme/agent", today=dt.date(2027, 5, 5))  # owners come from the marker
    assert tinit.build_plan(repo, later).actions == []
    assert tinit.build_plan(repo, _opts()).actions == []
    rc = cli_main(["template", "init", "--root", str(repo), "--repo", "acme/agent", "--apply"])
    assert rc == 0 and _snapshot(repo) == after


def test_rerun_with_new_owners_rewrites_previous_owners(repo: Path) -> None:
    tinit.apply_plan(repo, tinit.build_plan(repo, _opts()))
    plan = tinit.build_plan(repo, _opts(owners=("@acme/platform",), today=dt.date(2026, 3, 3)))
    assert {a.path for a in plan.actions} == {".github/CODEOWNERS", MARKER_REL}
    tinit.apply_plan(repo, plan)
    text = (repo / ".github" / "CODEOWNERS").read_text()
    assert {tuple(o) for _, o in rules(text)} == {("@acme/platform",)}
    marker = read_marker(repo)
    assert marker["initialized_on"] == "2026-01-02" and marker["updated_on"] == "2026-03-03"


def test_codeowners_rewrite_satisfies_test_codeowners(repo: Path) -> None:
    """The rewritten file still passes tests/ci_lab/lint/test_codeowners.py's rule/owner checks."""
    original = rules((repo / ".github" / "CODEOWNERS").read_text(encoding="utf-8"))
    tinit.apply_plan(repo, tinit.build_plan(repo, _opts()))
    text = (repo / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    rewritten = rules(text)
    assert [p for p, _ in rewritten] == [p for p, _ in original]
    owners = list(expected_owners(read_marker(repo)))
    for pattern, rule_owners in rewritten:
        assert rule_owners == owners, pattern
    assert "@MSFT-TKENDRICK" not in "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def test_crlf_files_keep_crlf(repo: Path) -> None:
    path = repo / ".github" / "CODEOWNERS"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    tinit.apply_plan(repo, tinit.build_plan(repo, _opts()))
    data = path.read_bytes()
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
    assert tinit.build_plan(repo, _opts()).actions == []


def test_customized_rules_are_kept(repo: Path) -> None:
    path = repo / ".github" / "CODEOWNERS"
    text = path.read_text(encoding="utf-8").replace("/.github/CODEOWNERS                     @MSFT-TKENDRICK",
                                                    "/.github/CODEOWNERS                     @security-team")
    path.write_text(text, encoding="utf-8")
    plan = tinit.build_plan(repo, _opts())
    assert any("kept customized owners on /.github/CODEOWNERS" in n for n in plan.notes)
    tinit.apply_plan(repo, plan)
    assert dict(rules(path.read_text()))["/.github/CODEOWNERS"] == ["@security-team"]


def test_reset_state_also_empties_example_state(repo: Path) -> None:
    plan = tinit.build_plan(repo, _opts(reset_state=True))
    assert any("held-out looks" in n for n in plan.notes)
    tinit.apply_plan(repo, plan)
    for rel in tinit.HARNESS_SLEEP_TASKS_RELS:
        lines = (repo / rel).read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["project"] == "harness"
    assert "legacy" in (repo / tinit.SLEEP_TASKS_REL).read_text()
    assert not (repo / "experiments" / "holdout-looks.jsonl").exists()
    assert (repo / "evals" / "assert" / "s" / "test_set.jsonl").is_file()
    assert read_marker(repo)["reset_state"] is True
    assert tinit.build_plan(repo, tinit.Options(repo="acme/agent", today=TODAY)).actions == []


def test_json_format(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["template", "init", "--root", str(repo), "--repo", "acme/agent", "--owners", "@acme/a,@bob",
                   "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["applied"] is False and data["owners"] == ["@acme/a", "@bob"]
    assert all("content" not in a for a in data["actions"])


# ---------------------------------------------------------------- refusals / inputs


def test_refuses_in_template_repo(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = _snapshot(repo)
    with pytest.raises(tinit.Refused, match="template repository"):
        tinit.build_plan(repo, _opts(repo=TEMPLATE_REPO))
    with pytest.raises(tinit.Refused):
        tinit.build_plan(repo, _opts(repo=TEMPLATE_REPO.lower()))
    rc = cli_main(["template", "init", "--root", str(repo), "--repo", TEMPLATE_REPO, "--owners", *OWNERS,
                   "--apply"])
    assert rc == 2 and "REFUSED" in capsys.readouterr().err
    assert _snapshot(repo) == before


def test_force_overrides_refusal(repo: Path) -> None:
    plan = tinit.build_plan(repo, _opts(repo=TEMPLATE_REPO, force=True))
    assert any(a.path == MARKER_REL for a in plan.actions)


def test_refuses_template_owner(repo: Path) -> None:
    with pytest.raises(tinit.Refused, match="template owner"):
        tinit.build_plan(repo, _opts(owners=("@MSFT-TKENDRICK", "@acme/x")))
    assert tinit.build_plan(repo, _opts(owners=("@MSFT-TKENDRICK",), force=True)).actions


@pytest.mark.parametrize("owners", [(), ("not-an-owner",), ("@bad owner!",)])
def test_requires_valid_owners(repo: Path, owners: tuple[str, ...]) -> None:
    with pytest.raises(tinit.InitError):
        tinit.build_plan(repo, _opts(owners=owners))


def test_requires_repo_without_origin(repo: Path) -> None:
    with pytest.raises(tinit.InitError, match="--repo"):
        tinit.build_plan(repo, _opts(repo=None))
    with pytest.raises(tinit.InitError, match="owner/name"):
        tinit.build_plan(repo, _opts(repo="not a repo"))


@pytest.mark.parametrize(("url", "repo"), [
    ("https://github.com/acme/agent.git", "acme/agent"),
    ("https://github.com/acme/agent", "acme/agent"),
    ("https://github.com/acme/agent/", "acme/agent"),
    ("git@github.com:acme/my.agent.git", "acme/my.agent"),
    ("ssh://git@github.com/acme/agent.git", "acme/agent"),
    ("https://x-access-token@github.com/MSFT-TKENDRICK/continuous-improvement.git", TEMPLATE_REPO),
    ("not a url", None),
])
def test_repo_from_remote(url: str, repo: str | None) -> None:
    assert tinit.repo_from_remote(url) == repo


def test_repo_defaults_to_origin_remote(repo: Path) -> None:
    if not shutil.which("git"):
        pytest.skip("git not installed")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "git@github.com:acme/agent.git"], check=True)
    assert tinit.build_plan(repo, _opts(repo=None)).repo == "acme/agent"
    subprocess.run(["git", "-C", str(repo), "remote", "set-url", "origin",
                    f"https://github.com/{TEMPLATE_REPO}.git"], check=True)
    with pytest.raises(tinit.Refused):
        tinit.build_plan(repo, _opts(repo=None))


def test_refuses_symlink_escape(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    shutil.rmtree(repo / "experiments" / "campaigns")
    try:
        (repo / "experiments" / "campaigns").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted")
    with pytest.raises(tinit.InitError, match="outside the repository"):
        tinit.build_plan(repo, _opts())
    assert (outside / "keep.txt").is_file()


def test_template_commit_validated(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["template", "init", "--root", str(repo), "--repo", "acme/agent", "--owners", "@acme/a",
                   "--template-commit", "nope"])
    assert rc == 2 and "hex" in capsys.readouterr().err
    plan = tinit.build_plan(repo, _opts(template_commit="abc1234"))
    marker = next(a for a in plan.actions if a.path == MARKER_REL)
    assert parse_marker(marker.content)["template_commit"] == "abc1234"


# ---------------------------------------------------------------- building blocks


def test_marker_roundtrip_and_errors() -> None:
    data = {"format": "ci-lab.template/1", "role": "derived", "initialized": True, "repository": "a/b",
            "owners": ["@a"], "template_repository": TEMPLATE_REPO, "template_owners": list(TEMPLATE_OWNERS)}
    assert parse_marker(render_marker(data)) == data
    for bad in ("role: derived\n", "format: \"x\"\nrole: \"nope\"\n", "just text\n"):
        with pytest.raises(MarkerError):
            parse_marker(bad)


def test_repo_marker_is_the_template() -> None:
    marker = read_marker(REPO)
    assert marker is not None and marker["role"] in ("template", "derived")
    if marker["role"] == "template":
        assert marker["initialized"] is False and marker["template_repository"] == TEMPLATE_REPO


def test_package_metadata_defaults_to_the_harness() -> None:
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "ci-lab-harness"
    assert "order-support" not in project["description"].lower()


def test_parse_owners() -> None:
    assert parse_owners(["@a/b, @c", "d@example.com", "@c"]) == ("@a/b", "@c", "d@example.com")
    with pytest.raises(OwnersError):
        parse_owners(["@"])


def test_rewrite_preserves_layout() -> None:
    text = "# c\n/a   @old  # why\n/b @custom\n\n/c @old @old2\n"
    rw = rewrite_owners(text, {"@old", "@old2"}, ["@new"])
    assert rw.text == "# c\n/a   @new  # why\n/b @custom\n\n/c @new\n"
    assert rw.kept_custom == ["/b"] and len(rw.changed) == 2


def test_script_is_stdlib_only(repo: Path) -> None:
    """scripts/template_init.py runs in the write job with system Python: no site-packages, no uv."""
    script = REPO / "scripts" / "template_init.py"
    shutil.copytree(REPO / "src" / "ci_lab" / "template", repo / "src" / "ci_lab" / "template")
    (repo / "src" / "ci_lab" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "scripts").mkdir()
    shutil.copy(script, repo / "scripts" / "template_init.py")
    # ci_lab/__init__.py may import third-party packages; the copy has an empty one to prove we don't need them
    p = subprocess.run([sys.executable, "-I", "-B", "-S", str(repo / "scripts" / "template_init.py"),
                        "--repo", "acme/agent", "--owners", "@acme/a", "--apply"],
                       capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0, p.stderr
    assert read_marker(repo)["initialized"] is True
