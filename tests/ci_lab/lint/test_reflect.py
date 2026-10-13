from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from ci_lab.lint import cli as lint_cli
from ci_lab.lint.reflect import (
    ReflectError,
    aggregate,
    file_glob,
    has_pii,
    load_templates,
    reflect_main,
    revert_targets,
    scan_session,
)
from ci_lab.rulespec import CorrectionRecord

# Synthetic markers: if any of these strings shows up in an output, raw text leaked.
SECRET = "ghp_" + "Z" * 36
EMAIL = "pat.doe@contoso-example.com"
CANARY_USER = "CANARY_USERTEXT_7731"
CANARY_CMD = "CANARY_CMDARG_4410"
CANARY_OUT = "CANARY_TOOLOUT_9902"
CANARIES = (SECRET, EMAIL, CANARY_USER, CANARY_CMD, CANARY_OUT, "contoso", "Z" * 20)

RULES = """\
schema_version: 1
rules:
  - id: obs.single-tracer-provider
    kind: banned_call
    names: [opentelemetry.trace.set_tracer_provider]
    include: ["src/**/*.py"]
    message: m
    fix: f
"""


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "lint" / "rules").mkdir(parents=True)
    (root / "lint" / "rules" / "seed.yaml").write_text(RULES, encoding="utf-8")
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    return root


def ev(typ: str, **data) -> dict:
    return {"id": "e", "parentId": None, "timestamp": "2026-01-01T00:00:00Z", "type": typ, "data": data}


def session_events(root: Path, *, variant: int = 0) -> list[dict]:
    mod = str(root / "src" / "pkg" / "mod.py")
    lint_out = ("[LINT][ERROR] src/pkg/mod.py:3\n  Violation: [obs.single-tracer-provider] m (call x)\n"
                f"  Fix: f\n[LINT] Failed with 1 error(s) {CANARY_OUT}\n<exited with exit code 1>")
    return [
        ev("session.start", context={"cwd": str(root), "gitRoot": str(root), "branch": "dev/x"}),
        ev("user.message", content=f"please refactor {CANARY_USER}", transformedContent="x"),
        ev("tool.execution_start", toolName="edit", toolCallId="1", arguments={"path": mod, "old_str": CANARY_CMD}),
        ev("tool.execution_complete", toolCallId="1", success=True, result={"content": "ok"}),
        ev("tool.execution_start", toolName="powershell", toolCallId="2",
           arguments={"command": f"uv run ci-lab lint  # {CANARY_CMD}"}),
        ev("tool.execution_complete", toolCallId="2", success=True, result={"content": lint_out}),
        ev("tool.execution_start", toolName="edit", toolCallId="3", arguments={"path": mod, "new_str": "y"}),
        ev("tool.execution_complete", toolCallId="3", success=False, result={"content": "boom"}),
        ev("tool.execution_start", toolName="edit", toolCallId="4", arguments={"path": mod, "new_str": "z"}),
        ev("user.message", content=f"Don't use time.sleep here again {CANARY_USER}"),
        ev("user.message", content=f"never print secrets like {SECRET} — stop using print"),
        ev("user.message", content=f"mail {EMAIL} and stop using print"),
        ev("tool.execution_start", toolName="bash", toolCallId="5",
           arguments={"command": f"git checkout -- src/pkg/mod.py && echo {CANARY_CMD}"}),
        ev("tool.execution_complete", toolCallId="5", success=True, result={"content": f"token {SECRET}"}),
        ev("tool.execution_start", toolName="view", toolCallId="6", arguments={"path": "C:/elsewhere/x.py"}),
        ev("assistant.message", content=f"assistant prose {CANARY_USER}", toolRequests=[]),
    ][: None if variant == 0 else 12]


def write_session(sessions: Path, name: str, events: list[dict], *, workspace_cwd: Path | None = None) -> None:
    d = sessions / name
    d.mkdir(parents=True)
    with (d / "events.jsonl").open("w", encoding="utf-8") as f:
        f.write("not json\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
    if workspace_cwd is not None:
        (d / "workspace.yaml").write_text(f"id: {name}\ncwd: {workspace_cwd}\n", encoding="utf-8")


def all_text(paths: list[Path]) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths if p.is_file())


# ---------------------------------------------------------------- units


def test_templates_are_valid_lint_rules():
    t = load_templates()
    assert {"no-sleep", "no-print", "no-console-log", "no-stdlib-re", "pin-actions-sha"} <= set(t)


def test_file_glob_generalizes_and_bounds(repo):
    assert file_glob(str(repo / "src" / "pkg" / "mod.py"), repo, repo) == "src/pkg/*.py"
    assert file_glob("src/pkg/new.py", repo, repo) == "src/pkg/*.py"
    assert file_glob("README.md", repo, repo) == "*.md"
    assert file_glob("src/nope/x.py", repo, repo) is None  # dir must exist in the repo
    assert file_glob("../outside.py", repo, repo) is None
    assert file_glob(str(repo.parent / "x.py"), repo, repo) is None


def test_revert_targets():
    assert revert_targets("git checkout -- a.py b.py; git status") == (["a.py", "b.py"], False)
    assert revert_targets("git restore --source HEAD~1 a.py") == (["a.py"], False)
    assert revert_targets("git restore --staged a.py") == ([], False)
    assert revert_targets("git reset --hard HEAD") == ([], True)
    assert revert_targets("git checkout main") == ([], False)


def test_pii_scan():
    assert has_pii(SECRET) and has_pii(EMAIL) and has_pii("call 425-555-0100")
    assert has_pii("api_key = abcdef123456")
    assert not has_pii("please stop using print in src/ci_lab")


def test_scan_session_detectors_and_privacy(repo, tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in session_events(repo)), encoding="utf-8")
    scan = scan_session(p, root=repo, cwd=repo, known_rules={"obs.single-tracer-provider"},
                        templates=load_templates())
    got = {(r.kind, r.tool, r.rule_hint, r.file_glob, r.count) for r in scan.records}
    assert got == {
        ("lint_fail", "ci-lab-lint", "obs.single-tracer-provider", "src/pkg/*.py", 1),
        ("revert", "edit", "obs.single-tracer-provider", "src/pkg/*.py", 1),
        ("repeated_fix", "edit", "obs.single-tracer-provider", "src/pkg/*.py", 2),
        ("user_correction", None, "no-sleep", None, 1),
    }
    assert scan.dropped == 3  # secret in user msg, email in user msg, secret in tool output
    dumped = json.dumps([r.model_dump() for r in scan.records])
    assert not any(c in dumped for c in CANARIES)


def test_aggregate_counts_sessions():
    a = CorrectionRecord(kind="user_correction", rule_hint="no-print", count=2)
    b = CorrectionRecord(kind="user_correction", rule_hint="no-print", count=1)
    (ls,) = aggregate([[a], [b], []])
    assert (ls.record.count, ls.sessions) == (3, 2)


# ---------------------------------------------------------------- end to end


def test_reflect_requires_consent(repo, tmp_path, capsys):
    rc = lint_cli.main(["reflect", "--source", "copilot-sessions", "--sessions-dir", str(tmp_path),
                        "--root", str(repo)])
    assert rc == 2 and "--i-consent-local-mining" in capsys.readouterr().err
    assert not (repo / "artifacts").exists()


def test_reflect_convergence_and_no_leaks(repo, tmp_path, capsys):
    sessions = tmp_path / "sessions"
    write_session(sessions, "s1", session_events(repo))
    write_session(sessions, "s2", session_events(repo, variant=1))
    write_session(sessions, "s3", session_events(repo), workspace_cwd=repo)
    other = tmp_path / "other-repo"
    (other / "src" / "pkg").mkdir(parents=True)
    write_session(sessions, "s4", session_events(other))
    (sessions / "empty").mkdir()

    rc = lint_cli.main(["reflect", "--source", "copilot-sessions", "--i-consent-local-mining",
                        "--sessions-dir", str(sessions), "--root", str(repo), "--keep"])
    assert rc == 0
    stdout = capsys.readouterr().out
    out = repo / "artifacts" / "reflect" / "proposals.yaml"
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["sessions"] == {"scanned": 4, "in_repo": 3}
    lessons = {(ls["kind"], ls.get("rule_hint"), ls.get("file_glob")): ls for ls in doc["lessons"]}
    assert lessons[("user_correction", "no-sleep", None)]["sessions"] == 3
    assert lessons[("lint_fail", "obs.single-tracer-provider", "src/pkg/*.py")]["proposal"] == \
        "existing:obs.single-tracer-provider"
    assert ("revert", "obs.single-tracer-provider", "src/pkg/*.py") in lessons  # sessions s1 + s3 only
    rules = doc["proposed_rules"]["rules"]
    assert [r["id"] for r in rules] == ["lesson.no-sleep"]
    assert rules[0]["severity"] == "warn" and rules[0]["names"] == ["time.sleep"]
    # no-print appears only alongside PII/secrets, which are dropped whole
    assert all(ls.get("rule_hint") != "no-print" for ls in doc["lessons"])
    produced = all_text([out, out.parent / "records.jsonl"]) + stdout
    assert not any(c in produced for c in CANARIES)
    assert "other-repo" not in produced


def test_reflect_single_session_does_not_converge(repo, tmp_path, capsys):
    sessions = tmp_path / "sessions"
    write_session(sessions, "s1", session_events(repo))
    out = tmp_path / "p.yaml"
    assert reflect_main(root=repo, sessions_dir=sessions, out=out, today="2026-01-02") == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["lessons"] == [] and doc["proposed_rules"]["rules"] == []
    assert doc["below_convergence"] >= 4
    with pytest.raises(ReflectError):
        reflect_main(root=repo, sessions_dir=sessions, out=out, min_sessions=1)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_reflect_draft_pr_branch(repo, tmp_path, capsys):
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                               *a], capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q", "-b", "dev/x")
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    for k in ("user.name", "user.email"):
        git("config", k, "t" if k == "user.name" else "t@example.invalid")
    head = git("rev-parse", "HEAD")
    sessions = tmp_path / "sessions"
    write_session(sessions, "s1", session_events(repo))
    write_session(sessions, "s2", session_events(repo))
    assert reflect_main(root=repo, sessions_dir=sessions, draft_pr=True, today="2026-01-03") == 0
    stdout = capsys.readouterr().out
    assert git("rev-parse", "HEAD") == head and git("status", "--porcelain") == ""
    assert "gh pr create --draft --base dev/x --head lessons/reflect-2026-01-03" in stdout
    files = git("ls-tree", "-r", "--name-only", "lessons/reflect-2026-01-03").splitlines()
    assert "lint/rules/lessons-2026-01-03.yaml" in files
    proposed = git("show", "lessons/reflect-2026-01-03:lint/rules/lessons-2026-01-03.yaml")
    from ci_lab.lint.spec import parse_rules
    assert [r.id for r in parse_rules(proposed)] == ["lesson.no-sleep"]
    body = (repo / "artifacts" / "reflect" / "pr-body.md").read_text(encoding="utf-8")
    assert not any(c in proposed + body + stdout for c in CANARIES)
    with pytest.raises(ReflectError, match="already exists"):
        reflect_main(root=repo, sessions_dir=sessions, draft_pr=True, today="2026-01-03")
    assert not list((repo / "artifacts" / "reflect").glob(".index-*"))
