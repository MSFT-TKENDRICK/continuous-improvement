from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.contracts import SPAN_LINT
from ci_lab.lint import cli as lint_cli
from ci_lab.lint.engine import format_json, format_text, glob_match, lint, run
from ci_lab.lint.spec import load_rules, parse_rules, rule_files

REPO = Path(__file__).resolve().parents[3]

RULES = """\
schema_version: 1
rules:
  - id: t.no-print
    kind: banned_call
    names: [print]
    include: ["src/**/*.py"]
    exclude: ["src/ok/**"]
    message: no print
    fix: use logging
    see: design §13.1 R5
  - id: t.short
    kind: max_lines
    max: 3
    severity: warn
    include: ["**/*.py"]
    message: too long
    fix: split
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "lint" / "rules").mkdir(parents=True)
    (tmp_path / "lint" / "rules" / "t.yaml").write_text(RULES, encoding="utf-8")
    (tmp_path / "src" / "ok").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
    (tmp_path / "src" / "ok" / "b.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "src" / "c.py").write_text("1\n2\n3\n4\n", encoding="utf-8")
    return tmp_path


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.hooksPath=", *args], capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize(("path", "globs", "ok"), [
    ("src/ci_lab/telemetry/core.py", ["src/ci_lab/telemetry/**"], True),
    ("src/ci_lab/obs.py", ["src/ci_lab/telemetry/**"], False),
    ("a.py", ["**/*.py"], True),
    ("x/y/a.py", ["**/*.py"], True),
    ("x/y/a.py", ["x/*.py"], False),
    (".github/workflows/l.yml", [".github/workflows/*.{yml,yaml}"], True),
    (".github/workflows/l.json", [".github/workflows/*.{yml,yaml}"], False),
])
def test_glob_match(path, globs, ok):
    assert glob_match(path, globs) is ok


def test_lint_text_format_and_exit(repo):
    res = run(repo)
    assert [(f.rule, f.path, f.line) for f in res.findings] == [
        ("t.no-print", "src/a.py", 2), ("t.short", "src/c.py", 4)]
    assert (res.errors, res.warnings, res.exit_code) == (1, 1, 1)
    out = format_text(res).splitlines()
    assert out[:4] == ["[LINT][ERROR] src/a.py:2", "  Violation: [t.no-print] no print (call print)",
                       "  Fix: use logging", "  See: design §13.1 R5"]
    assert out[4] == "[LINT][WARN] src/c.py:4"
    assert out[-1].startswith("[LINT] Failed with 1 error(s), 1 warning(s) (")
    doc = json.loads(format_json(res))
    assert doc["errors"] == 1 and doc["findings"][0]["rule"] == "t.no-print"


def test_warn_only_passes(repo):
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    res = run(repo)
    assert res.exit_code == 0 and res.warnings == 1
    assert format_text(res).splitlines()[-1].startswith("[LINT] Passed with 1 warning(s)")


def test_paths_scope(repo):
    assert [f.path for f in run(repo, paths=[repo / "src" / "c.py"]).findings] == ["src/c.py"]
    assert {f.path for f in run(repo, paths=[repo / "src"]).findings} == {"src/a.py", "src/c.py"}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_staged_reads_index_not_worktree(repo):
    git(repo, "init", "-q")
    git(repo, "add", "lint", "src/ok")
    git(repo, "commit", "-q", "-m", "init")
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    git(repo, "add", "src/a.py")
    (repo / "src" / "a.py").write_text("print(2)\n", encoding="utf-8")  # unstaged violation: ignored
    assert run(repo, staged=True).findings == []
    git(repo, "add", "src/a.py")
    res = run(repo, staged=True)
    assert [(f.rule, f.path) for f in res.findings] == [("t.no-print", "src/a.py")]
    assert res.files == 1


def test_run_emits_lint_span(repo, monkeypatch):
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    run(repo)
    (sp,) = [s for s in exp.get_finished_spans() if s.name == SPAN_LINT]
    assert sp.attributes["ci.lint.mode"] == "repo"
    assert sp.attributes["ci.lint.errors"] == 1 and sp.attributes["ci.lint.files"] == 3


def test_cli_exit_codes(repo, capsys):
    assert lint_cli.main(["lint", "--root", str(repo), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"] == 1
    (repo / "lint" / "rules" / "bad.yaml").write_text("schema_version: 1\nrules: [{id: x}]\n", encoding="utf-8")
    assert lint_cli.main(["lint", "--root", str(repo)]) == 2
    assert "rule load" in capsys.readouterr().err


def test_no_inline_suppression(repo):
    (repo / "src" / "a.py").write_text("print(1)  # noqa  # lint: disable=t.no-print\n", encoding="utf-8")
    assert run(repo).errors == 1


# ---------------------------------------------------------------- the real repo


def test_seed_rules_load_and_cite_design():
    rules = load_rules(rule_files(REPO))
    assert len(rules) >= 12
    assert all(r.see for r in rules), "every seed rule must point at a design id"
    kinds = {r.kind for r in rules}
    assert {"banned_call", "banned_import", "banned_attr_arg", "banned_text", "gha_banned_trigger",
            "gha_pinned_sha", "declarative_yaml_expression_free"} <= kinds


def test_repo_lints_clean_and_fast():
    t0 = time.perf_counter()
    res = run(REPO)
    assert time.perf_counter() - t0 < 5.0
    assert res.findings == [], format_text(res)


def test_seed_rules_catch_known_lessons(tmp_path):
    rules = load_rules(rule_files(REPO))
    files = {
        "src/ci_lab/campaign/x.py": "from opentelemetry import trace\ntrace.set_tracer_provider(tp)\n",
        "src/ci_lab/guards/x.py": "from order_support.oracle import grade\n",
        "src/ci_lab/rules/x.py": "import re\nre.compile('a')\n",
        "src/ci_lab/lessons/x.py": "P = 'datasets/heldout/cases.jsonl'\n",
        "src/ci_lab/sleep/x.py": "from ci_lab import obs\nobs.span('ci.x', {'a': str(o)})\n",
        "src/ci_lab/meta/x.py": "Path(run_dir, 'status.json').write_text(s)\n",
        "src/ci_lab/x.py": "import pythonnet\n",
        ".github/workflows/x.yml": "on: pull_request_target\njobs:\n  a:\n    steps:\n      - uses: actions/checkout@v4\n",
        ".github/extensions/x/extension.mjs": "console.log('hi');\n",
        "src/ci_lab/workflows/x.yaml": "kind: Workflow\nx: =Local.a\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    res = lint(tmp_path, rules, sorted(files))
    flagged = {f.path for f in res.findings if f.severity == "error"}
    assert flagged == set(files), format_text(res)


def test_rule_ids_unique_across_files(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    one = "schema_version: 1\nrules: [{id: x.y.z, kind: max_lines, max: 1, include: ['**'], message: m, fix: f}]\n"
    a.write_text(one, encoding="utf-8")
    b.write_text(one, encoding="utf-8")
    with pytest.raises(Exception, match="x.y.z"):
        load_rules([a, b])
    assert parse_rules(one)[0].id == "x.y.z"
