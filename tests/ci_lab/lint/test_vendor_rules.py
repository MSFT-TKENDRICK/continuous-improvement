from __future__ import annotations

from pathlib import Path

from ci_lab.lint.engine import format_text, lint
from ci_lab.lint.spec import load_rules, rule_files

REPO = Path(__file__).resolve().parents[3]


def _vendor_rules():
    return [r for r in load_rules(rule_files(REPO)) if r.id.startswith("vendor.")]


def test_vendor_rules_flag_the_excluded_stack(tmp_path):
    rules = _vendor_rules()
    assert {r.id for r in rules} == {"vendor.no-langchain-imports", "vendor.no-langchain-references"}
    files = {
        "src/a.py": "from langchain_core.messages import HumanMessage\n",
        "src/b.py": "import langsmith\n",
        "src/c.py": "BASE = 'https://gateway.smith.LangChain.com'\n",
        "pyproject.toml": "[project]\ndependencies = ['langgraph>=1']\n",
        "docs/x.md": "Scores are mirrored to LangSmith.\n",
        ".github/workflows/x.yml": "env:\n  LANGSMITH_API_KEY: x\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    res = lint(tmp_path, rules, sorted(files))
    assert {f.path for f in res.findings if f.severity == "error"} == set(files), format_text(res)


def test_vendor_rules_allow_the_supported_stack(tmp_path):
    files = {"src/a.py": "from agent_framework import Agent\nimport assert_ai\nlanguage = 'en'\n",
             "docs/x.md": "Rubric dimensions are judged by s1/llamacpp/qwen3.5-4b.\n"}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    assert lint(tmp_path, _vendor_rules(), sorted(files)).findings == []
