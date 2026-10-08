from __future__ import annotations

import textwrap

import pytest

from ci_lab.lint.checks import Source, run_check
from ci_lab.lint.spec import RuleLoadError, parse_rules


def rule(kind: str, **kw):
    base = {"id": "t.rule", "kind": kind, "include": ["**"], "message": "m", "fix": "f"}
    base.update(kw)
    import yaml
    return parse_rules(yaml.safe_dump({"schema_version": 1, "rules": [base]}))[0]


def hits(r, path: str, text: str) -> list[tuple[int, str]]:
    return [(h.line, h.detail) for h in run_check(r, Source(path, textwrap.dedent(text)))]


# ---------------------------------------------------------------- schema


def test_schema_rejects_unknown_keys_and_bad_values():
    with pytest.raises(RuleLoadError):
        rule("banned_call", names=["x"], surprise=1)
    with pytest.raises(RuleLoadError):
        rule("banned_text", pattern="(a)\\1")  # backreference: RE2 rejects
    with pytest.raises(RuleLoadError):
        rule("banned_text", pattern="a" * 300)
    with pytest.raises(RuleLoadError):
        rule("max_lines", max=0)
    with pytest.raises(RuleLoadError):
        rule("banned_call", names=["x"], include=["../etc/*"])
    with pytest.raises(RuleLoadError):
        rule("nope")
    with pytest.raises(RuleLoadError):
        parse_rules("schema_version: 1\nrules:\n  - {id: a.b.c, kind: max_lines, max: 3, include: ['**'], "
                    "message: m, fix: f}\n  - {id: a.b.c, kind: max_lines, max: 3, include: ['**'], "
                    "message: m, fix: f}\n")


def test_severity_default_error_and_warn():
    assert rule("max_lines", max=1).severity == "error"
    assert rule("max_lines", max=1, severity="warn").severity == "warn"


# ---------------------------------------------------------------- banned_call


SET_PROVIDER = ["opentelemetry.trace.set_tracer_provider", "*.set_tracer_provider"]


@pytest.mark.parametrize("code", [
    "from opentelemetry import trace\ntrace.set_tracer_provider(p)\n",
    "import opentelemetry.trace as t\nt.set_tracer_provider(p)\n",
    "from opentelemetry.trace import set_tracer_provider as stp\nstp(p)\n",
    "import opentelemetry\nopentelemetry.trace.set_tracer_provider(p)\n",
    "x = get_api()\nx.set_tracer_provider(p)\n",
])
def test_banned_call_resolves_aliases(code):
    assert hits(rule("banned_call", names=SET_PROVIDER), "a.py", code)


def test_banned_call_ignores_other_calls_and_definitions():
    code = "def set_tracer_provider(p):\n    pass\ntrace.get_tracer_provider()\nset_tracer_provider_x()\n"
    assert hits(rule("banned_call", names=SET_PROVIDER), "a.py", code) == []


def test_banned_call_exact_name_is_not_suffix():
    r = rule("banned_call", names=["re.compile"])
    assert hits(r, "a.py", "import re2\nre2.compile('x')\n") == []
    assert hits(r, "a.py", "from re import compile as c\nc('x')\n") == [(2, "call c")]


def test_syntax_error_files_are_skipped():
    assert hits(rule("banned_call", names=["x"]), "a.py", "def (:\n") == []


# ---------------------------------------------------------------- banned_import


def test_banned_import_forms():
    r = rule("banned_import", modules=["order_support.oracle", "assert_ai", "ci_lab.judge"])
    code = """\
        import assert_ai.core
        from order_support import oracle
        from order_support.oracle import rules
        from ..judge import s1
        import importlib
        importlib.import_module("ci_lab.judge.x")
        __import__("assert_ai")
        import assert_aix
        from order_support import tools
        """
    got = hits(r, "src/ci_lab/guards/mw.py", code)
    assert [line for line, _ in got] == [1, 2, 3, 4, 6, 7]


# ---------------------------------------------------------------- banned_attr_arg


ATTR_CALLS = ["set_attribute", "set_attributes", "span", "annotate", "add_event"]


def test_banned_attr_arg_flags_stringified_objects():
    r = rule("banned_attr_arg", calls=ATTR_CALLS)
    code = """\
        s.set_attribute("k", str(order))
        obs.span("ci.step", {"ci.x": repr(obj)})
        obs.annotate({"a": f"{customer}"})
        s.add_event("e", attributes={"a": "{}".format(x)})
        s.set_attributes({"a": "%s" % x})
        obs.span("ci.case", {"ci.case_id": case_id, "n": 3, "s": f"literal"})
        obs.span(f"ci.step.{phase}")
        s.set_attribute("k", value)
        log(str(x))
        """
    assert [line for line, _ in hits(r, "a.py", code)] == [1, 2, 3, 4, 5]


def test_banned_attr_arg_subset():
    r = rule("banned_attr_arg", calls=["set_attribute"], banned=["repr"])
    assert hits(r, "a.py", "s.set_attribute('k', str(x))\ns.set_attribute('k', repr(x))\n") == \
        [(2, "set_attribute(... repr() ...)")]


# ---------------------------------------------------------------- text


def test_banned_text_reports_line_numbers_once_per_line():
    r = rule("banned_text", pattern=r"\bconsole\.(?:log|info)\s*\(")
    code = "const a = 1;\nconsole.log(a); console.log(b);\n// ok: console.error(x)\nconsole.info (x)\n"
    assert hits(r, "x.mjs", code) == [(2, ""), (4, "")]


# ---------------------------------------------------------------- GitHub Actions


def test_gha_banned_trigger_forms():
    r = rule("gha_banned_trigger", triggers=["pull_request_target"])
    assert hits(r, "w.yml", "on: pull_request_target\njobs: {}\n") == [(1, "on: pull_request_target")]
    assert hits(r, "w.yml", "on: [push, pull_request_target]\n")
    doc = "# never pull_request_target\non:\n  pull_request:\n  pull_request_target:\n    types: [opened]\n"
    assert hits(r, "w.yml", doc) == [(4, "on: pull_request_target")]
    assert hits(r, "w.yml", "# pull_request_target is banned\non: [pull_request]\n") == []
    assert hits(r, "w.yml", "on: [\n") == [(1, "workflow YAML does not parse")]


def test_gha_pinned_sha():
    r = rule("gha_pinned_sha")
    sha = "a" * 40
    doc = f"""\
        jobs:
          a:
            uses: org/repo/.github/workflows/x.yml@{sha}
            steps:
              - uses: actions/checkout@{sha}  # v4.2.2
              - uses: ./.github/actions/local
              - uses: actions/checkout@v4
              - name: x
                uses: "astral-sh/setup-uv@main"
              - uses: docker://alpine:3
              - uses: docker://alpine@sha256:{"b" * 64}
              - uses: actions/checkout@{sha[:39]}
        """
    assert [line for line, _ in hits(r, "w.yml", doc)] == [7, 9, 10, 12]


# ---------------------------------------------------------------- declarative YAML


def test_declarative_yaml_expression_free():
    r = rule("declarative_yaml_expression_free")
    doc = """\
        kind: Workflow
        trigger:
          kind: OnConversationStart
          actions:
            - kind: InvokeFunctionTool
              id: a
              arguments: {x: "=Local.y"}
            - kind: ConditionGroup
              id: b
            - kind: SendActivity
              text: "a = b is fine"
        """
    assert hits(r, "src/w.yaml", doc) == [(7, "PowerFx expression ('=' prefix)"), (8, "kind: ConditionGroup")]
    assert hits(r, "src/c.yaml", "kind: Config\nx: '=not declarative'\n") == []
    assert hits(r, "src/w.yaml", "kind: Workflow\nx: [\n") == [(1, "declarative YAML does not parse")]


def test_max_lines():
    r = rule("max_lines", max=2)
    assert hits(r, "a.py", "1\n2\n") == []
    assert hits(r, "a.py", "1\n2\n3") == [(3, "3 lines > 2")]
