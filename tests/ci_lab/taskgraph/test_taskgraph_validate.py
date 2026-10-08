from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from ci_lab.taskgraph.model import (
    Budget,
    ContextRef,
    Criterion,
    OutputSpec,
    Rubric,
    TaskGraph,
)
from ci_lab.taskgraph.validate import (
    Problem,
    rubric_leaks,
    validate_graph,
    validate_rubric,
)
from ci_lab.taskgraph.vault import RubricVault


def codes(problems: list[Problem]) -> set[str]:
    return {p.code for p in problems}


def crit(measure: str, check: dict[str, Any], **kw: Any) -> Criterion:
    base: dict[str, Any] = {"id": "c-extra", "description": "Extra criterion under test", "measure": measure,
                            "check": check, "threshold": 0.5}
    base.update(kw)
    return Criterion(**base)


def with_extra(rubric: Rubric, c: Criterion) -> Rubric:
    return dataclasses.replace(rubric, criteria=(*rubric.criteria, c))


GOOD_CHECKS = [
    crit("deterministic", {"kind": "regex", "pattern": r"ORD-\d+", "negate": False}),
    crit("deterministic", {"kind": "json_schema", "schema": {"type": "object", "required": ["id"]}}),
    crit("deterministic", {"kind": "file_exists", "path": "out/summary.md", "independent": True}),
    crit("deterministic", {"kind": "command", "argv": ["ruff", "check", "."], "timeout_s": 30}),
    crit("deterministic", {"kind": "python", "callable": "ci_lab.tools.critic_checks.run_checks"}),
    crit("assert", {"suite": "order-support", "split": "test", "min_score": 1}),
    crit("llm", {"question": "Which tone does the reply use?", "type": "choice", "options": ["formal", "casual"]}),
    crit("s1", {"question": "Does the reply mention goodwill credit?", "type": "noul"}),
]


def test_base_rubric_is_valid(rubric_factory) -> None:
    assert validate_rubric(rubric_factory()) == []


@pytest.mark.parametrize("c", GOOD_CHECKS, ids=lambda c: f"{c.measure}-{c.check.get('kind', c.check.get('type', 'a'))}")
def test_valid_checks_pass(rubric_factory, c: Criterion) -> None:
    assert validate_rubric(with_extra(rubric_factory(), c)) == []


BAD_RUBRICS: list[tuple[str, dict[str, Any], str]] = [
    ("no criteria", {"criteria": ()}, "rubric.criteria"),
    ("pass 0", {"pass_score": 0}, "rubric.pass_score"),
    ("pass >1", {"pass_score": 1.5}, "rubric.pass_score"),
    ("canary short", {"canary": "abc"}, "rubric.canary"),
    ("canary upper", {"canary": "0123456789ABCDEF"}, "rubric.canary"),
    ("bad id", {"id": "Bad Id"}, "rubric.id"),
    ("version 0", {"version": 0}, "rubric.version"),
    ("bad deliverable", {"deliverable": "Bad"}, "rubric.deliverable"),
]


@pytest.mark.parametrize(("name", "kw", "code"), BAD_RUBRICS, ids=[b[0] for b in BAD_RUBRICS])
def test_rubric_level_rules(rubric_factory, name: str, kw: dict[str, Any], code: str) -> None:
    assert code in codes(validate_rubric(rubric_factory(**kw)))


def test_rubric_needs_an_oracle_and_unique_ids(rubric_factory) -> None:
    r = rubric_factory()
    assert "rubric.oracle" in codes(validate_rubric(dataclasses.replace(r, criteria=r.soft())))
    dup = with_extra(r, dataclasses.replace(r.criteria[0]))
    assert "criterion.duplicate" in codes(validate_rubric(dup))


BAD_CRITERIA: list[tuple[str, Criterion, str]] = [
    ("threshold 0", crit("assert", GOOD_CHECKS[5].to_json()["check"], threshold=0), "criterion.threshold"),
    ("threshold >1", crit("assert", GOOD_CHECKS[5].to_json()["check"], threshold=1.1), "criterion.threshold"),
    ("weight 0", crit("assert", GOOD_CHECKS[5].to_json()["check"], weight=0), "criterion.weight"),
    ("weight <0", crit("assert", GOOD_CHECKS[5].to_json()["check"], weight=-1), "criterion.weight"),
    ("measure", crit("vibes", {}), "criterion.measure"),
    ("description", crit("assert", GOOD_CHECKS[5].to_json()["check"], description=" "), "criterion.description"),
    ("crit id", crit("assert", GOOD_CHECKS[5].to_json()["check"], id="Bad id"), "criterion.id"),
    ("det kind", crit("deterministic", {"kind": "vibe"}), "check.deterministic"),
    ("regex bad", crit("deterministic", {"kind": "regex", "pattern": "("}), "check.deterministic"),
    ("regex key", crit("deterministic", {"kind": "regex", "pattern": "x", "flags": "i"}), "check.deterministic"),
    ("regex negate", crit("deterministic", {"kind": "regex", "pattern": "x", "negate": 1}), "check.deterministic"),
    ("schema", crit("deterministic", {"kind": "json_schema", "schema": {"type": 5}}), "check.deterministic"),
    ("schema type", crit("deterministic", {"kind": "json_schema", "schema": "obj"}), "check.deterministic"),
    ("file path", crit("deterministic", {"kind": "file_exists", "path": "../x"}), "check.deterministic"),
    ("shell string", crit("deterministic", {"kind": "command", "argv": "pytest -q"}), "check.deterministic"),
    ("not allowlisted", crit("deterministic", {"kind": "command", "argv": ["bash", "-c", "x"]}), "check.deterministic"),
    ("empty argv", crit("deterministic", {"kind": "command", "argv": []}), "check.deterministic"),
    ("cmd timeout", crit("deterministic", {"kind": "command", "argv": ["git"], "timeout_s": 0}), "check.deterministic"),
    ("python target", crit("deterministic", {"kind": "python", "callable": "os.system"}), "check.deterministic"),
    ("independent", crit("deterministic", {"kind": "file_exists", "path": "a", "independent": "y"}),
     "check.deterministic"),
    ("assert split", crit("assert", {"suite": "s", "min_score": 0.5}), "check.assert"),
    ("assert score 0", crit("assert", {"suite": "s", "split": "dev", "min_score": 0}), "check.assert"),
    ("assert score 2", crit("assert", {"suite": "s", "split": "dev", "min_score": 2}), "check.assert"),
    ("assert suite", crit("assert", {"suite": "", "split": "dev", "min_score": 0.5}), "check.assert"),
    ("long question", crit("s1", {"question": "x " * 151, "type": "noul"}), "check.s1"),
    ("no question", crit("s1", {"type": "noul"}), "check.s1"),
    ("bad type", crit("llm", {"question": "Q?", "type": "scale"}), "check.llm"),
    ("one option", crit("s1", {"question": "Q?", "type": "choice", "options": ["a"]}), "check.s1"),
    ("dup options", crit("s1", {"question": "Q?", "type": "choice", "options": ["a", "a"]}), "check.s1"),
    ("noul options", crit("s1", {"question": "Q?", "type": "noul", "options": ["yes"]}), "check.s1"),
    ("vague desc", crit("s1", {"question": "Q?", "type": "noul"}, description="Reply has a good tone"),
     "criterion.vague"),
    ("vague hyphen", crit("llm", {"question": "Is it high-quality?", "type": "noul"}), "criterion.vague"),
    ("vague robust", crit("s1", {"question": "Is the parser Robust?", "type": "noul"}), "criterion.vague"),
    ("unmeasurable", crit("s1", {"type": "noul"}, required=True), "criterion.unmeasurable"),
]


@pytest.mark.parametrize(("name", "c", "code"), BAD_CRITERIA, ids=[b[0] for b in BAD_CRITERIA])
def test_criterion_rules(rubric_factory, name: str, c: Criterion, code: str) -> None:
    problems = validate_rubric(with_extra(rubric_factory(), c))
    assert code in codes(problems), problems
    assert all(p.where.endswith("criterion:" + c.id) for p in problems if p.code.startswith(("check.", "criterion.")))


def test_vague_words_only_apply_to_soft_criteria(rubric_factory) -> None:
    c = crit("deterministic", {"kind": "regex", "pattern": "x"}, description="Output is clean and robust")
    assert validate_rubric(with_extra(rubric_factory(), c)) == []


# ---------------------------------------------------------------- graph


def sealed_graph(tmp_path, graph_factory, rubric_factory) -> tuple[TaskGraph, RubricVault]:
    vault = RubricVault.for_run(tmp_path)
    ids = [d.id for d in graph_factory().deliverables]
    return graph_factory({t: vault.seal(rubric_factory(t)) for t in ids}), vault


def edit(g: TaskGraph, tid: str, **kw: Any) -> TaskGraph:
    return dataclasses.replace(g, deliverables=tuple(
        dataclasses.replace(d, **kw) if d.id == tid else d for d in g.deliverables))


def test_valid_graph(tmp_path, graph_factory, rubric_factory) -> None:
    assert validate_graph(graph_factory()) == []
    g, vault = sealed_graph(tmp_path, graph_factory, rubric_factory)
    assert validate_graph(g, vault=vault) == []


def test_input_paths_are_not_output_mentions(graph_factory) -> None:
    g = edit(graph_factory(), "triage", instructions="Read out/summary.md and data/log.txt, then write out/triage.json.")
    assert validate_graph(g) == []


BAD_GRAPHS: list[tuple[str, str, dict[str, Any], str]] = [
    ("id format", "audit", {"id": "Audit"}, "id.format"),
    ("id reserved", "audit", {"id": "sealed"}, "id.reserved"),
    ("id dup", "audit", {"id": "reply"}, "id.duplicate"),
    ("dep unknown", "audit", {"depends_on": ("ghost",)}, "dep.unknown"),
    ("dep self", "audit", {"depends_on": ("audit",)}, "dep.self"),
    ("dep dup", "audit", {"depends_on": ("summary", "summary")}, "dep.duplicate"),
    ("cycle", "summary", {"depends_on": ("reply",)}, "graph.cycle"),
    ("title", "audit", {"title": " "}, "title.empty"),
    ("instr empty", "audit", {"instructions": "  "}, "instructions.empty"),
    ("instr long", "audit", {"instructions": "x" * 1201}, "instructions.length"),
    ("no path", "audit", {"output": OutputSpec("file")}, "output.path"),
    ("bad path", "audit", {"output": OutputSpec("patch", "../x.diff")}, "output.path"),
    ("sealed path", "audit", {"output": OutputSpec("file", "sealed/x.json")}, "output.path"),
    ("schema kind", "audit", {"output": OutputSpec("file", "out/a.md", {"type": "object"})}, "output.schema"),
    ("schema bad", "audit", {"output": OutputSpec("json", "out/a.json", {"type": 3})}, "output.schema"),
    ("two outputs", "audit", {"instructions": "Write out/audit.md and also notes/extra.md."}, "output.ambiguous"),
    ("ctx ref", "audit", {"context": (ContextRef("deliverable", "reply"),)}, "context.ref"),
    ("budget", "audit", {"budget": Budget(max_attempts=0)}, "budget"),
    ("budget weight", "audit", {"budget": Budget(weight={"llm": 0})}, "budget"),
    ("commitment", "audit", {"rubric_commitment": "abc"}, "commitment.format"),
]


@pytest.mark.parametrize(("name", "tid", "kw", "code"), BAD_GRAPHS, ids=[b[0] for b in BAD_GRAPHS])
def test_graph_rules(graph_factory, name: str, tid: str, kw: dict[str, Any], code: str) -> None:
    assert code in codes(validate_graph(edit(graph_factory(), tid, **kw)))


def test_empty_graph(graph_factory) -> None:
    assert "graph.empty" in codes(validate_graph(dataclasses.replace(graph_factory(), deliverables=())))


def test_commitment_must_match_sealed_rubric(tmp_path, graph_factory, rubric_factory) -> None:
    g, vault = sealed_graph(tmp_path, graph_factory, rubric_factory)
    assert "commitment.missing" in codes(validate_graph(edit(g, "audit", rubric_commitment="1" * 64), vault=vault))
    c = g.deliverable("audit").rubric_commitment
    (vault.root / f"{c}.json").write_text("{}", encoding="utf-8")
    assert codes(validate_graph(g, vault=vault)) == {"commitment.tampered"}
    swapped = edit(g, "summary", rubric_commitment=g.deliverable("reply").rubric_commitment)
    assert "rubric.deliverable" in codes(validate_graph(swapped, vault=vault))
    bad = vault.seal(rubric_factory("reply", canary="nothex"))
    assert "rubric.canary" in codes(validate_graph(edit(g, "reply", rubric_commitment=bad), vault=vault))


@pytest.mark.parametrize("leak", [
    "Also: does the report cite at least two distinct order identifiers taken from the log?",
    "Tag the file with 0123456789abcdef.",
    "Make sure c-format holds.",
    "This is graded by the order-support suite.",
])
def test_instructions_must_not_leak_rubric(tmp_path, graph_factory, rubric_factory, leak: str) -> None:
    g, vault = sealed_graph(tmp_path, graph_factory, rubric_factory)
    text = g.deliverable("summary").instructions + " " + leak
    assert codes(validate_graph(edit(g, "summary", instructions=text), vault=vault)) == {"rubric.leak"}


def test_short_overlap_is_not_a_leak(rubric_factory) -> None:
    assert rubric_leaks("Cite two distinct order identifiers from the log.", rubric_factory()) == []
