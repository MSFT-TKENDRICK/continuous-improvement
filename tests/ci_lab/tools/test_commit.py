from __future__ import annotations

from pathlib import Path

from ci_lab.tools.commit import (
    CO_AUTHOR,
    changed_paths,
    edits_since,
    make_commit_tool,
    parse_trailers,
)


def _tool(repo, layout, max_edits=2, **kw):
    wt, _ = repo
    return make_commit_tool(wt, max_edits, component_globs=layout.component_globs, experiment_id="exp-1",
                            variant="arm-a", surface_globs=layout.surface, frozen_globs=layout.frozen, **kw)


def _write(wt: Path, rel: str, text: str) -> None:
    p = wt / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def test_commit_has_trailers(repo, layout, gitrun):
    wt, base = repo
    commit = _tool(repo, layout)
    _write(wt, f"{layout.harness}/prompts/system.md", "Be concise.\n")
    out = commit("prompt", "Shorter replies should reduce   rambling\nanswers.")
    assert out.startswith("committed"), out
    msg = gitrun(wt, "log", "-1", "--format=%B")
    trailers = parse_trailers(msg)
    assert trailers["RRSI-Component"] == ["prompt"]
    assert trailers["RRSI-Hypothesis"] == ["Shorter replies should reduce rambling answers."]
    assert trailers["OES-Experiment"] == ["exp-1"]
    assert trailers["OES-Variant"] == ["arm-a"]
    assert trailers["Co-authored-by"] == [CO_AUTHOR]
    # git itself recognizes them as trailers
    parsed = gitrun(wt, "log", "-1", "--format=%(trailers:key=RRSI-Component,valueonly)").strip()
    assert parsed == "prompt"
    [edit] = commit.edits
    assert edit.component == "prompt" and edit.files == (f"{layout.harness}/prompts/system.md",)
    assert edits_since(wt, base) == [edit]
    assert changed_paths(wt) == []


def test_rejects_unknown_component_and_empty(repo, layout):
    commit = _tool(repo, layout)
    assert commit("tools", "x").startswith("ERROR: unknown component")
    assert commit("prompt", "  ").startswith("ERROR: hypothesis")
    assert commit("prompt", "x" * 500).startswith("ERROR: hypothesis is")
    assert commit("prompt", "nothing written").startswith("ERROR: nothing to commit")


def test_rejects_component_path_mismatch(repo, layout, gitrun):
    wt, base = repo
    commit = _tool(repo, layout)
    _write(wt, f"{layout.harness}/skills/refunds/SKILL.md", "# Refunds\nnew\n")
    out = commit("prompt", "mislabelled")
    assert out.startswith("ERROR: cannot commit") and "not 'prompt'" in out
    assert gitrun(wt, "rev-parse", "HEAD").strip() == base
    assert commit("skill", "skill edit").startswith("committed")


def test_rejects_mixed_components_and_frozen(repo, layout):
    wt, _ = repo
    commit = _tool(repo, layout)
    _write(wt, f"{layout.harness}/prompts/system.md", "x\n")
    _write(wt, f"{layout.harness}/tool_specs.yaml", "lookup_order: {}\n")
    assert "not 'prompt'" in commit("prompt", "two components")
    (wt / f"{layout.harness}/tool_specs.yaml").write_text("lookup_order:\n  description: Look up an order by id.\n",
                                                           encoding="utf-8", newline="\n")
    _write(wt, f"{layout.harness}/helper.py", "print(1)\n")
    assert "frozen" in commit("prompt", "code")
    (wt / f"{layout.harness}/helper.py").write_text("print('code is frozen')\n", encoding="utf-8", newline="\n")
    _write(wt, "evals/assert/x/eval_config.yaml", "suite: y\n")
    assert "outside the editable surface" in commit("prompt", "eval")


def test_budget_and_allowed_components(repo, layout, gitrun):
    wt, base = repo
    commit = _tool(repo, layout, max_edits=1, allowed_components=("prompt",))
    _write(wt, f"{layout.harness}/agent.yaml", "name: OrderSupport\n")
    assert "may only edit prompt" in commit("config", "nope")
    (wt / f"{layout.harness}/agent.yaml").unlink()
    gitrun(wt, "checkout", "--", f"{layout.harness}/agent.yaml")
    _write(wt, f"{layout.harness}/prompts/system.md", "one\n")
    assert commit("prompt", "first").startswith("committed")
    _write(wt, f"{layout.harness}/prompts/system.md", "two\n")
    assert "budget exhausted" in commit("prompt", "second")
    assert len(edits_since(wt, base)) == 1


def test_overlapping_component_globs(repo, layout):
    wt, base = repo
    commit = _tool(repo, layout)
    _write(wt, f"{layout.harness}/skills/refunds/memory.md", "- learned\n")
    assert commit("memory", "remember").startswith("committed")
    _write(wt, f"{layout.harness}/agent.yaml", "name: OrderSupport\nmax_turns: 4\n")
    assert commit("context_mgmt", "fewer turns").startswith("committed")
    assert [e.component for e in edits_since(wt, base)] == ["memory", "context_mgmt"]
