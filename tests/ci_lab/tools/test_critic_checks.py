from __future__ import annotations

from pathlib import Path

import pytest

from ci_lab.tools.commit import make_commit_tool
from ci_lab.tools.critic_checks import (
    ArmDiff,
    CriticConfig,
    FileChange,
    LeakCorpus,
    check_components,
    check_denylist,
    check_leaks,
    check_paths,
    check_sizes,
    check_specs,
    check_tool_bindings,
    collect_diff,
    diff_text,
    run_checks,
    tool_names,
)

H = "src/order_support/harness"


@pytest.fixture
def cfg(layout) -> CriticConfig:
    return CriticConfig(surface_globs=layout.surface, component_globs=layout.component_globs,
                        frozen_globs=layout.frozen)


def _diff(*files: FileChange) -> ArmDiff:
    return ArmDiff("b", "h", list(files))


def _mod(path: str, before: str | None, after: str | None) -> FileChange:
    return FileChange(path, "A" if before is None else ("D" if after is None else "M"), before, after)


def _commit(repo, layout, rel: str, text: str, component: str, hyp: str = "fix it") -> str:
    wt, _ = repo
    (wt / rel).parent.mkdir(parents=True, exist_ok=True)
    (wt / rel).write_text(text, encoding="utf-8", newline="\n")
    tool = make_commit_tool(wt, 5, component_globs=layout.component_globs, experiment_id="e", variant="v")
    return tool(component, hyp)


def test_clean_diff_passes(repo, layout, cfg):
    wt, base = repo
    assert _commit(repo, layout, f"{H}/prompts/system.md",
                   "You are a helpful order support agent.\nConfirm the order before acting.\n",
                   "prompt").startswith("committed")
    diff = collect_diff(wt, base)
    assert [f.path for f in diff.files] == [f"{H}/prompts/system.md"]
    assert diff.commits[0].component == "prompt" and diff.commits[0].hypothesis == "fix it"
    assert run_checks(diff, cfg) == []
    assert "+Confirm the order" in diff_text(wt, base)


def test_path_guard(repo, layout, cfg, gitrun):
    wt, base = repo
    (wt / "evals/assert/x/eval_config.yaml").write_text("suite: hacked\n", encoding="utf-8")
    gitrun(wt, "commit", "-qam", "eval edit")
    (wt / f"{H}/prompts/new.md").write_text("dirty\n", encoding="utf-8")
    reasons = check_paths(collect_diff(wt, base), cfg)
    assert any("outside the editable surface" in r for r in reasons)
    assert any("uncommitted changes" in r for r in reasons)
    frozen = check_paths(_diff(_mod(f"{H}/helper.py", "a", "b")), cfg)
    assert frozen == [f"path: {H}/helper.py is frozen"]
    assert "path:" in check_paths(_diff(_mod("../escape.md", None, "x")), cfg)[0]


def test_component_tag_vs_path(repo, layout, cfg, gitrun):
    wt, base = repo
    (wt / f"{H}/tool_specs.yaml").write_text("lookup_order:\n  description: Find an order.\n", encoding="utf-8")
    gitrun(wt, "commit", "-qa", "-m", "x\n\nRRSI-Component: prompt\nRRSI-Hypothesis: h\n")
    (wt / f"{H}/prompts/system.md").write_text("untagged\n", encoding="utf-8")
    gitrun(wt, "commit", "-qam", "no trailers")
    reasons = check_components(collect_diff(wt, base), cfg)
    assert any("tagged 'prompt' but changes" in r and "client_tool" in r for r in reasons)
    assert any("no valid RRSI-Component" in r for r in reasons)


def test_spec_checks(cfg):
    ok = _mod(f"{H}/agent.yaml", "name: A\nmodel:\n  id: gpt-5\n", "name: A\nmodel:\n  id: gpt-5\nmax_turns: 3\n")
    assert check_specs(_diff(ok), cfg) == []
    bad = [
        _mod(f"{H}/agent.yaml", "name: A\n", "name: [unclosed\n"),
        _mod(f"{H}/agent.yaml", "name: A\n", "name: A\ninstructions: =Env.SECRET\n"),
        _mod(f"{H}/agent.yaml", "name: A\n", "name: B\n"),
        _mod(f"{H}/agent.yaml", "name: A\nmodel:\n  id: gpt-5\n", "name: A\nmodel:\n  id: gpt-6\n"),
    ]
    reasons = check_specs(_diff(*bad), cfg)
    for needle in ("does not parse", "expression value at instructions", "changes 'name'", "model.id"):
        assert any(needle in r for r in reasons), (needle, reasons)
    cfg.spec_validator = lambda path, text: ["custom problem"] if "max_turns" in text else []
    assert check_specs(_diff(ok), cfg) == [f"spec: {H}/agent.yaml: custom problem"]


def test_leak_screen(cfg):
    case = "my package from order ORD-10042 arrived crushed and the blender inside no longer turns on at all"
    cfg.leak_corpus = LeakCorpus.build([case], ["ORD-10042", "Alice Johnson", "bob"])
    leak = _mod(f"{H}/prompts/system.md", "base\n",
                "base\nIf the package arrived crushed and the blender inside no longer turns on, refund.\n")
    reasons = check_leaks(_diff(leak), cfg)
    assert reasons and "8-gram" in reasons[0]
    lit = _mod(f"{H}/prompts/system.md", "base\n", "base\nFor ord-10042 or alice johnson, escalate.\n")
    reasons = check_leaks(_diff(lit), cfg)
    assert any("ORD-10042" in r for r in reasons) and any("Alice Johnson" in r for r in reasons)
    # literals shorter than 4 chars are ignored; pre-existing material is not a leak
    assert check_leaks(_diff(_mod(f"{H}/prompts/system.md", "x\n", "x\nbob\n")), cfg) == []
    cfg.baseline_text = "For ORD-10042 we already had a note."
    assert check_leaks(_diff(_mod(f"{H}/prompts/system.md", "x\n", "x\nORD-10042\n")), cfg) == []
    generic = _mod(f"{H}/prompts/system.md", "base\n", "base\nVerify damage claims with a photo first.\n")
    assert check_leaks(_diff(generic), cfg) == []


@pytest.mark.parametrize("text", [
    "The judge rewards short answers.", "Maximize your score.", "Graders check tone.", "the evaluator", "Rubric",
    "ASSERT checks this", "assert-ai", "test cases mention refunds", "never flag policy_violation", "benchmark",
])
def test_denylist_hits(cfg, text):
    assert check_denylist(_diff(_mod(f"{H}/prompts/system.md", "base\n", f"base\n{text}\n")), cfg)


def test_denylist_ignores_benign_and_preexisting(cfg):
    benign = "Assert nothing you cannot verify. Grade A packaging is fine. The order is scorched? Escalate."
    assert check_denylist(_diff(_mod(f"{H}/prompts/system.md", "base\n", benign)), cfg) == []
    assert check_denylist(_diff(_mod(f"{H}/prompts/system.md", "the judge\n", "the judge\nmore\n")), cfg) == []


def test_no_new_tool_bindings(cfg):
    before = "kind: Prompt\nname: A\ntools:\n  - kind: function\n    name: lookup_order\n    bindings:\n" \
             "      - name: lookup_order\n"
    added = before + "  - kind: function\n    name: refund_all\n"
    rebind = before.replace("      - name: lookup_order\n", "      - name: shell\n")
    assert check_tool_bindings(_diff(_mod(f"{H}/agent.yaml", before, before + "description: x\n")), cfg) == []
    assert "refund_all" in check_tool_bindings(_diff(_mod(f"{H}/agent.yaml", before, added)), cfg)[0]
    assert "binding:shell" in check_tool_bindings(_diff(_mod(f"{H}/agent.yaml", before, rebind)), cfg)[0]
    specs = _mod(f"{H}/tool_specs.yaml", "lookup_order: {}\n", "lookup_order: {}\ncancel_order: {}\n")
    assert "cancel_order" in check_tool_bindings(_diff(specs), cfg)[0]
    assert tool_names({"tools": {"a": {}, "b": {}}}) == {"a", "b"}


def test_size_limits(cfg):
    cfg.max_files, cfg.max_file_bytes, cfg.max_added_bytes = 1, 100, 50
    big = _mod(f"{H}/prompts/a.md", None, "x" * 200)
    reasons = check_sizes(_diff(big, _mod(f"{H}/prompts/b.md", None, "y")), cfg)
    assert any("2 files" in r for r in reasons)
    assert any("200 bytes; limit 100" in r for r in reasons)
    assert any("bytes added" in r for r in reasons)
    assert check_sizes(_diff(), cfg) == ["size: empty diff (no committed edits)"]


def test_deleted_file_is_not_parsed(cfg):
    gone = _mod(f"{H}/agent.yaml", "name: A\n", None)
    assert check_specs(_diff(gone), cfg) == [] and check_tool_bindings(_diff(gone), cfg) == []


def test_collect_diff_reads_head(repo, layout, cfg):
    wt, base = repo
    _commit(repo, layout, f"{H}/skills/refunds/SKILL.md", "# Refunds\nThe judge likes refunds.\n", "skill")
    diff = collect_diff(wt, base, "HEAD")
    assert Path(diff.files[0].path).name == "SKILL.md"
    assert any(r.startswith("denylist:") for r in run_checks(diff, cfg))
