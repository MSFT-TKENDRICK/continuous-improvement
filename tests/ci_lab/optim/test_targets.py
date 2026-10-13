import os

import pytest

from ci_lab.optim.targets import TargetError, TextTarget, on_surface, resolve_targets

H = "harness"


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.md", "a/../../x.md", ".git/config", "C:/x.md", ""])
def test_parse_rejects_unsafe_paths(bad):
    with pytest.raises(TargetError):
        TextTarget.parse(bad)


def test_parse_key_and_backslashes():
    t = TextTarget.parse("src\\a\\agent.yaml#options.instructions")
    assert t.path == "src/a/agent.yaml" and t.key == "options.instructions"
    assert t.id == "src/a/agent.yaml#options.instructions"


def test_resolve_components_and_explicit(worktree):
    got = resolve_targets(worktree, ["prompt", "skill", "memory", "config", f"{H}/agent.yaml#instructions"])
    assert [(c, t.id) for c, t in got] == [
        ("prompt", f"{H}/prompts/system.md"),
        ("skill", f"{H}/skills/harness-editing/SKILL.md"),
        ("prompt", f"{H}/agent.yaml#instructions"),
    ]
    assert [c for c, _ in resolve_targets(worktree, [], default_focus=("skill",))] == ["skill"]


def test_resolve_respects_surface_and_existence(worktree):
    with pytest.raises(TargetError):
        resolve_targets(worktree, ["prompt"], surface_globs=["docs/**"])
    with pytest.raises(TargetError):
        resolve_targets(worktree, [f"{H}/prompts/missing.md"])
    assert on_surface(f"{H}/x.md", [f"{H}/**"])
    assert not on_surface(f"{H}/frozen/x.md", [f"{H}/**"], [f"{H}/frozen/**"])
    assert on_surface("SKILL.md", ["**/SKILL.md"])


def test_read_write_text_and_yaml_key(worktree):
    t = TextTarget.parse(f"{H}/agent.yaml#instructions")
    assert t.read(worktree) == "Base."
    t.write(worktree, "New\ninstructions.")
    assert t.read(worktree) == "New\ninstructions."
    assert "name: HarnessImprover" in (worktree / H / "agent.yaml").read_text()
    with pytest.raises(TargetError):
        TextTarget.parse(f"{H}/agent.yaml#name.sub").read(worktree)
    p = TextTarget.parse(f"{H}/prompts/system.md")
    p.write(worktree, "hello\n")
    assert (worktree / p.path).read_bytes() == b"hello\n"


@pytest.mark.skipif(os.name == "nt" and not os.environ.get("CI_LAB_TEST_SYMLINKS"),
                    reason="symlink creation needs privileges on Windows")
def test_symlink_rejected(worktree, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("x")
    (worktree / H / "prompts" / "link.md").symlink_to(outside)
    with pytest.raises(TargetError):
        TextTarget.parse(f"{H}/prompts/link.md").file(worktree)
