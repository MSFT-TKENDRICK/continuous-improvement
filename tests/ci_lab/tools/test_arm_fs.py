from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ci_lab.tools.arm_fs import ArmFS, make_arm_fs
from ci_lab.tools.paths import (
    PathRejected,
    components_for,
    glob_match,
    normalize_rel,
    safe_join,
)


@pytest.mark.parametrize("bad", [
    "", "   ", "../x.md", "a/../../x.md", "a/./b.md", "a//b.md", "/etc/passwd", "\\\\server\\share\\x.md",
    "C:\\Windows\\x.md", "C:x.md", ".git/config", "a/.GIT/hooks/pre-commit", "a/b.md:stream", "a/b.md.",
    "a/b.md ", "PROGRA~1/x.md", "a/con.md", "a/NUL", "a\x00b.md",
])
def test_normalize_rejects_attacks(bad: str) -> None:
    with pytest.raises(PathRejected):
        normalize_rel(bad)


def test_normalize_accepts_and_posixifies() -> None:
    assert normalize_rel("src\\order_support\\harness\\prompts\\system.md") == \
        "src/order_support/harness/prompts/system.md"


def test_glob_semantics() -> None:
    assert glob_match("a/b/c.md", "a/**")
    assert glob_match("a/c.md", "a/**/*.md") and glob_match("a/b/c/d.md", "a/**/*.md")
    assert not glob_match("a/b/c.md", "a/*.md")
    assert glob_match("x/y/memory.md", "**/memory.md")
    assert not glob_match("ab/c.md", "a/**")


def test_components_memory_wins_over_skill(layout) -> None:
    assert components_for(f"{layout.harness}/skills/refunds/memory.md", layout.component_globs) == {"memory"}
    assert components_for(f"{layout.harness}/skills/refunds/SKILL.md", layout.component_globs) == {"skill"}
    assert components_for(f"{layout.harness}/agent.yaml", layout.component_globs) == {"config", "context_mgmt"}


def _tools(repo, layout, **kw):
    return make_arm_fs(repo[0], layout.surface, layout.frozen, **kw)


def test_list_read_write_happy_path(repo, layout) -> None:
    t = _tools(repo, layout)
    listing = t["list_files"]().splitlines()
    assert f"{layout.harness}/prompts/system.md" in listing
    assert f"{layout.harness}/helper.py" not in listing  # frozen
    assert not any(p.startswith("evals/") for p in listing)
    assert t["list_files"]("**/*.yaml").splitlines() == [f"{layout.harness}/agent.yaml",
                                                         f"{layout.harness}/tool_specs.yaml"]
    assert "verify identity" in t["read_file"](f"{layout.harness}/prompts/system.md")
    out = t["write_file"](f"{layout.harness}/prompts/new/extra.md", "hello")
    assert out.startswith("wrote 5 bytes")
    assert (repo[0] / layout.harness / "prompts/new/extra.md").read_text() == "hello"
    assert t["write_file"].arm_fs.written == [f"{layout.harness}/prompts/new/extra.md"]


@pytest.mark.parametrize("path", [
    "../outside.md", "src/order_support/harness/../../../outside.md", "/tmp/x.md", "C:\\x.md",
    ".git/config", "src/order_support/harness/.git/x.md", "src/order_support/harness/helper.py",
    "src/order_support/harness/run.ps1", "evals/assert/x/eval_config.yaml", "README.md",
    "src/order_support/agent.py", "src/order_support/harness/prompts/x.md:ads",
    "src/order_support/HARNESS/prompts/system.md", "src/order_support/harness/prompts/system.exe",
])
def test_write_attacks_rejected(repo, layout, path: str) -> None:
    t = _tools(repo, layout)
    before = {p: p.read_bytes() for p in repo[0].rglob("*") if p.is_file() and ".git" not in p.parts}
    out = t["write_file"](path, "pwned")
    assert out.startswith("ERROR"), out
    after = {p: p.read_bytes() for p in repo[0].rglob("*") if p.is_file() and ".git" not in p.parts}
    assert before == after
    assert not (repo[0].parent / "outside.md").exists()


def test_read_outside_surface_rejected(repo, layout) -> None:
    t = _tools(repo, layout)
    assert t["read_file"]("evals/assert/x/eval_config.yaml").startswith("ERROR")
    assert t["read_file"](f"{layout.harness}/helper.py").startswith("ERROR")
    assert t["read_file"]("../../etc/passwd").startswith("ERROR")


def test_case_alias_of_existing_dir_rejected(repo, layout) -> None:
    fs = ArmFS(repo[0], ("src/order_support/harness/**", "src/order_support/Harness/**"))
    with pytest.raises(PathRejected, match="case alias"):
        fs.write("src/order_support/Harness/prompts/system.md", "x")


def test_size_caps(repo, layout) -> None:
    t = _tools(repo, layout, max_bytes=16)
    assert "limit is 16" in t["write_file"](f"{layout.harness}/prompts/system.md", "x" * 17)
    assert "[truncated at 16 bytes" in t["read_file"](f"{layout.harness}/prompts/system.md")


def test_writable_globs_narrow_writes_not_reads(repo, layout) -> None:
    t = _tools(repo, layout, writable_globs=layout.component_globs["prompt"])
    assert t["write_file"](f"{layout.harness}/agent.yaml", "x: 1\n").startswith("ERROR")
    assert "OrderSupport" in t["read_file"](f"{layout.harness}/agent.yaml")
    assert t["write_file"](f"{layout.harness}/prompts/system.md", "ok\n").startswith("wrote")


def _link(target: Path, link: Path, *, directory: bool) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32" and directory:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False)
        if r.returncode == 0:
            return
    pytest.skip("cannot create symlinks/junctions here")


def test_symlinked_dir_escape_rejected(repo, layout, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _link(outside, repo[0] / layout.harness / "prompts" / "linked", directory=True)
    t = _tools(repo, layout)
    out = t["write_file"](f"{layout.harness}/prompts/linked/evil.md", "x")
    assert out.startswith("ERROR")
    assert not (outside / "evil.md").exists()
    assert f"{layout.harness}/prompts/linked/evil.md" not in t["list_files"]()
    with pytest.raises(PathRejected):
        safe_join(repo[0], f"{layout.harness}/prompts/linked/evil.md")


def test_symlinked_file_rejected(repo, layout, tmp_path: Path) -> None:
    secret = tmp_path / "secret.md"
    secret.write_text("secret")
    _link(secret, repo[0] / layout.harness / "prompts" / "secret.md", directory=False)
    t = _tools(repo, layout)
    assert t["read_file"](f"{layout.harness}/prompts/secret.md").startswith("ERROR")
    assert t["write_file"](f"{layout.harness}/prompts/secret.md", "x").startswith("ERROR")
    assert secret.read_text() == "secret"
