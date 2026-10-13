from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ci_lab.gitops.names import archive_tag, arm_branch, check_ref_format, ledger_branch, sleep_branch
from ci_lab.gitops.safe_path import UnsafePathError, is_link_like, match_globs, safe_join, safe_write_text


# ---------------------------------------------------------------- names


def test_archive_tag_and_branches():
    assert archive_tag("cmp-a-r001", "a") == "exp-archive/cmp-a-r001/a"
    assert ledger_branch("cmp-a") == "exp-ledger/cmp-a"
    assert sleep_branch("20250101", 2) == "exp/sleep-20250101-2/cand"
    assert arm_branch("cmp-a-r001", "b").endswith("/b")


@pytest.mark.parametrize("eid,arm", [("Bad", "a"), ("cmp-a-r001", "A"), ("cmp-a-r001", "a..b"),
                                     ("cmp-a-r001", "a/b"), ("x", "a")])
def test_archive_tag_rejects_bad_parts(eid, arm):
    with pytest.raises(ValueError):
        archive_tag(eid, arm)


@pytest.mark.parametrize("ref", ["a..b", "a b", "a~1", "a^", "a:b", "a.lock", "/a", "a//b", "@{x", "a\\b", ""])
def test_check_ref_format_rejects(ref):
    with pytest.raises(ValueError):
        check_ref_format(ref)


def test_check_ref_format_accepts():
    assert check_ref_format("exp/x-1/a") == "exp/x-1/a"


# ---------------------------------------------------------------- safe_join


ATTACKS = [
    "", "../x", "a/../../x", "a/..", "a\\..\\b", "/etc/passwd", "\\\\server\\share\\x", "//server/share",
    "C:foo", "C:\\x", "c:/x", ".git/config", "a/.GIT/x", ".git", "GIT~1/config", "a/.git.bak/x",
    "con", "CON.txt", "a/nul.json", "com1", "LPT9.log", "a:stream", "file.txt::$DATA", "a\x00b",
    "trailing.", "trailing ", "a/b./c", "x?y", "x*y", "x|y", "x<y", 'x"y', "x\x01y",
]


@pytest.mark.parametrize("rel", ATTACKS)
def test_safe_join_rejects(tmp_path, rel):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, rel)


@pytest.mark.parametrize("rel,expected", [("a/b.json", "a/b.json"), ("a\\b\\c.txt", "a/b/c.txt"),
                                          ("./a/./b", "a/b"), ("harness/prompt.md", "harness/prompt.md")])
def test_safe_join_accepts(tmp_path, rel, expected):
    p = safe_join(tmp_path, rel)
    assert p == Path(os.path.realpath(tmp_path)).joinpath(*expected.split("/"))


def test_safe_join_rejects_case_alias_of_dot_git(tmp_path):
    (tmp_path / ".git").mkdir()
    for rel in (".Git/hooks/x", ".GIT"):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, rel)


def _make_dir_symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks need privileges here: {exc}")


def test_safe_join_rejects_symlink_component(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _make_dir_symlink(root / "link", outside)
    assert is_link_like(root / "link")
    with pytest.raises(UnsafePathError):
        safe_join(root, "link/evil.txt")
    with pytest.raises(UnsafePathError):
        safe_write_text(root, "link/evil.txt", "x")
    assert not (outside / "evil.txt").exists()


def test_safe_join_rejects_file_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("s", encoding="utf-8")
    try:
        os.symlink(tmp_path / "secret", root / "f")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks need privileges here: {exc}")
    with pytest.raises(UnsafePathError):
        safe_join(root, "f")


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_safe_join_rejects_junction(tmp_path):
    import _winapi

    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        _winapi.CreateJunction(str(outside), str(root / "j"))
    except OSError as exc:
        pytest.skip(f"cannot create junction: {exc}")
    assert is_link_like(root / "j")
    with pytest.raises(UnsafePathError):
        safe_join(root, "j/evil.txt")
    with pytest.raises(UnsafePathError):
        safe_write_text(root, "j/sub/evil.txt", "x")
    assert not (outside / "sub").exists()


def test_safe_join_rejects_symlinked_root_child_but_allows_symlinked_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "rootlink"
    _make_dir_symlink(link, real)
    p = safe_join(link, "a/b.txt")
    assert str(p).startswith(os.path.realpath(real))


def test_safe_write_text_creates_parents(tmp_path):
    p = safe_write_text(tmp_path, "a/b/c.txt", "hi")
    assert p.read_text(encoding="utf-8") == "hi"


# ---------------------------------------------------------------- match_globs


@pytest.mark.parametrize("rel,globs,ok", [
    ("harness/prompt.md", ["harness/**"], True),
    ("harness/a/b/c.md", ["harness/**/*.md"], True),
    ("harness/c.md", ["harness/**/*.md"], True),
    ("harness/c.py", ["harness/**/*.md"], False),
    ("src/x.py", ["harness/**"], False),
    ("harness\\x.md", ["harness/*.md"], True),
    ("harness/a/x.md", ["harness/*.md"], False),
    ("README.md", ["*.md"], True),
    ("a/README.md", ["*.md"], False),
    ("a/README.md", ["**/README.md"], True),
    ("x.json", ["x.[jt]son"], True),
    ("x.json", [], False),
])
def test_match_globs(rel, globs, ok):
    assert match_globs(rel, globs) is ok
