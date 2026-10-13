from __future__ import annotations

from pathlib import Path

import pytest

from ci_lab.gitops import git as g
from ci_lab.gitops.publish_refs import PushRejected, TagConflict, is_ancestor, push_with_lease, remote_sha, tag_archive
from ci_lab.gitops.slots import SlotPool, SlotPoolExhausted, wt_root


def _commit(git, repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    git(repo, "add", "--", name)
    git(repo, "commit", "-q", "-m", f"edit {name}")
    return git(repo, "rev-parse", "HEAD")


def test_tree_hash(git_repo, git):
    head = git(git_repo, "rev-parse", "HEAD")
    assert g.tree_hash(git_repo, head) == git(git_repo, "rev-parse", "HEAD^{tree}")
    assert g.tree_hash(git_repo, head, "harness") == git(git_repo, "rev-parse", "HEAD:harness")
    assert g.tree_hash(git_repo, head, "missing") is None
    with pytest.raises(ValueError):
        g.tree_hash(git_repo, "--output=x")


def test_wt_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_WT_ROOT", str(tmp_path))
    assert wt_root() == tmp_path
    monkeypatch.delenv("CI_WT_ROOT")
    assert wt_root().is_absolute()


# ---------------------------------------------------------------- slots


def test_slot_pool_acquire_reset_release_remove(git_repo, git, tmp_path):
    base = git(git_repo, "rev-parse", "HEAD")
    second = _commit(git, git_repo, "harness/prompt.md", "be nicer\n")
    pool = SlotPool(git_repo, "cmp-test", root=tmp_path / "wt")

    a = pool.acquire(base, "exp/cmp-test-r001/a")
    assert a.path == tmp_path / "wt" / "cmp-test" / "s0"
    assert git(a.path, "rev-parse", "HEAD") == base
    assert git(a.path, "branch", "--show-current") == "exp/cmp-test-r001/a"
    b = pool.acquire(second, "exp/cmp-test-r001/b")
    assert b.index == 1 and git(b.path, "rev-parse", "HEAD") == second
    assert pool.leased() == [0, 1]

    (a.path / ".venv").mkdir()
    (a.path / ".venv" / "keep.txt").write_text("k", encoding="utf-8")
    (a.path / "junk.txt").write_text("j", encoding="utf-8")
    (a.path / "harness" / "prompt.md").write_text("dirty\n", encoding="utf-8")
    a = pool.reset(a, second, "exp/cmp-test-r002/a")
    assert git(a.path, "rev-parse", "HEAD") == second
    assert git(a.path, "branch", "--show-current") == "exp/cmp-test-r002/a"
    assert not (a.path / "junk.txt").exists()
    assert (a.path / ".venv" / "keep.txt").exists()
    assert (a.path / "harness" / "prompt.md").read_text(encoding="utf-8") == "be nicer\n"
    assert git(a.path, "status", "--porcelain") == ""

    pool.release(a)
    assert pool.leased() == [1]
    assert git(a.path, "branch", "--show-current") == ""
    # branch is free again after release: can be checked out in the main repo
    git(git_repo, "branch", "-f", "exp/cmp-test-r002/a", base)

    again = pool.acquire(base, "exp/cmp-test-r003/a")
    assert again.index == 0  # warm slot reused, not s2
    assert (again.path / ".venv" / "keep.txt").exists()

    pool.remove(b)
    assert not b.path.exists()
    assert str(b.path).replace("\\", "/") not in git(git_repo, "worktree", "list", "--porcelain").replace("\\", "/")
    assert [s.index for s in pool.slots()] == [0]
    pool.remove(again)
    assert pool.slots() == []


def test_slot_pool_max_slots_and_validation(git_repo, git, tmp_path):
    base = git(git_repo, "rev-parse", "HEAD")
    pool = SlotPool(git_repo, "cmp-max", root=tmp_path / "wt", max_slots=1)
    s = pool.acquire(base, "exp/cmp-max-r001/a")
    with pytest.raises(SlotPoolExhausted):
        pool.acquire(base, "exp/cmp-max-r001/b")
    with pytest.raises(ValueError):
        pool.reset(s, base, "bad..branch")
    with pytest.raises(ValueError):
        pool.reset(s, "no-such-commit-xyz")
    with pytest.raises(ValueError):
        SlotPool(git_repo, "BAD", root=tmp_path / "wt")
    pool.remove(s)


def test_slot_pool_reclaims_dead_lease(git_repo, git, tmp_path):
    base = git(git_repo, "rev-parse", "HEAD")
    pool = SlotPool(git_repo, "cmp-dead", root=tmp_path / "wt", max_slots=1)
    s = pool.acquire(base, "exp/cmp-dead-r001/a")
    data = pool._load()
    data["slots"]["0"]["lease"]["pid"] = 2**31 - 7  # not a live pid
    pool._save(data)
    s2 = pool.acquire(base, "exp/cmp-dead-r001/b")
    assert s2.index == s.index
    pool.remove(s2)


def test_evaluator_worktree(git_repo, git, tmp_path):
    base = git(git_repo, "rev-parse", "HEAD")
    second = _commit(git, git_repo, "README.md", "v2\n")
    pool = SlotPool(git_repo, "cmp-eval", root=tmp_path / "wt")
    p = pool.evaluator(base)
    assert git(p, "rev-parse", "HEAD") == base
    p = pool.evaluator(second)
    assert git(p, "rev-parse", "HEAD") == second
    assert git(p, "branch", "--show-current") == ""


# ---------------------------------------------------------------- publish_refs


@pytest.fixture
def remote(git_repo, git, tmp_path):
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(bare))
    git(git_repo, "remote", "add", "origin", str(bare))
    return bare


def test_push_with_lease(git_repo, git, remote):
    base = git(git_repo, "rev-parse", "HEAD")
    assert remote_sha(git_repo, "origin", "exp/x") is None
    git(git_repo, "branch", "exp/x", base)
    assert push_with_lease(git_repo, "origin", "exp/x", None) == base
    assert remote_sha(git_repo, "origin", "exp/x") == base
    # idempotent re-push
    assert push_with_lease(git_repo, "origin", "exp/x", None) == base

    new = _commit(git, git_repo, "README.md", "v2\n")
    git(git_repo, "branch", "-f", "exp/x", new)
    with pytest.raises(PushRejected):
        push_with_lease(git_repo, "origin", "exp/x", "0" * 40)  # stale lease
    assert remote_sha(git_repo, "origin", "exp/x") == base
    assert push_with_lease(git_repo, "origin", "exp/x", base) == new
    assert remote_sha(git_repo, "origin", "exp/x") == new

    with pytest.raises(PushRejected):  # must-not-exist lease on an existing branch
        push_with_lease(git_repo, "origin", "exp/x", None, local=base)
    with pytest.raises(ValueError):
        push_with_lease(git_repo, "--upload-pack=evil", "exp/x", None)
    with pytest.raises(ValueError):
        push_with_lease(git_repo, "origin", "bad..name", None)


def test_tag_archive_and_is_ancestor(git_repo, git, remote):
    base = git(git_repo, "rev-parse", "HEAD")
    new = _commit(git, git_repo, "README.md", "v2\n")
    tag = tag_archive(git_repo, "cmp-a-r001", "a", base)
    assert tag == "exp-archive/cmp-a-r001/a"
    assert git(git_repo, "rev-parse", f"refs/tags/{tag}") == base
    assert tag_archive(git_repo, "cmp-a-r001", "a", base, remote="origin") == tag
    assert git(remote, f"--git-dir={remote}", "rev-parse", f"refs/tags/{tag}") == base
    with pytest.raises(TagConflict):
        tag_archive(git_repo, "cmp-a-r001", "a", new)

    assert is_ancestor(git_repo, base, new)
    assert not is_ancestor(git_repo, new, base)
    assert is_ancestor(git_repo, base, base)
    with pytest.raises(g.GitError):
        is_ancestor(git_repo, base, "0" * 40)
