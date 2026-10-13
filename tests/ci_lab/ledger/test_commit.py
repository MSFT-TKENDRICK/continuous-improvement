from __future__ import annotations

import pytest

from ci_lab.ledger.atomic import atomic_write_json
from ci_lab.ledger.commit import LedgerConflict, LedgerPathError, ledger_commit
from ci_lab.ledger.layout import Layout


def _write_round(repo, cid="demo-1", eid="demo-1-r01", n=1):
    lay = Layout(repo)
    atomic_write_json(lay.experiment_json(cid, eid), {"id": eid, "n": n})
    atomic_write_json(lay.frontier_json(cid), {"incumbent": "x", "n": n})
    return lay


def test_commit_on_branch_without_touching_user_index(git_repo, git):
    head = git(git_repo, "rev-parse", "HEAD")
    (git_repo / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    git(git_repo, "add", "src/app.py")  # user has staged work
    staged_before = git(git_repo, "diff", "--cached", "--name-only")
    lay = _write_round(git_repo)

    res = ledger_commit(git_repo, [lay.campaign_dir("demo-1")], "Record demo-1 r01")

    assert res.created and res.parent == head
    assert git(git_repo, "rev-parse", "refs/heads/main") == res.commit
    files = git(git_repo, "show", "--name-only", "--format=", res.commit).splitlines()
    assert sorted(files) == ["experiments/campaigns/demo-1/frontier.json",
                             "experiments/campaigns/demo-1/rounds/demo-1-r01/experiment.json"]
    # the staged src change was NOT committed and the user's index still holds it
    assert git(git_repo, "show", f"{res.commit}:src/app.py") == "x = 1"
    assert git(git_repo, "show", ":src/app.py") == "x = 2"
    assert git(git_repo, "ls-files", "experiments") == ""
    assert staged_before == "src/app.py"
    assert git(git_repo, "log", "-1", "--format=%s", res.commit) == "Record demo-1 r01"


def test_commit_to_other_branch_and_noop(git_repo, git):
    lay = _write_round(git_repo)
    res = ledger_commit(git_repo, ["experiments/campaigns/demo-1"], "ledger", ref="exp-ledger/demo-1")
    assert res.ref == "refs/heads/exp-ledger/demo-1"
    assert git(git_repo, "rev-parse", "HEAD") != res.commit  # checkout untouched
    again = ledger_commit(git_repo, [lay.campaign_dir("demo-1")], "ledger", ref="exp-ledger/demo-1")
    assert not again.created and again.commit == res.commit


def test_commit_new_ref_is_ledger_only_root(git_repo, git):
    _write_round(git_repo)
    res = ledger_commit(git_repo, ["experiments"], "first", ref="refs/heads/ledger-only")
    assert res.created and res.parent is None
    assert all(p.startswith("experiments/")
               for p in git(git_repo, "ls-tree", "-r", "--name-only", res.commit).splitlines())


@pytest.mark.parametrize("bad", ["src/app.py", "README.md", ".", "", "experiments/../src/app.py",
                                 "experimentsX/a.json"])
def test_rejects_non_ledger_paths(git_repo, bad):
    _write_round(git_repo)
    with pytest.raises(LedgerPathError):
        ledger_commit(git_repo, [bad], "nope")


def test_rejects_absolute_path_outside_repo(git_repo, tmp_path):
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(LedgerPathError):
        ledger_commit(git_repo, [outside], "nope")


def test_cas_conflict_detected(git_repo, git):
    lay = _write_round(git_repo)
    stale = git(git_repo, "rev-parse", "main")
    first = ledger_commit(git_repo, [lay.campaign_dir("demo-1")], "one", expected_old=stale)
    assert first.created
    _write_round(git_repo, n=2)
    with pytest.raises(LedgerConflict) as ei:
        ledger_commit(git_repo, [lay.campaign_dir("demo-1")], "two", expected_old=stale)
    assert ei.value.actual == first.commit
    assert git(git_repo, "rev-parse", "main") == first.commit


def test_deletion_is_committed(git_repo, git):
    lay = _write_round(git_repo)
    ledger_commit(git_repo, [lay.campaign_dir("demo-1")], "add")
    lay.frontier_json("demo-1").unlink()
    res = ledger_commit(git_repo, [lay.frontier_json("demo-1")], "drop frontier")
    assert res.created
    assert "frontier.json" not in git(git_repo, "ls-tree", "-r", "--name-only", res.commit)


def test_temp_index_cleaned_up(git_repo):
    lay = _write_round(git_repo)
    ledger_commit(git_repo, [lay.campaign_dir("demo-1")], "add")
    leftovers = list((git_repo / ".git" / "ci-lab").glob("index-*"))
    assert leftovers == []
