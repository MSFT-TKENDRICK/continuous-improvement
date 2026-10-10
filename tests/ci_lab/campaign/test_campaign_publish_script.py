"""scripts/campaign_publish.py: the stdlib-only privileged publisher of campaign-scheduled.yml.

Real git against a temp repo; the GitHub side runs in dry-run (the publisher's own mutations are
covered by tests/ci_lab/publish). These tests pin the validation the privileged job adds on top.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "campaign_publish.py"
CID = "camp1"
HARNESS = "harness/prompts/analyst.md"
LEDGER = f"experiments/campaigns/{CID}"
TRUSTED_LAYER = {"eid": f"{CID}-r01", "arm": "v1", "branch": f"exp/{CID}-r01/v1", "head": "c" * 40, "pr": 11}


def _load():
    spec = importlib.util.spec_from_file_location("campaign_publish_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


cp = _load()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@e", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def write(repo: Path, rel: str, text: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    write(r, HARNESS, "v0\n")
    write(r, "README.md", "readme\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    # trusted ledger branch: stack.json with one published layer
    git(r, "checkout", "-q", "-b", f"exp-ledger/{CID}")
    write(r, f"{LEDGER}/campaign.json", json.dumps({"campaignId": CID}))
    write(r, f"{LEDGER}/stack.json", json.dumps({"layers": [TRUSTED_LAYER], "stack_number": 5}))
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "ledger")
    git(r, "checkout", "-q", "main")
    return r


def arm(repo: Path, name: str, rel: str) -> str:
    git(repo, "checkout", "-q", "-b", name, "main")
    write(repo, rel, f"{name}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", name)
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    return sha


def artifact(tmp_path: Path, *, cid: str = CID, forged_stack: bool = True) -> Path:
    led = tmp_path / "art" / "ledger"
    led.mkdir(parents=True, exist_ok=True)
    (led / "campaign.json").write_text(json.dumps({"campaignId": cid}), encoding="utf-8")
    (led / "rounds").mkdir(exist_ok=True)
    (led / "rounds" / f"{CID}-r02.json").write_text("{}", encoding="utf-8")
    if forged_stack:  # the model job must not be able to redirect PR bases / stack edits
        (led / "stack.json").write_text(json.dumps({"layers": [], "stack_number": 999}), encoding="utf-8")
    return led


def requests(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "art" / "deferred.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def req(eid: str, winner: str | None, heads: dict[str, str]) -> dict:
    return {"eid": eid, "winner": winner, "heads": heads, "title": f"Round {eid}", "body": "body"}


def run(repo: Path, reqs: Path, led: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, str, str]:
    code = cp.main(["--cid", CID, "--repo", "o/r", "--requests", str(reqs), "--ledger", str(led),
                    "--checkout", str(repo), "--base-ref", "main", "--trusted-ref", f"exp-ledger/{CID}",
                    "--dry-run", *extra])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_publishes_on_trusted_stack_and_ignores_forged_one(repo: Path, tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    v1 = arm(repo, "a1", HARNESS)
    reqs = requests(tmp_path, req(f"{CID}-r02", "v1", {"v1": v1}), req(f"{CID}-r03", None, {}))
    code, out, err = run(repo, reqs, artifact(tmp_path), capsys)
    assert code == 0, err
    assert json.loads(out)["published"] == [{"eid": f"{CID}-r02", "layers": 2, "winner": "v1"},
                                            {"eid": f"{CID}-r03", "layers": 2, "winner": None}]
    dest = repo / LEDGER
    stack = json.loads((dest / "stack.json").read_text(encoding="utf-8"))
    assert stack["layers"][0] == TRUSTED_LAYER  # built on the trusted stack
    assert stack["layers"][1]["branch"] == f"exp/{CID}-r02/v1" and stack["layers"][1]["head"] == v1
    assert stack["stack_number"] == 5
    assert (dest / "rounds" / f"{CID}-r02.json").is_file()  # the rest of the artifact is imported
    assert [json.loads(x)["eid"] for x in (dest / "published.jsonl").read_text().splitlines()] == \
        [f"{CID}-r02", f"{CID}-r03"]

    # the workflow commits the ledger to exp-ledger/<cid> via a temp index (worktree stays at
    # github.sha); mirror that, then a rerun skips everything
    branch = f"exp-ledger/{CID}"
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp_path / "ledger.index")}
    g = ["git", "-C", str(repo)]
    subprocess.run([*g, "read-tree", branch], check=True, env=env)
    subprocess.run([*g, "add", "-A", LEDGER], check=True, env=env)
    tree = subprocess.run([*g, "write-tree"], check=True, env=env, capture_output=True, text=True).stdout.strip()
    commit = git(repo, "commit-tree", tree, "-p", branch, "-m", "publish")
    git(repo, "update-ref", f"refs/heads/{branch}", commit)
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert "README.md" in git(repo, "ls-tree", "-r", "--name-only", branch)  # base tree kept
    assert json.loads(git(repo, "show", f"{branch}:{LEDGER}/stack.json")) == stack
    led2 = tmp_path / "art2"
    os.rename(tmp_path / "art", led2)
    code, out, err = run(repo, led2 / "deferred.jsonl", led2 / "ledger", capsys)
    assert code == 0, err
    assert all(r.get("skipped") for r in json.loads(out)["published"])


def test_refuses_arm_edits_outside_the_harness_surface(repo: Path, tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    bad = arm(repo, "a2", ".github/workflows/evil.yml")
    code, _, err = run(repo, requests(tmp_path, req(f"{CID}-r02", "v1", {"v1": bad})), artifact(tmp_path), capsys)
    assert code == 1 and "REFUSED" in err and "edits outside" in err
    assert not (repo / LEDGER / "rounds").exists()  # nothing imported after a refusal


def test_refuses_the_frozen_harness_manifest(repo: Path, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    bad = arm(repo, "frozen", "harness/harness.yaml")
    code, _, err = run(repo, requests(tmp_path, req(f"{CID}-r02", "v1", {"v1": bad})),
                       artifact(tmp_path), capsys)
    assert code == 1 and "edits outside" in err and "harness.yaml" in err


def _commit(repo: Path, rel: str, text: str, msg: str) -> str:
    write(repo, rel, text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def test_refuses_transient_out_of_surface_edit_reverted_later(repo: Path, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    # net diff vs main only touches the harness, but the pushed ancestry carries a workflow
    git(repo, "checkout", "-q", "-b", "a3", "main")
    _commit(repo, ".github/workflows/evil.yml", "x\n", "add evil")
    git(repo, "rm", "-q", ".github/workflows/evil.yml")
    _commit(repo, HARNESS, "a3\n", "revert evil + harness edit")
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    assert git(repo, "diff", "--name-only", "main", sha) == HARNESS
    code, _, err = run(repo, requests(tmp_path, req(f"{CID}-r02", "v1", {"v1": sha})), artifact(tmp_path), capsys)
    assert code == 1 and "edits outside" in err and ".github/workflows/evil.yml" in err


def test_refuses_merge_commits_and_criss_cross_bases(repo: Path, tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    # criss-cross: main and side each merge the other's tip -> two merge bases
    git(repo, "checkout", "-q", "-b", "side", "main")
    side1 = _commit(repo, HARNESS, "side\n", "side")
    git(repo, "checkout", "-q", "main")
    main1 = _commit(repo, "README.md", "main1\n", "main1")
    git(repo, "merge", "-q", "--no-ff", "--no-edit", "-s", "ours", side1)
    git(repo, "checkout", "-q", "side")
    git(repo, "merge", "-q", "--no-ff", "--no-edit", "-s", "ours", main1)
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    assert len(git(repo, "merge-base", "--all", "main", head).split()) == 2
    code, _, err = run(repo, requests(tmp_path, req(f"{CID}-r02", "v1", {"v1": head})), artifact(tmp_path), capsys)
    assert code == 1 and "exactly one merge base" in err
    # a single merge base but a merge commit in arm history
    git(repo, "checkout", "-q", "-b", "m1", "main")
    _commit(repo, HARNESS, "m1\n", "m1")
    git(repo, "checkout", "-q", "-b", "m2", "main")
    m2 = _commit(repo, "harness/prompts/reflector.md", "m2\n", "m2")
    git(repo, "checkout", "-q", "m1")
    git(repo, "merge", "-q", "--no-ff", "--no-edit", m2)
    merged = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    code, _, err = run(repo, requests(tmp_path, req(f"{CID}-r02", "v1", {"v1": merged})), artifact(tmp_path), capsys)
    assert code == 1 and "merge commits are not allowed" in err


def test_missing_trusted_ref_falls_back_to_checkout_head(repo: Path, tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    # the ledger branch was merged + deleted: trust what github.sha (HEAD) has, never the artifact
    git(repo, "merge", "-q", "--ff-only", f"exp-ledger/{CID}")
    git(repo, "branch", "-q", "-D", f"exp-ledger/{CID}")
    v1 = arm(repo, "a4", HARNESS)
    code, out, err = run(repo, requests(tmp_path, req(f"{CID}-r02", "v1", {"v1": v1})), artifact(tmp_path), capsys)
    assert code == 0, err
    stack = json.loads((repo / LEDGER / "stack.json").read_text(encoding="utf-8"))
    assert stack["layers"][0] == TRUSTED_LAYER and stack["stack_number"] == 5
    assert json.loads(out)["published"][0]["layers"] == 2
    # and with neither the ref nor a committed ledger, the forged stack is still dropped
    assert cp.restore_trusted(repo, repo / "experiments" / "campaigns" / "nope", "exp-ledger/gone") == "HEAD"


@pytest.mark.parametrize(("row", "match"), [
    (req("other-r02", None, {}), "does not belong"),
    (req(f"{CID}-r02", "v9", {"v1": "a" * 40}), "has no head"),
    (req(f"{CID}-r02", "v1", {"v1": "not-a-sha"}), "sha"),
    (req(f"{CID}-r02", "v1", {"v1": "f" * 40}), "not a commit"),
    ({"eid": f"{CID}-r02", "winner": None, "heads": [], "title": "t", "body": "b"}, "malformed"),
])
def test_refuses_bad_requests(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], row: dict,
                              match: str) -> None:
    code, _, err = run(repo, requests(tmp_path, row), artifact(tmp_path), capsys)
    assert code == 1 and match in err.lower()


def test_refuses_too_many_requests(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [req(f"{CID}-r{i:02d}", None, {}) for i in range(1, cp.MAX_REQUESTS + 2)]
    code, _, err = run(repo, requests(tmp_path, *rows), artifact(tmp_path), capsys)
    assert code == 1 and "requests >" in err


def test_refuses_foreign_or_unsafe_ledger(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reqs = requests(tmp_path, req(f"{CID}-r02", None, {}))
    code, _, err = run(repo, reqs, artifact(tmp_path, cid="othercamp"), capsys)
    assert code == 1 and "not campaign" in err
    led = tmp_path / "art" / "ledger"
    (led / "campaign.json").write_text(json.dumps({"campaignId": CID}), encoding="utf-8")
    (led / "bad name.json").write_text("{}", encoding="utf-8")
    code, _, err = run(repo, reqs, led, capsys)
    assert code == 1 and "unsafe ledger entry" in err
    (led / "bad name.json").unlink()
    try:
        os.symlink(repo / "README.md", led / "link.json")
    except (OSError, NotImplementedError):
        return  # symlinks need privileges on some Windows hosts
    code, _, err = run(repo, reqs, led, capsys)
    assert code == 1 and "unsafe ledger entry" in err


def test_ledger_paths_are_validated(tmp_path: Path) -> None:
    ledger = cp.DirLedger(tmp_path)
    for rel in ("../x.json", "a//b.json", "", "campaigns/../../x"):
        with pytest.raises(cp.PublishError):
            ledger.read_json(rel)


def test_script_runs_with_isolated_stdlib_only_python() -> None:
    # -I -S: no site-packages, so any third-party import in the closure fails here
    proc = subprocess.run([sys.executable, "-I", "-S", "-B", str(SCRIPT), "--help"], capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, proc.stderr
