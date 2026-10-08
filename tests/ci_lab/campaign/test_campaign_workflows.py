"""Static lint of campaign-scheduled.yml / tests.yml plus repo-wide workflow invariants (design C10)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WF = ROOT / ".github" / "workflows"
SHA_PIN = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
ALL = sorted(p.name for p in WF.glob("*.yml"))


def load(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text(encoding="utf-8"))


def triggers(doc: dict) -> dict:
    return doc.get("on", doc.get(True))


def steps(job: dict) -> list[dict]:
    return job.get("steps") or []


@pytest.mark.parametrize("name", ALL)
def test_no_runner_context_in_job_env(name):
    """`runner.*` is not available in job-level env (actionlint); derive paths in a step."""
    for jname, job in load(name)["jobs"].items():
        assert "runner." not in repr(job.get("env") or {}), (name, jname)


@pytest.mark.parametrize("name", ["campaign-scheduled.yml", "tests.yml"])
def test_hardening(name):
    doc = load(name)
    raw = (WF / name).read_text(encoding="utf-8")
    assert "pull_request_target" not in raw and "secrets." not in raw
    for jname, job in doc["jobs"].items():
        assert isinstance(job.get("permissions"), dict), jname
        assert isinstance(job.get("timeout-minutes"), int), jname
        for st in steps(job):
            if "uses" in st:
                assert SHA_PIN.match(st["uses"]), st["uses"]
                if st["uses"].startswith("actions/checkout@"):
                    assert st.get("with", {}).get("persist-credentials") is False, jname
            run = st.get("run", "")
            assert "${{" not in run, (jname, run)  # expressions only via env:
    assert "uv.lock is intentionally not committed" in raw and "--frozen" not in raw


def test_tests_workflow_runs_pytest_and_canvas_tests_read_only():
    doc = load("tests.yml")
    assert set(triggers(doc)) == {"pull_request", "push"}
    assert doc["permissions"] == {"contents": "read"}
    (job,) = doc["jobs"].values()
    runs = "\n".join(s.get("run", "") for s in steps(job))
    assert "uv run pytest" in runs and "node --test .github/extensions/ci-harness-dashboard/test/" in runs
    assert "GH_TOKEN" not in repr(job) and "COPILOT_GITHUB_TOKEN" not in repr(job)


def test_campaign_jobs_split_privileges():
    doc = load("campaign-scheduled.yml")
    assert doc["permissions"] == {}
    assert set(triggers(doc)) == {"schedule", "workflow_dispatch"}
    assert doc["concurrency"]["cancel-in-progress"] is False
    jobs = doc["jobs"]
    assert set(jobs) == {"opt_in", "gate", "round", "publish"}
    gate = jobs["gate"]
    assert gate["permissions"] == {}
    gate_run = steps(gate)[0]["run"]
    # dispatch inputs reach the shell only through env and are validated before any use
    assert "inputs.cid" in steps(gate)[0]["env"]["CID"]
    assert "^[a-z0-9][a-z0-9-]{2,40}$" in gate_run and "^[1-9]$" in gate_run
    rnd = jobs["round"]
    assert rnd["permissions"] == {"contents": "read", "copilot-requests": "write"}
    runs = "\n".join(s.get("run", "") for s in steps(rnd))
    assert "--defer-publish" in runs and "campaign publish" not in runs
    assert "git push" not in runs and "gh pr" not in runs
    # absolute --rounds: ROUNDS more than the ledger records; fail closed on half-published rounds
    assert ".rounds | length" in runs and '--rounds "$target"' in runs
    assert 'refs/remotes/origin/exp/$eid/' in runs and 'refs/tags/exp-archive/$eid/' in runs
    assert runs.index("exp-archive/$eid/") < runs.index("--defer-publish")
    pub = jobs["publish"]
    assert pub["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert pub["environment"] == "campaign-publish"
    assert "deferred == 'true'" in pub["if"]


def test_campaign_publish_job_runs_no_third_party_code():
    pub = load("campaign-scheduled.yml")["jobs"]["publish"]
    uses = [s["uses"].split("@")[0] for s in steps(pub) if "uses" in s]
    assert uses == ["actions/checkout", "actions/download-artifact"]
    assert next(s for s in steps(pub) if "uses" in s)["with"]["ref"] == "${{ github.sha }}"
    text = repr(pub)
    assert not re.search(r"\buv\b", text) and "setup-uv" not in text and "pip" not in text
    runs = [s["run"] for s in steps(pub) if "run" in s]
    assert any(r.strip().startswith("python3 -I -B scripts/campaign_publish.py") for r in runs)
    assert not any("merge" in r for r in runs)
    # the worktree stays at github.sha (the script's code); the ledger commit uses a temp index
    assert not any("git checkout" in r for r in runs)
    assert any("GIT_INDEX_FILE" in r and "commit-tree" in r for r in runs)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@e", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_arm_bundle_roundtrip_matches_workflow(tmp_path: Path):
    """The round job's `git bundle create` / the publish job's `git fetch <bundle>` refspecs."""
    origin, work, pub = tmp_path / "origin.git", tmp_path / "work", tmp_path / "pub"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(work))
    (work / "a.txt").write_text("a\n", encoding="utf-8")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "base")
    git(work, "push", "-q", "origin", "HEAD:main")
    git(work, "fetch", "-q")
    git(work, "checkout", "-q", "-b", "exp/c1-r01/v1")
    (work / "a.txt").write_text("v1\n", encoding="utf-8")
    git(work, "commit", "-qam", "v1")
    v1 = git(work, "rev-parse", "HEAD")
    git(work, "checkout", "-q", "-b", "tmp-arm", "main")
    (work / "a.txt").write_text("v2\n", encoding="utf-8")
    git(work, "commit", "-qam", "v2")
    v2 = git(work, "rev-parse", "HEAD")
    git(work, "checkout", "-q", "main")
    git(work, "branch", "-q", "-D", "tmp-arm")  # head survives only via the ci-heads pin
    git(work, "update-ref", f"refs/heads/ci-heads/{v2}", v2)
    bundle = tmp_path / "arms.bundle"
    git(work, "bundle", "create", str(bundle), "--branches=exp/*", "--branches=ci-heads/*", "--not", "--remotes")
    git(tmp_path, "clone", "-q", str(origin), str(pub))
    git(pub, "bundle", "verify", str(bundle))
    git(pub, "fetch", "-q", str(bundle), "refs/heads/exp/*:refs/heads/exp/*", "refs/heads/ci-heads/*:refs/ci-heads/*")
    assert git(pub, "cat-file", "-t", v1) == "commit" and git(pub, "cat-file", "-t", v2) == "commit"
    assert git(pub, "rev-parse", "exp/c1-r01/v1") == v1
