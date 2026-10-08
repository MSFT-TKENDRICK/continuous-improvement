from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ci_lab.sleep.bundle import FileChange, make_patch, write_bundle

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "sleep_publish.py"
SKILL = "src/order_support/harness/skills/order-support/SKILL.md"
STATE = "experiments/sleep/state.json"
PENDING = "experiments/sleep/tasks.pending.jsonl"


def _load():
    spec = importlib.util.spec_from_file_location("sleep_publish_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


sp = _load()


class FakeRun:
    """Real git against the temp repo; gh and push are intercepted and recorded."""

    def __init__(self, existing_pr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.existing_pr = existing_pr
        self.body: str | None = None

    def __call__(self, args, **kw):
        args = [str(a) for a in args]
        self.calls.append(args)
        if args[0] == "gh":
            if args[1:3] == ["pr", "list"]:
                return subprocess.CompletedProcess(args, 0, self.existing_pr, "")
            if args[1:3] == ["pr", "create"]:
                self.body = kw.get("input")
                return subprocess.CompletedProcess(args, 0, "https://github.com/o/r/pull/9\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")
        if "push" in args:
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.run(args, check=True, capture_output=True, text=True, **kw)

    def gh(self, *sub: str) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "gh" and tuple(c[1:1 + len(sub)]) == sub]


def head(repo: Path, h) -> str:
    return h.git(repo, "rev-parse", "HEAD")


def bundle(tmp_path: Path, *, patch: str, base_sha: str, kind: str = "sleep", accepted: bool = True,
           status: str | None = None, date: str = "20260921", ledger_update: bool | None = None,
           results: dict | None = None) -> Path:
    out = tmp_path / f"bundle-{kind}"
    status = status or ("accepted" if accepted else ("pending_review" if kind == "usage" else "rejected"))
    night_id = f"{kind}-{date}-1"
    write_bundle(out, patch=patch, experiment={"night_id": night_id},
                 results=results or {"status": status, "reasons": ["order-support: @everyone <b>|x|</b>"],
                                     "targets": {"order-support": {"status": status, "gate": {
                                         "delta_lcb": 0.12, "delta": 0.05, "critical_incumbent": 0,
                                         "critical_candidate": 0}}}},
                 base_sha=base_sha, night_id=night_id, date=date, accepted=accepted,
                 ledger_update=bool(patch) if ledger_update is None else ledger_update, status=status, kind=kind)
    return out


def skill_and_state(repo: Path) -> str:
    old_skill = (repo / SKILL).read_text(encoding="utf-8")
    old_state = (repo / STATE).read_text(encoding="utf-8")
    return make_patch([FileChange(SKILL, old_skill, old_skill + "\n## Verify identity\nAlways verify.\n"),
                       FileChange(STATE, old_state, old_state.replace('"night": 0', '"night": 1'))])


def state_only(repo: Path) -> str:
    old = (repo / STATE).read_text(encoding="utf-8")
    return make_patch([FileChange(STATE, old, old.replace('"night": 0', '"night": 1'))])


def run_main(b: Path, repo: Path, run: FakeRun, monkeypatch, *extra: str, attempt: str = "3") -> int:
    monkeypatch.setenv("SLEEP_RUN_ATTEMPT", attempt)
    monkeypatch.setenv("SLEEP_BASE_REF", "main")
    return sp.main(["--bundle", str(b), "--repo", str(repo), *extra], run=run)


# ------------------------------------------------------------------ happy paths

def test_publishes_accepted_candidate_as_draft_pr(sleep_repo, tmp_path, h, monkeypatch, capsys):
    b = bundle(tmp_path, patch=skill_and_state(sleep_repo), base_sha=head(sleep_repo, h))
    run = FakeRun()
    assert run_main(b, sleep_repo, run, monkeypatch) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"published": True, "branch": "exp/sleep-20260921-3/cand",
                   "url": "https://github.com/o/r/pull/9", "pr": "created"}
    assert h.git(sleep_repo, "rev-parse", "--abbrev-ref", "HEAD") == "exp/sleep-20260921-3/cand"
    assert sorted(h.git(sleep_repo, "show", "--name-only", "--format=", "HEAD").splitlines()) == [STATE, SKILL]
    msg = h.git(sleep_repo, "log", "-1", "--format=%B")
    assert "Sleep-Night-Id: sleep-20260921-1" in msg and "Sleep-Decision: accepted" in msg
    assert "Sleep-Bundle-Manifest: sha256:" in msg
    [create] = run.gh("pr", "create")
    assert "--draft" in create and create[create.index("--base") + 1] == "main"
    assert "@\u200beveryone" in run.body and "<b>" not in run.body and "sleep-20260921-1" in run.body
    assert not any("merge" in a for c in run.calls for a in c)
    assert [c for c in run.calls if "push" in c][0][-1] == "HEAD:refs/heads/exp/sleep-20260921-3/cand"


def test_lessons_proposal_file_is_published_with_note(sleep_repo, tmp_path, h, monkeypatch):
    lessons = "experiments/sleep/lessons/sleep-20260921-1.json"
    patch = make_patch([FileChange(STATE, (sleep_repo / STATE).read_text(encoding="utf-8"),
                                   (sleep_repo / STATE).read_text(encoding="utf-8").replace('"night": 0', '"night": 1')),
                        FileChange(lessons, None, '{"status": "proposed"}\n')])
    b = bundle(tmp_path, patch=patch, base_sha=head(sleep_repo, h), accepted=False)
    run = FakeRun()
    assert run_main(b, sleep_repo, run, monkeypatch) == 0
    assert sorted(h.git(sleep_repo, "show", "--name-only", "--format=", "HEAD").splitlines()) == sorted([STATE, lessons])
    assert "--draft" in run.gh("pr", "create")[0]
    assert "never adopted or enforced" in run.body and f"`{lessons}`" in run.body
    assert not any("merge" in a for c in run.calls for a in c)


def test_ledger_only_rejected_night_and_existing_pr(sleep_repo, tmp_path, h, monkeypatch):
    b = bundle(tmp_path, patch=state_only(sleep_repo), base_sha=head(sleep_repo, h), accepted=False)
    run = FakeRun(existing_pr="https://github.com/o/r/pull/2\n")
    assert run_main(b, sleep_repo, run, monkeypatch) == 0
    assert run.gh("pr", "create") == []


def test_nothing_to_publish_and_dry_run(sleep_repo, tmp_path, h, monkeypatch, capsys):
    sha = head(sleep_repo, h)
    empty = bundle(tmp_path, patch="", base_sha=sha, accepted=False, status="error")
    assert run_main(empty, sleep_repo, FakeRun(), monkeypatch) == 0
    assert "nothing to publish" in capsys.readouterr().out
    b = bundle(tmp_path, patch=skill_and_state(sleep_repo), base_sha=sha)
    run = FakeRun()
    assert run_main(b, sleep_repo, run, monkeypatch, "--dry-run") == 0
    assert h.git(sleep_repo, "rev-parse", "--abbrev-ref", "HEAD") == "main" and run.gh() == []


def test_usage_bundle_publishes_pending_file_only(sleep_repo, tmp_path, h, monkeypatch):
    patch = make_patch([FileChange(PENDING, None, '{"format": "skillopt_sleep.tasks.v1", "reviewed": false}\n')])
    b = bundle(tmp_path, patch=patch, base_sha=head(sleep_repo, h), kind="usage", accepted=False,
               date="20260923", results={"kind": "usage", "status": "pending_review"})
    run = FakeRun()
    assert run_main(b, sleep_repo, run, monkeypatch) == 0
    assert h.git(sleep_repo, "rev-parse", "--abbrev-ref", "HEAD") == "exp/usage-20260923/tasks"
    assert "Usage-Night-Id: usage-20260923-1" in h.git(sleep_repo, "log", "-1", "--format=%B")
    assert "--draft" in run.gh("pr", "create")[0] and "reviewed: true" in run.body


# ------------------------------------------------------------------ refusals

def _refused(b: Path, repo: Path, monkeypatch, capsys, needle: str) -> None:
    run = FakeRun()
    assert run_main(b, repo, run, monkeypatch) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err and needle in err, err
    assert run.gh() == [] and not any("push" in c for c in run.calls)


RAW = "diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n"


@pytest.mark.parametrize("patch,needle", [
    (RAW.format(p="experiments/sleep/../../.github/workflows/x.yml"), "illegal path"),
    (RAW.format(p=".github/workflows/sleep-nightly.yml"), "allowlist"),
    (RAW.format(p="scripts/sleep_publish.py"), "allowlist"),
    (RAW.format(p="experiments/sleep/.git/config"), "illegal path"),
    (RAW.format(p=PENDING), "pending"),
    ("diff --git a/experiments/sleep/a b/experiments/sleep/b\n", "unexpected patch line"),
    ("diff --git a/experiments/sleep/x.json b/experiments/sleep/x.json\nold mode 100644\nnew mode 100755\n",
     "forbidden"),
    ("diff --git a/experiments/sleep/x.json b/experiments/sleep/x.json\nrename from a\nrename to b\n", "forbidden"),
    ("diff --git a/experiments/sleep/x.json b/experiments/sleep/x.json\ndeleted file mode 100644\n", "forbidden"),
    ("diff --git a/experiments/sleep/x.json b/experiments/sleep/x.json\nnew file mode 120000\n", "unexpected"),
    ("diff --git a/experiments/sleep/x.json b/experiments/sleep/x.json\n--- a/experiments/sleep/x.json\n"
     "+++ b/.github/x\n@@ -1 +1 @@\n-a\n+b\n", "headers"),
    (RAW.format(p="experiments/sleep/x.json") + "Binary files differ\n", "unexpected line"),
])
def test_rejects_malicious_patches(sleep_repo, tmp_path, h, monkeypatch, capsys, patch, needle):
    b = bundle(tmp_path, patch=patch, base_sha=head(sleep_repo, h))
    _refused(b, sleep_repo, monkeypatch, capsys, needle)


def test_rejects_skill_change_when_not_accepted(sleep_repo, tmp_path, h, monkeypatch, capsys):
    b = bundle(tmp_path, patch=skill_and_state(sleep_repo), base_sha=head(sleep_repo, h), accepted=False)
    _refused(b, sleep_repo, monkeypatch, capsys, "not accepted")


def test_rejects_usage_bundle_touching_reviewed_tasks(sleep_repo, tmp_path, h, monkeypatch, capsys):
    old = (sleep_repo / "experiments/sleep/tasks.jsonl").read_text(encoding="utf-8")
    patch = make_patch([FileChange("experiments/sleep/tasks.jsonl", old, old + "{}\n")])
    b = bundle(tmp_path, patch=patch, base_sha=head(sleep_repo, h), kind="usage", accepted=False, date="20260923")
    _refused(b, sleep_repo, monkeypatch, capsys, "pending.jsonl")


def test_rejects_digest_mismatch(sleep_repo, tmp_path, h, monkeypatch, capsys):
    b = bundle(tmp_path, patch=state_only(sleep_repo), base_sha=head(sleep_repo, h), accepted=False)
    (b / "results.json").write_text('{"status": "accepted"}\n', encoding="utf-8")
    _refused(b, sleep_repo, monkeypatch, capsys, "digest mismatch")
    b2 = bundle(tmp_path / "2", patch=state_only(sleep_repo), base_sha=head(sleep_repo, h), accepted=False)
    (b2 / "candidate.patch").write_text(skill_and_state(sleep_repo), encoding="utf-8", newline="\n")
    _refused(b2, sleep_repo, monkeypatch, capsys, "digest mismatch")


def test_rejects_base_sha_mismatch(sleep_repo, tmp_path, monkeypatch, capsys):
    b = bundle(tmp_path, patch=skill_and_state(sleep_repo), base_sha="c" * 40)
    _refused(b, sleep_repo, monkeypatch, capsys, "base sha mismatch")


def test_rejects_extra_file_and_inconsistent_manifest(sleep_repo, tmp_path, h, monkeypatch, capsys):
    sha = head(sleep_repo, h)
    b = bundle(tmp_path, patch=state_only(sleep_repo), base_sha=sha, accepted=False)
    (b / "extra.sh").write_text("echo hi\n", encoding="utf-8")
    _refused(b, sleep_repo, monkeypatch, capsys, "exactly")
    b2 = bundle(tmp_path / "2", patch=state_only(sleep_repo), base_sha=sha, accepted=True, status="rejected")
    _refused(b2, sleep_repo, monkeypatch, capsys, "inconsistent")


def test_rejects_dirty_checkout_and_bad_env(sleep_repo, tmp_path, h, monkeypatch, capsys):
    b = bundle(tmp_path, patch=state_only(sleep_repo), base_sha=head(sleep_repo, h), accepted=False)
    run = FakeRun()
    assert run_main(b, sleep_repo, run, monkeypatch, attempt="1; rm -rf /") == 1
    assert "bad run attempt" in capsys.readouterr().err
    (sleep_repo / STATE).write_text("{}\n", encoding="utf-8")
    _refused(b, sleep_repo, monkeypatch, capsys, "local modifications")


def test_patch_that_does_not_apply_is_refused(sleep_repo, tmp_path, h, monkeypatch, capsys):
    patch = make_patch([FileChange(STATE, "totally different\n", "x\n")])
    b = bundle(tmp_path, patch=patch, base_sha=head(sleep_repo, h), accepted=False)
    _refused(b, sleep_repo, monkeypatch, capsys, "git -C")
    assert h.git(sleep_repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_script_is_stdlib_only():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods - {"__future__"} <= set(sys.stdlib_module_names), mods
