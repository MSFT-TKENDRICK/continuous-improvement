from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from skillopt_sleep.types import TaskRecord

from ci_lab.contracts import FailureRecord, Transcript
from ci_lab.sleep.backend import ReflectRequest
from ci_lab.sleep.reflector import make_maf_reflector, render_request
from ci_lab.sleep.registry import HARNESS_EDITING
from ci_lab.sleep.target import make_harness_run_target, materialize_harness
from ci_lab.testing import Call, FakeChatClient

ROOT = Path(__file__).resolve().parents[3]
SKILL_REL = Path("skills/harness-editing/SKILL.md")


def test_materialize_harness_writes_requested_skill_path(tmp_path):
    base = tmp_path / "base"
    (base / "prompts").mkdir(parents=True)
    (base / "prompts" / "system.md").write_text("SYSTEM", encoding="utf-8")
    d1 = materialize_harness("SKILL A", "", tmp_path / "root", base, skill_rel=SKILL_REL)
    assert (d1 / SKILL_REL).read_text(encoding="utf-8") == "SKILL A"
    assert (d1 / "prompts" / "system.md").exists()
    assert materialize_harness("SKILL A", "", tmp_path / "root", base, skill_rel=SKILL_REL) == d1
    assert materialize_harness("SKILL B", "", tmp_path / "root", base, skill_rel=SKILL_REL) != d1


def test_harness_run_target_injects_skill_and_records_tool_calls(tmp_path):
    base = tmp_path / "base"
    shutil.copytree(ROOT / "harness", base)
    clients: list[FakeChatClient] = []

    def factory():
        client = FakeChatClient([[Call("list_files", {"glob": "**/*"})], "Inspected the harness."])
        clients.append(client)
        return client

    run = make_harness_run_target(
        factory,
        harness_root=tmp_path / "materialized",
        base_harness=base,
        owner_agent=HARNESS_EDITING.owner_agent,
        skill_path=HARNESS_EDITING.skill_path,
    )
    task = TaskRecord(id="t1", project="harness-editing", intent="Inspect the harness files.")
    skill = "---\nname: harness-editing\ndescription: SKILL-MARKER-123\n---\n\nInspect before editing.\n"
    reply, tools, transcript = run(task, skill, "")
    assert reply == "Inspected the harness." and tools == ["list_files"]
    assert isinstance(transcript, Transcript) and transcript.tool_calls[0].name == "list_files"
    assert list((tmp_path / "materialized").glob("*/skills/harness-editing/SKILL.md"))
    messages, options = clients[0].requests[0]
    rendered = str(options.get("instructions", "")) + " ".join(message.text or "" for message in messages)
    assert "SKILL-MARKER-123" in rendered


def _req() -> ReflectRequest:
    f = FailureRecord(case_id="c1", suite="harness", category="inspect_before_edit",
                      rule_ids=("check.tool_called:read_file",), rubric_scores={"rule_checks": 0.5},
                      excerpt="Sorry")
    return ReflectRequest(failures=(f,), n_successes=2, target="skill", edit_budget=2, learned=("old line",))


def test_render_request_contains_only_typed_fields():
    doc = json.loads(render_request(_req()))
    assert set(doc) == {"target", "edit_budget", "n_successes", "learned_lines", "failures"}
    assert set(doc["failures"][0]) == {"case_id", "suite", "category", "rule_ids", "rubric_scores", "excerpt"}


def test_maf_reflector_returns_typed_edits():
    edits = [{"op": "add", "content": "Always call read_file first.", "rationale": "r"}]
    client = FakeChatClient([[Call("submit_edits", {"edits": edits})], "done"])
    res = make_maf_reflector(lambda: client)(_req())
    assert [(e.target, e.op, e.content) for e in res.edits] == [("skill", "add", "Always call read_file first.")]
    msgs, _ = client.requests[0]
    assert any("check.tool_called:read_file" in (m.text or "") for m in msgs)


@pytest.mark.parametrize("profile", ["copilot", "offline"])
def test_real_profiles_fail_closed_without_assert_domain(tmp_path, profile, monkeypatch):
    from ci_lab.contracts import Profile
    from ci_lab.sleep import wiring
    from ci_lab.sleep.night import SleepConfig

    monkeypatch.setattr(wiring, "client_factory", lambda p, purpose: (lambda: FakeChatClient()))
    monkeypatch.setattr(wiring, "_probe", lambda c: None)
    cfg = SleepConfig(repo_root=tmp_path, out_dir=tmp_path / "out")
    with pytest.raises(wiring.WiringError, match="ASSERT"):
        wiring.build_deps(Profile(profile), cfg)


def test_cli_fake_run_and_dry_run(sleep_repo, tmp_path, monkeypatch, capsys, h):
    from ci_lab import cli

    out = tmp_path / "bundle"
    gh_out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    rc = cli.main(["sleep", "run", "--profile", "fake", "--out", str(out), "--repo", str(sleep_repo),
                   "--max-tasks", "10", "--max-minutes", "5", "--date", "20260921"])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["night_id"] == "sleep-20260921-3" and printed["status"] in ("accepted", "rejected")
    outputs = dict(line.split("=", 1) for line in gh_out.read_text(encoding="utf-8").splitlines())
    assert outputs["ledger_update"] == "true" and outputs["status"] == printed["status"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_sha"] == h.git(sleep_repo, "rev-parse", "HEAD")
    assert cli.main(["sleep", "dry-run", "--repo", str(sleep_repo)]) == 0
    out_doc = json.loads(capsys.readouterr().out)
    assert out_doc["harness-editing"]["n_tasks"] == 6
    assert out_doc["trace-triage"]["n_tasks"] == 6


def test_cli_dry_run_rejects_heldout_export(sleep_repo, tmp_path):
    from ci_lab import cli

    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"dataset_split": "heldout", "task": {"id": "a", "project": "p", "intent": "i"}}) + "\n",
                 encoding="utf-8")
    assert cli.main(["sleep", "dry-run", "--repo", str(sleep_repo), "--agl-export", str(p)]) == 1
