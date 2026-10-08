from __future__ import annotations

import json
from pathlib import Path

import pytest
from skillopt_sleep.types import TaskRecord

from ci_lab.contracts import FailureRecord, Transcript
from ci_lab.sleep.backend import ReflectRequest
from ci_lab.sleep.reflector import make_maf_reflector, render_request
from ci_lab.sleep.target import compose_instructions, make_maf_run_target, materialize_harness
from ci_lab.sleep.wiring import PolicyOracle
from ci_lab.testing import Call, FakeChatClient


def test_materialize_harness_writes_skill_path(tmp_path):
    base = tmp_path / "base"
    (base / "prompts").mkdir(parents=True)
    (base / "prompts" / "system.md").write_text("SYSTEM", encoding="utf-8")
    d1 = materialize_harness("SKILL A", "", tmp_path / "root", base)
    assert (d1 / "skills" / "order-support" / "SKILL.md").read_text(encoding="utf-8") == "SKILL A"
    assert (d1 / "prompts" / "system.md").exists()
    assert materialize_harness("SKILL A", "", tmp_path / "root", base) == d1
    assert materialize_harness("SKILL B", "", tmp_path / "root", base) != d1
    text = compose_instructions(d1)
    assert text.startswith("SYSTEM") and "SKILL A" in text


def test_maf_run_target_injects_skill_and_records_real_tool_calls(tmp_path):
    clients: list[FakeChatClient] = []

    def factory():
        c = FakeChatClient([[Call("lookup_order", {"order_id": "NW-10001"})], "Your order shipped."])
        clients.append(c)
        return c

    executed = []

    def execute(name, args):
        executed.append((name, args))
        return {"order_id": args.get("order_id"), "status": "shipped"}

    run = make_maf_run_target(factory, harness_root=tmp_path / "h", system_prompt="SYS", execute=execute)
    t = TaskRecord(id="t1", project="p", intent="Where is NW-10001?")
    reply, tools, tr = run(t, "SKILL-MARKER-123", "")
    assert reply == "Your order shipped." and tools == ["lookup_order"]
    assert executed == [("lookup_order", {"order_id": "NW-10001"})]
    assert isinstance(tr, Transcript) and tr.tool_calls[0].result["status"] == "shipped"
    msgs, opts = clients[0].requests[0]
    assert "SKILL-MARKER-123" in str(opts.get("instructions", "")) + " ".join(m.text or "" for m in msgs)
    assert list((tmp_path / "h").glob("*/skills/order-support/SKILL.md"))


def _req() -> ReflectRequest:
    f = FailureRecord(case_id="c1", suite="refunds", category="verify_identity",
                      rule_ids=("check.tool_called:lookup_order",), rubric_scores={"rule_checks": 0.5},
                      excerpt="Sorry")
    return ReflectRequest(failures=(f,), n_successes=2, target="skill", edit_budget=2, learned=("old line",))


def test_render_request_contains_only_typed_fields():
    doc = json.loads(render_request(_req()))
    assert set(doc) == {"target", "edit_budget", "n_successes", "learned_lines", "failures"}
    assert set(doc["failures"][0]) == {"case_id", "suite", "category", "rule_ids", "rubric_scores", "excerpt"}


def test_maf_reflector_returns_typed_edits():
    edits = [{"op": "add", "content": "Always call lookup_order first.", "rationale": "r"}]
    client = FakeChatClient([[Call("submit_edits", {"edits": edits})], "done"])
    res = make_maf_reflector(lambda: client)(_req())
    assert [(e.target, e.op, e.content) for e in res.edits] == [("skill", "add", "Always call lookup_order first.")]
    msgs, _ = client.requests[0]
    assert any("check.tool_called:lookup_order" in (m.text or "") for m in msgs)


def test_policy_oracle_fallback():
    from ci_lab.contracts import ToolCallRecord

    msgs = [{"role": "user", "content": "refund NW-10001, I'm a@example.com"}, {"role": "assistant", "content": "ok"}]
    lookup = ToolCallRecord("1", "lookup_order", {"order_id": "NW-10001"},
                            {"order_id": "NW-10001", "email": "a@example.com", "refund_eligible": True,
                             "total": 50}, 0)
    refund = ToolCallRecord("2", "issue_refund", {"order_id": "NW-10001", "amount": 20}, {"ok": True}, 1)
    assert PolicyOracle().check(Transcript(case_id="x", messages=msgs, tool_calls=(lookup, refund))) == []
    v = PolicyOracle().check(Transcript(case_id="x", messages=msgs, tool_calls=(refund,)))
    assert [x.rule_id for x in v] == ["refund.unverified_identity"]


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
    assert json.loads(capsys.readouterr().out)["order-support"]["n_tasks"] == 6


def test_cli_dry_run_rejects_heldout_export(sleep_repo, tmp_path):
    from ci_lab import cli

    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"dataset_split": "heldout", "task": {"id": "a", "project": "p", "intent": "i"}}) + "\n",
                 encoding="utf-8")
    assert cli.main(["sleep", "dry-run", "--repo", str(sleep_repo), "--agl-export", str(p)]) == 1
