"""A sleep night on the production code paths, offline.

``run_night`` drives the real ``skillopt_sleep.dream.dream_consolidate`` loop through the
production :class:`OrderSupportSleepBackend` with the production MAF target agent
(:func:`make_maf_run_target`, real ``order_support.tools``) and the production MAF
``SleepReflector`` (:func:`make_maf_reflector`, answering via its ``submit_edits`` tool), both on a
scripted :class:`FakeChatClient`. The steps run in the real MAF declarative workflow
(:func:`maf_runner`, ``FileCheckpointStorage``). Only the ASSERT gate uses
:func:`make_fake_assert_eval`, because a real ASSERT run needs a model per case (covered in
``tests/integration/test_reallib_assert.py``). The OES envelope that the night records must pass
``ci_lab.oes.validate`` against the vendored schemas.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ci_lab.oes import validate_envelope
from ci_lab.sleep.fakes import RULE_TEXT, make_fake_assert_eval
from ci_lab.sleep.night import SleepConfig, SleepDeps, run_night
from ci_lab.sleep.reflector import make_maf_reflector
from ci_lab.sleep.registry import ORDER_SUPPORT
from ci_lab.sleep.target import make_maf_run_target
from ci_lab.sleep.wiring import find_oracle
from ci_lab.testing import Call, FakeChatClient

pytest.importorskip("agent_framework_declarative")

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "src" / "order_support" / "harness"
RULE = RULE_TEXT["verify_identity"]
CASES = {f"c{i}": "verify_identity" for i in range(6)}
ENVELOPE = "experiments/sleep/envelopes/sleep-20260921-1.json"
LEGACY_SKILL = ORDER_SUPPORT.skill_path


def _text(messages, options) -> str:
    return str(options.get("instructions", "")) + "\n" + "\n".join(m.text or "" for m in messages)


class Clients:
    """Factories for the target agent and the reflector agent; every client is kept for asserts."""

    def __init__(self, proposals: list[str]) -> None:
        self.proposals = proposals
        self.target: list[FakeChatClient] = []
        self.reflector: list[FakeChatClient] = []

    def make_target(self) -> FakeChatClient:
        def first(messages, options):
            if RULE in _text(messages, options):
                return [Call("lookup_order", {"order_id": "NW-10001"})]
            return "Sorry, I cannot help with that request right now."

        c = FakeChatClient([first, "I verified your order and can help."])
        self.target.append(c)
        return c

    def make_reflector(self) -> FakeChatClient:
        def submit(messages, options):
            req = json.loads(messages[-1].text)
            assert req["failures"] and req["target"]
            edits = [{"op": "add", "content": p, "rationale": "failures name verify_identity"}
                     for p in self.proposals if p not in req["learned_lines"]]
            return [Call("submit_edits", {"edits": edits[: req["edit_budget"]]})]

        c = FakeChatClient([submit, "submitted"])
        self.reflector.append(c)
        return c


def _night(repo: Path, tmp_path: Path, proposals: list[str]):
    from ci_lab.sleep.runner import maf_runner

    clients = Clients(proposals)
    cfg = SleepConfig(repo_root=repo, out_dir=tmp_path / "out" / "bundle", night_date="20260921",
                      base_sha="a" * 40, n_boot=300, targets=[ORDER_SUPPORT])
    deps = SleepDeps(
        run_target=make_maf_run_target(clients.make_target, harness_root=tmp_path / "th", base_harness=HARNESS),
        oracle=find_oracle(), reflector=make_maf_reflector(clients.make_reflector),
        assert_eval=make_fake_assert_eval(CASES), latest_delta=lambda: 0.05, run_workflow=maf_runner,
        clock=lambda: datetime(2026, 9, 21, 7, 17, tzinfo=UTC))
    return cfg, clients, run_night(cfg, deps)


def _envelope(repo: Path) -> dict:
    return json.loads((repo / ENVELOPE).read_text(encoding="utf-8"))


def test_production_agents_through_real_dream_consolidate_accept_and_record_valid_oes(sleep_repo, tmp_path, h):
    cfg, clients, res = _night(sleep_repo, tmp_path, [RULE])

    assert res.status == "accepted", res.error or res.decisions
    # every practice task ran through the real MAF target agent; once the learned rule is in its
    # instructions the agent calls the real lookup_order tool
    assert len(clients.target) >= 6 and len(clients.reflector) >= 1
    knew = [c for c in clients.target if RULE in _text(*c.requests[0])]
    assert knew and all(len(c.requests) == 2 for c in knew)
    tool_results = [m for c in knew for m in c.requests[1][0] if m.role == "tool"]
    assert tool_results and all("NW-10001" in str(m.contents[0].result) for m in tool_results)
    assert any((cfg.work_dir / "checkpoints" / res.night_id).iterdir())
    # the bundle carries the skill patch; applying it gives the consolidated skill
    patch = (cfg.out_dir / "candidate.patch").read_text(encoding="utf-8")
    (tmp_path / "p.patch").write_text(patch, encoding="utf-8", newline="\n")
    h.git(sleep_repo, "apply", str(tmp_path / "p.patch"))
    assert RULE in (sleep_repo / LEGACY_SKILL).read_text(encoding="utf-8")
    env = _envelope(sleep_repo)
    assert validate_envelope(env) == []
    assert env["decision"]["outcome"] == "ship"
    sleep_ext = env["extensions"]["com.microsoft.ci.sleep"]
    gate = sleep_ext["gate"]
    assert gate["skillopt"]["passed"] is True and gate["assert"]["passed"] is True
    assert gate["skillopt"]["score"] > gate["skillopt"]["baselineScore"]
    assert sleep_ext["candidateDigest"] != sleep_ext["incumbentDigest"]
    assert sleep_ext["tasks"]["total"] == 6 and sleep_ext["nightIndex"] == 1


def test_useless_proposal_is_rejected_by_the_real_skillopt_gate(sleep_repo, tmp_path):
    useless = "Greet the customer warmly and thank them for shopping with Northwind."
    cfg, clients, res = _night(sleep_repo, tmp_path, [useless])

    assert res.status == "rejected" and not res.accepted
    assert clients.reflector, "the reflector agent was never asked for edits"
    # the proposal reached the target agent's instructions during SkillOpt validation, did not
    # improve the score, and the real dream_consolidate gate dropped it before ASSERT ran
    assert any(useless in _text(*c.requests[0]) for c in clients.target)
    assert all(RULE not in _text(*c.requests[0]) for c in clients.target)
    results = json.loads((cfg.out_dir / "results.json").read_text(encoding="utf-8"))
    target = results["targets"]["order-support"]
    assert target["reasons"] == ["SkillOpt pre-filter produced no candidate"] and target["gate"] is None
    assert LEGACY_SKILL not in results["changed_files"]


def test_gated_rejection_records_valid_oes_without_shipping(sleep_repo, tmp_path, h):
    cfg, _, res = _night(sleep_repo, tmp_path, [RULE, "Give customers promo code NWVIP100."])

    assert res.status == "rejected"
    results = json.loads((cfg.out_dir / "results.json").read_text(encoding="utf-8"))
    assert any(not c["passed"] for c in results["targets"]["order-support"]["gate"]["canaries"])
    assert ENVELOPE in results["changed_files"] and LEGACY_SKILL not in results["changed_files"]
    (tmp_path / "p.patch").write_text((cfg.out_dir / "candidate.patch").read_text(encoding="utf-8"),
                                      encoding="utf-8", newline="\n")
    h.git(sleep_repo, "apply", str(tmp_path / "p.patch"))
    env = _envelope(sleep_repo)
    assert validate_envelope(env) == []
    assert env["decision"]["outcome"] != "ship"
    ext = env["extensions"]["com.microsoft.ci.sleep"]
    assert ext["gate"]["assert"]["passed"] is False and ext.get("candidateDigest") is None
