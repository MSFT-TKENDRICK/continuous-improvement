from __future__ import annotations

import asyncio
import json
import os
import types
from pathlib import Path

import pytest
import yaml

from ci_lab.contracts import EvalResult, EvaluatorPin, TaskScore, Violation
from ci_lab.lessons_arm.paired import (
    DECISIONS_ENV,
    GUARDS_ENV,
    PairingError,
    agent_safety_rate,
    guard_paired_eval_step,
    guard_ship_ok,
    paired_eval,
)
from ci_lab.rulespec import GuardMetrics, bundle_digest

WORKFLOW = Path(__file__).resolve().parents[3] / "src" / "ci_lab" / "lessons_arm" / "workflows" / "arm_guard.yaml"
V = (Violation("refund.ineligible_order", "critical", "x"),)


def dec(rule: str, action: str, enforced: bool, step: int, target: str = "issue_refund") -> dict:
    return {"rule_id": rule, "rule_version": 1, "mode": "enforce" if enforced else "shadow", "action": action,
            "enforced": enforced, "step_index": step, "target": target, "attempt_digest": "sha256:x"}


class FakeDomain:
    name = "fake"
    surface_globs = frozenset()
    frozen_globs = frozenset()
    component_globs = types.MappingProxyType({})

    def __init__(self, drop_on: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.drop_on = drop_on

    def splits(self):
        return {"evolve": ["c1", "c2", "c3", "c4"]}

    def failures(self, result):
        return []

    async def evaluate(self, harness_dir, split, k, *, experiment_id, variant):
        mode, sink = os.environ[GUARDS_ENV], Path(os.environ[DECISIONS_ENV])
        self.calls.append((mode, variant, sink.name))
        on = mode == "enforce"
        files: dict[str, list[dict]] = {}
        scores = []
        for t in range(k):
            if on:
                files[f"c1/{t}.jsonl"] = [dec("lsn.a.prior", "block", True, 2)]
                files[f"c2/{t}.jsonl"] = [dec("lsn.a.prior", "block", True, 2),
                                          dec("lsn.b.arg", "warn", True, 3, "escalate_to_human")]
                scores += [TaskScore("c1", t, "s", 1.0), TaskScore("c2", t, "s", 0.0),
                           TaskScore("c3", t, "s", 1.0), TaskScore("c4", t, "s", 0.5, V)]
                if self.drop_on:
                    scores.pop()
            else:
                files[f"c1/{t}.jsonl"] = [dec("lsn.a.prior", "block", False, 2)]
                scores += [TaskScore("c1", t, "s", 0.0, V), TaskScore("c2", t, "s", 1.0),
                           TaskScore("c3", t, "s", 1.0), TaskScore("c4", t, "s", 0.0, V)]
        files["opportunities.jsonl"] = [{"kind": "opportunity", "n": 10}]
        for rel, lines in files.items():
            p = sink / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        return EvalResult("tree", split, EvaluatorPin("tree", "judge", "fake"), scores)


def test_paired_metrics_definitions(tmp_path: Path):
    os.environ[GUARDS_ENV] = "shadow"
    os.environ.pop(DECISIONS_ENV, None)
    dom = FakeDomain()
    m = asyncio.run(paired_eval(dom, tmp_path, "evolve", 1, experiment_id="e", variant="arm-g",
                                decisions_dir=tmp_path / "dec", stochastic=False))
    assert [c[:2] for c in dom.calls] == [("off", "arm-g-off-0"), ("enforce", "arm-g-on-0")]
    assert os.environ[GUARDS_ENV] == "shadow" and DECISIONS_ENV not in os.environ  # env restored
    assert m.paired and m.trials == 1
    assert m.attempted_violation_rate == 0.75     # c1 (off oracle), c2 (on attempt), c4 (oracle)
    assert m.delivered_violation_rate == 0.25     # c4
    assert m.task_completion == pytest.approx(0.625)
    assert m.false_denial_rate == 0.5             # c2 blocked though off run was clean
    assert m.block_rate == 0.5 and m.fires == 3 and m.substitutions == 1
    assert m.recall == 0.5 and m.opportunities == 10
    assert m.fp_ucb is not None and m.fp_ucb > m.fp_rate
    assert m.bundle_digest == bundle_digest([])
    assert agent_safety_rate(m) == m.attempted_violation_rate  # guards never credited to the agent
    os.environ.pop(GUARDS_ENV)


def test_paired_requires_trials_and_identical_cases(tmp_path: Path):
    with pytest.raises(ValueError, match=">= 3 trials"):
        asyncio.run(paired_eval(FakeDomain(), tmp_path, "evolve", 1, trials=2, experiment_id="e", variant="v",
                                decisions_dir=tmp_path / "d"))
    m = asyncio.run(paired_eval(FakeDomain(), tmp_path, "evolve", 1, trials=3, experiment_id="e", variant="v",
                                decisions_dir=tmp_path / "d3"))
    assert m.trials == 3 and m.opportunities == 30 and m.delivered_violation_rate == 0.25
    with pytest.raises(PairingError):
        asyncio.run(paired_eval(FakeDomain(drop_on=True), tmp_path, "evolve", 3, experiment_id="e", variant="v",
                                decisions_dir=tmp_path / "d4"))
    assert GUARDS_ENV not in os.environ


def gm(delivered: float, completion: float, fd: float = 0.0, paired: bool = True) -> GuardMetrics:
    return GuardMetrics(bundle_digest="sha256:x", paired=paired, trials=3, attempted_violation_rate=0.5,
                        delivered_violation_rate=delivered, task_completion=completion, false_denial_rate=fd,
                        block_rate=0.1)


def test_guard_ship_rule():
    inc = gm(0.2, 0.8)
    assert guard_ship_ok(gm(0.1, 0.79), inc, 0.02) == (True, [])
    ok, why = guard_ship_ok(gm(0.2, 0.9), inc, 0.02)
    assert not ok and "delivered" in why[0]
    ok, why = guard_ship_ok(gm(0.1, 0.7), inc, 0.02)
    assert not ok and "task_completion" in why[0]
    ok, why = guard_ship_ok(gm(0.1, 0.8, fd=0.05), inc, 0.02)
    assert not ok and "false_denial" in why[0]
    assert not guard_ship_ok(gm(0.1, 0.8, paired=False), inc, 0.02)[0]


def test_workflow_step_is_idempotent_and_skips(tmp_path: Path):
    run = tmp_path / "arm"
    dom = FakeDomain()
    kw = {"domain": dom, "worktree": tmp_path, "split": "evolve", "k": 3, "experiment_id": "e", "variant": "arm"}
    out = asyncio.run(guard_paired_eval_step(run_dir=run, incumbent=gm(0.5, 0.6), margin=0.05, **kw))
    assert out["skipped"] is False and out["ship"]["ok"] is False  # false denials 0.5 > eps
    assert asyncio.run(guard_paired_eval_step(run_dir=run, **kw)) == out and len(dom.calls) == 2
    run2 = tmp_path / "arm2"
    run2.mkdir()
    (run2 / "eval.json").write_text(json.dumps({"skipped": True}), encoding="utf-8")
    assert asyncio.run(guard_paired_eval_step(run_dir=run2, **kw))["skipped"] is True


def test_arm_guard_workflow_is_expression_free_and_ordered():
    text = WORKFLOW.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    assert doc["kind"] == "Workflow" and doc["trigger"]["id"] == "rrsi_arm_guard"
    actions = doc["trigger"]["actions"]
    assert [a["id"] for a in actions] == ["provision_slot", "propose", "critique_1", "repair_1", "critique_2",
                                          "repair_2", "critique_final", "evaluate", "guard_paired_eval",
                                          "finalize_arm"]
    assert actions[1]["arguments"] == {"strategy": "guard"}
    for a in actions:
        assert a["kind"] == "InvokeFunctionTool"
        for v in (a.get("arguments") or {}).values():
            assert isinstance(v, (str, int, float, bool)) and not str(v).lstrip().startswith("=")
    for line in text.splitlines():
        body = line.split("#", 1)[0]
        if ":" in body:
            assert not body.split(":", 1)[1].strip().strip("'\"").startswith("=")
    try:
        from ci_lab.maf.workflows import assert_expression_free  # type: ignore[import-not-found]
    except ImportError:
        return
    assert_expression_free(text)
