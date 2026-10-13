from __future__ import annotations

import asyncio
import json
import re
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from agent_framework import Agent, tool

from ci_lab.bus.types import StudentCorrection
from ci_lab.contracts import (
    ArmContext,
    ArmDirective,
    CriticVerdict,
    FailureRecord,
    Profile,
)
from ci_lab.meta.brief import (
    SanitizeResultsMiddleware,
    arm_attempt,
    failure_section,
    student_brief_tools,
)
from ci_lab.meta.run import default_arm_brief, write_brief, write_failures
from ci_lab.meta.spec_loader import default_builder, load_spec
from ci_lab.testing import Call, FakeChatClient
from ci_lab.workflows.steps import arm_tools

FAILURE = FailureRecord(case_id="change-abc", suite="harness_agent_harness_policy", category="change_no_id",
                        rule_ids=("judge.policy_violation",), rubric_scores={"policy_violation": 0.0},
                        excerpt="Change issued without verifying access to the resource.")
FORBIDDEN = ("change-abc", "harness_agent_harness_policy", "policy_violation")


def assert_clean(text: str) -> None:
    assert not [f for f in FORBIDDEN if f in text], text
    assert not re.search(r"\d", text), text


def correction(text: str) -> StudentCorrection:
    data = json.loads(text)
    return StudentCorrection(text=data["text"], attempt=data["attempt"])


def test_student_spec_has_no_rubric_tools_and_builds():
    spec = load_spec("student")
    assert spec.name == "CiStudent" and "submit_output" in spec.tools
    assert not [t for t in spec.tools if re.search(r"brief|history|document|rubric|vault|eval", t)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        default_builder()(spec, client=FakeChatClient(), bindings={t: (lambda: "x") for t in spec.tools},
                          loop_should_continue=lambda **_: False, loop_next_message=lambda **_: "x")


def test_arm_attempt():
    assert arm_attempt("Arm A/1", 2) == "arm-a-1@2" and arm_attempt("", 0) == "arm@1"


def test_proposer_brief_hides_suites_ids_and_scores(tmp_path):
    ctx = ArmContext(experiment_id="exp", directive=ArmDirective(arm="a1", edit_budget=1), worktree=tmp_path,
                     base_commit="abc", failures=(FAILURE,), profile=Profile.FAKE, run_dir=tmp_path)
    brief = default_arm_brief(ctx, ["prompt"])
    assert "Components you may edit: prompt" in brief and "Edit budget: at most 1" in brief
    section = brief.split("## Failures to address (1 record(s))\n\n")[1]
    assert "change_no_id" in section and "without verifying access to the resource" in section
    assert_clean(section)
    assert failure_section((), attempt="a1@1") == ""


@pytest.mark.parametrize("name,payload", [
    ("critique", {"passed": False, "reasons": ["change-abc still fails policy_violation at 0.4 (threshold 0.8)",
                                               "edit does not mention identity verification before changes"],
                  "source": "critic", "base": "abc123", "head": "def456"}),
    ("analysis", {"summary": "harness_agent_harness_policy fails 3/5",
                  "patterns": [{"name": "changes skip identity check", "case_ids": ["change-abc"], "count": 4}]}),
])
def test_student_documents_are_sanitized_corrections(tmp_path, name, payload):
    (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    write_brief(tmp_path, "Brief for change-abc")
    write_failures(tmp_path, [FAILURE])
    tools = student_brief_tools(tmp_path, [FAILURE], attempt="a1@1", allowed=load_spec("proposer").documents)
    text = correction(tools["read_brief"](name)).text
    assert_clean(text)
    assert "identity" in text
    assert "change-abc" not in tools["read_brief"]("brief")  # redacted, not converted
    assert tools["read_brief"]("failures").startswith("ERROR")  # not a proposer document


def test_history_is_sanitized(tmp_path):
    tools = student_brief_tools(tmp_path, [FAILURE], attempt="a1@1")
    assert not tools["read_history"]().startswith("{")  # notice passes through
    (tmp_path / "history.jsonl").write_text(
        json.dumps({"round": 1, "summary": "tried policy_violation fix on change-abc, score 0.2"}) + "\n",
        encoding="utf-8")
    for text in (tools["read_history"](), tools["read_brief"]("history")):
        assert_clean(correction(text).text)


def test_middleware_sanitizes_background_results():
    answer = "ROOT CAUSE: change-abc fails policy_violation (0.0) because changes skip identity checks."

    @tool(name="background_agents_get_task_results")
    def results(task_id: int) -> str:
        return answer

    @tool(name="echo")
    def echo() -> str:
        return answer

    client = FakeChatClient([[Call("background_agents_get_task_results", {"task_id": 1})], [Call("echo", {})],
                             "done"])
    agent = Agent(client=client, tools=[results, echo], middleware=[SanitizeResultsMiddleware([FAILURE],
                                                                                              attempt="a1@1")])
    asyncio.run(agent.run("go"))
    out = [str(getattr(c, "result", "") or "") for m in client.requests[-1][0] for c in m.contents]
    sanitized = [o for o in out if o.startswith("{")]
    assert len(sanitized) == 1 and "identity" in correction(sanitized[0]).text
    assert_clean(correction(sanitized[0]).text)
    assert answer in out  # other tools untouched


def test_repair_message_is_sanitized(tmp_path):
    sent: list[tuple[str, list[str]]] = []

    async def reinvoke(message, reasons):
        sent.append((message, reasons))

    reasons = ["change-abc fails policy_violation at 0.2", "the edit never asks for identity verification"]
    ctx = SimpleNamespace(dir=tmp_path, arm="a1", reinvoke_proposer=reinvoke, tracker=None,
                          round=SimpleNamespace(env=SimpleNamespace(deps=None, hyper={"bus": False}),
                                                begin=lambda: {"failures": [asdict(FAILURE)]}),
                          verdict=lambda n: CriticVerdict(passed=False, reasons=reasons, repairs=0))
    asyncio.run(arm_tools(ctx)["repair"](attempt=1))
    message, raw = sent[0]
    assert raw == reasons and "identity verification" in message
    assert "change-abc" not in message and "0.2" not in message and "submit_proposal again" in message
