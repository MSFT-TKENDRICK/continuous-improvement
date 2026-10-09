import asyncio

import pytest

from ci_lab import cli
from ci_lab.contracts import (
    ARM_BRANCH_RE,
    STRATEGIES,
    RolloutKey,
    arm_branch,
    op_id,
    round_experiment_id,
    strategy_may_edit,
)
from ci_lab.testing import Call, FakeChatClient, MemoryJournal, MemoryOutbox


def test_ids_are_validated_and_deterministic():
    exp = round_experiment_id("tone-a1", 3)
    assert exp == "tone-a1-r03"
    assert arm_branch(exp, "v1") == "exp/tone-a1-r03/v1"
    assert ARM_BRANCH_RE.match(arm_branch(exp, "inc"))
    for bad in ("../x", "V1", "a" * 17, "v1/x"):
        with pytest.raises(ValueError):
            arm_branch(exp, bad)
    with pytest.raises(ValueError):
        round_experiment_id("Bad_Campaign", 1)
    assert op_id("a", 1) == op_id("a", 1) != op_id("a", 2)
    key = RolloutKey("e", "v1", "c01", 0)
    assert key.rollout_id == RolloutKey("e", "v1", "c01", 0, attempt=1).rollout_id
    assert key.rollout_id != RolloutKey("e", "v1", "c01", 1).rollout_id


def test_agl_contract_owns_only_structural_components():
    assert "agl" in STRATEGIES
    assert all(strategy_may_edit("agl", c) for c in (
        "agent", "loop", "workflow", "mcp", "client_tool", "config", "context_mgmt", "memory"))
    assert not any(strategy_may_edit("agl", c) for c in ("prompt", "skill", "guard"))


def test_fake_client_drives_a_real_maf_tool_loop():
    from agent_framework import Agent

    seen = []

    def lookup(path: str) -> str:
        """Read a path."""
        seen.append(path)
        return "found"

    client = FakeChatClient([[Call("lookup", {"path": "file-1"})], "Found it."])
    agent = Agent(client=client, instructions="sys", tools=[lookup])
    result = asyncio.run(agent.run("where is CASE-1?"))
    assert seen == ["file-1"]
    assert result.text == "Found it."
    last_msgs, _ = client.requests[-1]
    assert any(c.type == "function_result" for m in last_msgs for c in m.contents)


def test_memory_outbox_and_journal_dedupe():
    box = MemoryOutbox()
    assert box.run_once("op", lambda: 1) == 1
    assert box.run_once("op", lambda: 2) == 1
    assert box.run_once("op2", lambda: 3, reconcile=lambda: "found") == "found"
    assert box.calls == ["op"]
    j = MemoryJournal()
    k = RolloutKey("e", "v", "c")
    j.start(k, {"x": 1})
    j.event(k, "reward", {"value": 1.0}, event_id="a")
    j.event(k, "reward", {"value": 9.0}, event_id="a")
    assert [e["data"]["value"] for e in j.events(k)] == [1.0]


def test_no_dotnet_bridge_installed():
    report = cli.doctor()
    assert report["no_dotnet"], report["dotnet_bridges"]


def test_cli_doctor_runs():
    assert cli.main(["doctor"]) == 0
