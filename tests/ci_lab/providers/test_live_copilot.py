"""Live Copilot round-trip. Run explicitly: ``pytest -m copilot tests/ci_lab/providers``."""

import asyncio

import pytest
from agent_framework import Agent, tool

from ci_lab.providers.copilot import CopilotChatClient, copilot_scope

pytestmark = pytest.mark.copilot


def test_live_tool_round_trip():
    calls: list[str] = []
    records: list[dict] = []

    @tool(approval_mode="never_require")
    def lookup_order(order_id: str) -> str:
        """Look up the status code of an order by id."""
        calls.append(order_id)
        return f"order {order_id} status code: ZEBRA-{len(order_id)}7"

    async def main():
        async with CopilotChatClient(model="gpt-5-mini", timeout_s=180, on_model_request=records.append) as client:
            agent = Agent(client=client, instructions="Use tools to answer. Reply with the exact status code.",
                          tools=[lookup_order])
            with copilot_scope("live-test"):
                return await agent.run("What is the status code of order A1? Call lookup_order.")

    result = asyncio.run(main())
    assert calls == ["A1"]
    assert "ZEBRA-27" in result.text
    assert result.usage_details and result.usage_details.get("input_token_count", 0) > 0
    assert records and all(r["status"] == "ok" for r in records)
    assert any(r["response"]["tool_calls"] for r in records)
    assert all(r["model"] for r in records)
