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
    def read_file(path: str) -> str:
        """Look up the status code of a file by path."""
        calls.append(path)
        return f"file {path} status code: ZEBRA-{len(path)}7"

    async def main():
        async with CopilotChatClient(model="gpt-5-mini", timeout_s=180, on_model_request=records.append) as client:
            agent = Agent(client=client, instructions="Use tools to answer. Reply with the exact status code.",
                          tools=[read_file])
            with copilot_scope("live-test"):
                return await agent.run("What is the status code of file A1? Call read_file.")

    result = asyncio.run(main())
    assert calls == ["A1"]
    assert "ZEBRA-27" in result.text
    assert result.usage_details and result.usage_details.get("input_token_count", 0) > 0
    assert records and all(r["status"] == "ok" for r in records)
    assert any(r["response"]["tool_calls"] for r in records)
    assert all(r["model"] for r in records)
