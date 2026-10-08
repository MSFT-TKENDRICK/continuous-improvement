"""Shared fixtures for the MAF order-support agent tests (offline: FakeChatClient only)."""

from __future__ import annotations

import pytest
from assert_ai.core.otel import LiveOTelExporter

from ci_lab.testing import FakeChatClient
from order_support import agent


@pytest.fixture
def captured():
    """Spans ASSERT's own live collector sees; call the result to export them as OTelSpans."""
    exporter = LiveOTelExporter()
    exporter.setup()
    exporter.clear()
    yield lambda: exporter.export_session("test")
    exporter.clear()


@pytest.fixture
def use_client(monkeypatch):
    """Pin a FakeChatClient (or any client) as the agent's client for one test."""
    for name in (agent.TIMEOUT_ENV, agent.EVALS_TIMEOUT_ENV, "ORDER_AGENT_TEMPERATURE", agent.HARNESS_ENV):
        monkeypatch.delenv(name, raising=False)

    def use(*script, client=None, **kwargs):
        client = client if client is not None else FakeChatClient(script=list(script), **kwargs)
        agent.set_client_override(client)
        return client

    yield use
    agent.set_client_override(None)
