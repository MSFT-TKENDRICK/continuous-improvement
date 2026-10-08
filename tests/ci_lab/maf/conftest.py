from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

AGENT_YAML = """\
kind: Prompt
name: OrderSupport
description: Answers order questions.
instructions: You are the order support agent.
model:
  id: gpt-5-mini
  provider: GitHubCopilot
  options:
    maxOutputTokens: 256
    reasoningEffort: low
tools:
  - kind: function
    name: lookup_order
    description: Look up an order by id.
    bindings:
      - name: lookup_order
    parameters:
      properties:
        order_id:
          kind: string
          description: The order id.
          required: true
"""


@pytest.fixture
def agent_text() -> str:
    return AGENT_YAML


@pytest.fixture
def write(tmp_path: Path) -> Callable[[str, str], Path]:
    def _write(rel: str, text: str) -> Path:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text), encoding="utf-8")
        return p

    return _write


@pytest.fixture
def agent_yaml(write: Callable[[str, str], Path]) -> Path:
    return write("agents/order_support.yaml", AGENT_YAML)


@pytest.fixture
def lookup_calls() -> list[str]:
    return []


@pytest.fixture
def lookup_order(lookup_calls: list[str]) -> Callable[[str], str]:
    def lookup_order(order_id: str) -> str:
        lookup_calls.append(order_id)
        return f"order {order_id}: shipped"

    return lookup_order
