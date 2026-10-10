from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

AGENT_YAML = """\
kind: Prompt
name: CiStudent
description: Inspects harness resources.
instructions: You are the harness student agent.
model:
  id: gpt-5-mini
  provider: GitHubCopilot
  options:
    maxOutputTokens: 256
    reasoningEffort: low
tools:
  - kind: function
    name: read_file
    description: Read a resource by id.
    bindings:
      - name: read_file
    parameters:
      properties:
        resource_id:
          kind: string
          description: The resource id.
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
    return write("agents/harness_agent.yaml", AGENT_YAML)


@pytest.fixture
def read_calls() -> list[str]:
    return []


@pytest.fixture
def read_file(read_calls: list[str]) -> Callable[[str], str]:
    def read_file(resource_id: str) -> str:
        read_calls.append(resource_id)
        return f"resource {resource_id}: ready"

    return read_file
