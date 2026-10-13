from __future__ import annotations

import importlib
import importlib.util

import pytest


def test_powerfx_not_importable() -> None:
    assert importlib.util.find_spec("powerfx") is None
    with pytest.raises(ImportError):
        importlib.import_module("powerfx")


def test_declarative_imports_without_engine() -> None:
    import agent_framework_declarative
    from agent_framework_declarative import _models

    assert agent_framework_declarative.AgentFactory is not None
    assert agent_framework_declarative.WorkflowFactory is not None
    assert _models._get_engine() is None
