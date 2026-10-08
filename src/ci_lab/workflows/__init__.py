"""Declarative, expression-free MAF workflows for RRSI campaigns (design §5, C5).

The YAML files here contain only ``InvokeFunctionTool`` / ``InvokeAzureAgent``
actions with literal arguments. Dynamic context (run dir, experiment id, arm)
is bound into the registered tool closures by :mod:`ci_lab.workflows.steps`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

WORKFLOW_DIR = Path(__file__).resolve().parent
ROUND_YAML = WORKFLOW_DIR / "round.yaml"
ARM_AGENT_YAML = WORKFLOW_DIR / "arm_agent.yaml"
ARM_GEPA_YAML = WORKFLOW_DIR / "arm_gepa.yaml"
ARM_SKILLOPT_YAML = WORKFLOW_DIR / "arm_skillopt.yaml"
# v2.4 §13: owned by the lessons arm (path only; importing ci_lab.lessons_arm here is not needed).
ARM_GUARD_YAML = WORKFLOW_DIR.parent / "lessons_arm" / "workflows" / "arm_guard.yaml"
ARM_YAMLS = {"agent": ARM_AGENT_YAML, "gepa": ARM_GEPA_YAML, "skillopt": ARM_SKILLOPT_YAML,
             "guard": ARM_GUARD_YAML}  # by strategy
ARM_YAML = ARM_AGENT_YAML
CALIBRATE_YAML = WORKFLOW_DIR / "calibrate.yaml"
CONFIRM_YAML = WORKFLOW_DIR / "confirm.yaml"
WORKFLOW_FILES = {"round": ROUND_YAML, **{f"arm_{k}": v for k, v in ARM_YAMLS.items()},
                  "calibrate": CALIBRATE_YAML, "confirm": CONFIRM_YAML}

ALLOWED_ACTIONS = frozenset({"InvokeFunctionTool", "InvokeAzureAgent"})
FORBIDDEN_ACTIONS = frozenset({"If", "ConditionGroup", "Foreach", "GotoAction", "BreakLoop", "ContinueLoop"})


def _strings(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(k, f"{path}.<key>")
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _strings(v, f"{path}[{i}]")


def assert_expression_free(source: str | Path) -> dict[str, Any]:
    """Validate the no-PowerFx contract (I3) and return the parsed document.

    Rejects any string starting with ``=``, any control-flow action, any action
    kind outside :data:`ALLOWED_ACTIONS` and any non-literal (nested) argument."""
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or doc.get("kind") != "Workflow":
        raise ValueError("not a declarative Workflow document")
    for path, value in _strings(doc):
        if value.lstrip().startswith("="):
            raise ValueError(f"PowerFx expression at {path}: {value!r}")
    trigger = doc.get("trigger") or {}
    if trigger.get("kind") != "OnConversationStart":
        raise ValueError("trigger must be OnConversationStart")
    actions = trigger.get("actions") or []
    if not actions:
        raise ValueError("workflow has no actions")
    ids: set[str] = set()
    for action in actions:
        kind = action.get("kind")
        if kind in FORBIDDEN_ACTIONS or kind not in ALLOWED_ACTIONS:
            raise ValueError(f"action kind {kind!r} not allowed")
        if not action.get("id") or action["id"] in ids:
            raise ValueError(f"missing or duplicate action id {action.get('id')!r}")
        ids.add(action["id"])
        if kind == "InvokeFunctionTool":
            for key, val in (action.get("arguments") or {}).items():
                if isinstance(val, (dict, list)):
                    raise ValueError(f"{action['id']}.{key}: arguments must be literal scalars")
        else:
            msg = (action.get("input") or {}).get("messages")
            if not isinstance(msg, str) or not (action.get("agent") or {}).get("name"):
                raise ValueError(f"{action['id']}: agent actions need agent.name and a literal message")
    return doc


def function_names(source: str | Path) -> set[str]:
    doc = assert_expression_free(source)
    return {a["functionName"] for a in doc["trigger"]["actions"] if a["kind"] == "InvokeFunctionTool"}


def agent_names(source: str | Path) -> set[str]:
    doc = assert_expression_free(source)
    return {a["agent"]["name"] for a in doc["trigger"]["actions"] if a["kind"] == "InvokeAzureAgent"}
