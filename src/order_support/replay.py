"""Convert the labelled dataset into an ASSERT judge-only inference set.

Each case becomes one ``type: prompt`` inference row whose events reproduce the
observable trace: the agent policy as a system message, the conversation, the
tool calls (``tool_call`` edits), and the final assistant response. Labels,
tags and notes are never copied into the row, so the judge cannot see them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "evals" / "datasets" / "order_support.yaml"
REPLAY_DIR = REPO_ROOT / "evals" / "assert" / "judge_replay"
INFERENCE_SET_PATH = REPLAY_DIR / "inference_set.jsonl"

BEHAVIOR = "order_support_policy_compliance"
TARGET = "replay:evals/datasets/order_support.yaml"
TESTER = "human-authored"
_VIEWS = ["target", "combined"]
_ACTOR = {"user": "tester", "assistant": "target"}


def load_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = raw["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in dataset")
    return cases


def _message(role: str, content: str) -> dict[str, Any]:
    return {"view": list(_VIEWS), "actor": _ACTOR[role],
            "edit": {"type": "add_message", "message": {"role": role, "content": content}}}


def _tool(call: dict[str, Any]) -> dict[str, Any]:
    result = call.get("result")
    result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    return {"view": list(_VIEWS), "actor": "tool",
            "edit": {"type": "tool_call", "tool_name": call["name"],
                     "tool_args": dict(call.get("arguments") or {}), "tool_result": result_text}}


def case_to_row(case: dict[str, Any]) -> dict[str, Any]:
    obs = case["observable"]
    conversation = obs["conversation"]
    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError(f"{case['id']}: conversation must end with a user turn")
    last = len(conversation) - 1
    calls_after: dict[int, list[dict[str, Any]]] = {}
    for call in obs.get("tool_calls") or []:
        idx = int(call.get("after_message", last))
        if not 0 <= idx <= last or conversation[idx]["role"] != "user":
            raise ValueError(f"{case['id']}: after_message must point at a user turn")
        calls_after.setdefault(idx, []).append(call)

    events: list[dict[str, Any]] = [{
        "view": list(_VIEWS), "actor": "system",
        "edit": {"type": "set_system_message",
                 "message": {"role": "system", "content": str(obs["agent_policy"]).strip()}},
    }]
    for i, turn in enumerate(conversation):
        events.append(_message(turn["role"], turn["content"]))
        events.extend(_tool(c) for c in calls_after.get(i, []))
    events.append(_message("assistant", obs["final_response"]))
    return {
        "type": "prompt",
        "test_case_id": case["id"],
        "behavior": BEHAVIOR,
        "target": TARGET,
        "tester_model": TESTER,
        "stop_reason": "completed",
        "dimensions": {"scenario": case["tags"][0]},
        "events": events,
    }


def render(cases: list[dict[str, Any]] | None = None) -> str:
    rows = [case_to_row(c) for c in (cases if cases is not None else load_cases())]
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def build(path: Path = INFERENCE_SET_PATH) -> int:
    text = render()
    path.write_text(text, encoding="utf-8", newline="\n")
    return text.count("\n")


def is_current(path: Path = INFERENCE_SET_PATH) -> bool:
    return path.exists() and path.read_text(encoding="utf-8") == render()
