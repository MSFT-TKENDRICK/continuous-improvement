"""Recorded AG-UI approval exchange (``tests/fixtures/chat/approval_flow.json``) and its drift check.

The fixture is the contract the canvas (layer 33) builds against: real request bodies and the
parsed SSE events of ``ci-lab chat serve --profile fake --dry-run-launch`` for a draft run, the
launch run that ends in an approval interrupt, and the resume runs (approved, rejected, and the
legacy ``{interruptId, approved}`` shape). Volatile values (ids, paths, timestamps) are replaced
by stable placeholders.

Regenerate after an intentional protocol change::

    uv run --no-sync python tests/ci_lab/chat/test_approval_fixture.py
"""

from __future__ import annotations

import importlib.metadata as md
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "chat" / "approval_flow.json"
TOKEN = "fixture-token-0123456789"
HEADERS = {"content-type": "application/json", "accept": "text/event-stream", "x-ci-chat-token": "<CI_CHAT_TOKEN>"}


def _sse(client: Any, body: dict[str, Any]) -> list[dict[str, Any]]:
    with client.stream("POST", "/agui", json=body, headers={"x-ci-chat-token": TOKEN}) as resp:
        assert resp.status_code == 200, resp.status_code
        return [json.loads(line[5:]) for line in resp.iter_lines() if line.startswith("data:")]


def _messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return next(e for e in events if e["type"] == "MESSAGES_SNAPSHOT")["messages"]


def _state(events: list[dict[str, Any]]) -> dict[str, Any]:
    return [e for e in events if e["type"] == "STATE_SNAPSHOT"][-1]["snapshot"]


def _body(thread: str, run: str, messages: list[dict[str, Any]], state: dict[str, Any],
          **extra: Any) -> dict[str, Any]:
    return {"threadId": thread, "runId": run, "messages": messages, "state": state, "tools": [], "context": [],
            "forwardedProps": {}, **extra}


def record_raw(root: Path) -> dict[str, Any]:
    """Drive the real app (fake profile, dry-run launch) and return the un-normalized exchange."""
    from fastapi.testclient import TestClient

    from ci_lab.chat.agent import build_agent
    from ci_lab.chat.fake import FakeDesignerClient
    from ci_lab.chat.server import create_app
    from ci_lab.chat.tools import ChatConfig, ChatTools

    config = ChatConfig(run_root=root / "runs", chat_dir=root / "chat", dry_run_launch=True)
    client = TestClient(create_app(build_agent(FakeDesignerClient(), ChatTools(config)), token=TOKEN))
    flows: dict[str, Any] = {}

    def step(name: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        events = _sse(client, body)
        flows.setdefault("_order", []).append(name)
        flows[name] = {"request": {"method": "POST", "path": "/agui", "headers": HEADERS, "body": body},
                       "events": events}
        return events

    draft = step("1_draft", _body("thread-approve", "run-1", [
        {"id": "user-1", "role": "user", "content": "Please draft chat-demo: 2 arms, 1 round, local."}], {}))
    launch_messages = [*_messages(draft), {"id": "user-2", "role": "user", "content": "launch chat-demo"}]
    for thread, suffix, resume in (
        ("thread-approve", "approved", lambda iid: [{"interruptId": iid, "status": "resolved",
                                                     "payload": {"approved": True}}]),
        ("thread-reject", "rejected", lambda iid: [{"interruptId": iid, "status": "resolved",
                                                    "payload": {"approved": False}}]),
        ("thread-legacy", "approved_legacy", lambda iid: [{"interruptId": iid, "approved": True}]),
    ):
        interrupted = step(f"2_launch_interrupt_{suffix}",
                           _body(thread, "run-2", launch_messages, _state(draft)))
        iid = interrupted[-1]["outcome"]["interrupts"][0]["id"]
        step(f"3_resume_{suffix}", _body(thread, "run-3", _messages(interrupted), _state(interrupted),
                                         resume=resume(iid)))
    return flows


_ID_PATTERNS = (
    ("interrupt", re.compile(r"af-call-[0-9a-f]{32}")),
    ("uuid", re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")),
    ("call", re.compile(r"call_[0-9a-f]{12}")),
)


def normalize(raw: dict[str, Any], *, root: Path) -> dict[str, Any]:
    text = json.dumps(raw)
    for value, placeholder in ((sys.executable, "<python>"), (str(root), "<tmp>")):
        variants = [value]
        for _ in range(3):
            variants.append(json.dumps(variants[-1])[1:-1])
        for v in sorted(set(variants), key=len, reverse=True):
            text = text.replace(v, placeholder)
    text = re.sub(r"<tmp>((?:\\+[\w.-]+)+)", lambda m: "<tmp>" + re.sub(r"\\+", "/", m.group(1)), text)
    text = re.sub(r'(\\*"created\\*": )\d+(?:\.\d+)?', r"\g<1>0", text)
    seen: dict[str, str] = {}
    for kind, pattern in _ID_PATTERNS:
        def sub(m: re.Match[str], kind: str = kind) -> str:
            key = m.group(0)
            if key not in seen:
                seen[key] = f"<{kind}-{sum(1 for v in seen.values() if v.startswith(f'<{kind}-')) + 1}>"
            return seen[key]
        text = pattern.sub(sub, text)
    return json.loads(text)


def key_paths(obj: Any, prefix: str = "") -> set[str]:
    """Every dict key path in ``obj`` (``a.b[].c``); list elements are merged."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out.add(path)
            out |= key_paths(v, path)
    elif isinstance(obj, list):
        for item in obj:
            out |= key_paths(item, f"{prefix}[]")
    return out


def shape(flows: dict[str, Any]) -> dict[str, Any]:
    """Event types and field names per step (the drift-checked part of the fixture)."""
    return {name: {"request": sorted(key_paths(flows[name]["request"]["body"])),
                   "events": [[e["type"], sorted(key_paths(e))] for e in flows[name]["events"]]}
            for name in flows["_order"]}


def protocol(flows: dict[str, Any]) -> dict[str, Any]:
    finished = flows["2_launch_interrupt_approved"]["events"][-1]
    interrupt = finished["outcome"]["interrupts"][0]
    return {
        "interrupt_run_ends_with": {"type": finished["type"], "outcome.type": finished["outcome"]["type"]},
        "interrupt_fields": sorted(interrupt),
        "interrupt_tool": "metadata.agent_framework.function_call.{call_id,name,arguments}",
        "approval_custom_event": next(e["name"] for e in flows["2_launch_interrupt_approved"]["events"]
                                      if e["type"] == "CUSTOM"),
        "resume_canonical": flows["3_resume_approved"]["request"]["body"]["resume"],
        "resume_rejected": flows["3_resume_rejected"]["request"]["body"]["resume"],
        "resume_legacy": flows["3_resume_approved_legacy"]["request"]["body"]["resume"],
        "resume_rules": [
            "send a NEW run on the SAME threadId to the SAME server process (approval state is in memory)",
            "messages = the MESSAGES_SNAPSHOT of the interrupted run; state = its last STATE_SNAPSHOT",
            "payload must contain a boolean 'approved' (or legacy 'accepted'); status 'cancelled' rejects",
            "an interrupt id is single-use and bound to its thread; unknown ids never execute the tool",
        ],
    }


def build_fixture() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ci-chat-fixture-") as tmp:
        root = Path(tmp)
        flows = normalize(record_raw(root), root=root)
    return {
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "regenerate": "uv run --no-sync python tests/ci_lab/chat/test_approval_fixture.py",
        "server": "ci-lab chat serve --profile fake --dry-run-launch (in-process via fastapi TestClient)",
        "agent_framework_ag_ui": md.version("agent-framework-ag-ui"),
        "sse": "each event is one 'data: <json>' line followed by a blank line; ': keepalive' comments may appear",
        "protocol": protocol(flows),
        "order": flows["_order"],
        "steps": {name: flows[name] for name in flows["_order"]},
    }


def _flows(fixture: dict[str, Any]) -> dict[str, Any]:
    return {"_order": fixture["order"], **fixture["steps"]}


def test_fixture_matches_live_server() -> None:
    assert FIXTURE.is_file(), "missing fixture; regenerate: uv run --no-sync python tests/ci_lab/chat/test_approval_fixture.py"
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    live = build_fixture()
    assert recorded["order"] == live["order"]
    assert shape(_flows(recorded)) == shape(_flows(live)), (
        "AG-UI event types/field names drifted; review the change and regenerate: " + recorded["regenerate"])
    assert recorded["protocol"] == live["protocol"]


def test_fixture_documents_the_approval_contract() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    steps = fixture["steps"]
    interrupt_end = steps["2_launch_interrupt_approved"]["events"][-1]
    assert interrupt_end["outcome"]["type"] == "interrupt"
    (interrupt,) = interrupt_end["outcome"]["interrupts"]
    assert interrupt["reason"] == "tool_call"
    assert interrupt["metadata"]["agent_framework"]["function_call"]["name"] == "launch_campaign"
    assert {"TOOL_CALL_RESULT"} & {e["type"] for e in steps["3_resume_approved"]["events"]}
    assert {"TOOL_CALL_RESULT"} & {e["type"] for e in steps["3_resume_approved_legacy"]["events"]}
    assert "TOOL_CALL_RESULT" not in {e["type"] for e in steps["3_resume_rejected"]["events"]}
    assert TOKEN not in FIXTURE.read_text(encoding="utf-8")


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(build_fixture(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE}")
