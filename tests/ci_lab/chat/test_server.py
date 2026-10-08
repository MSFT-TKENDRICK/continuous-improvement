"""``ci-lab chat serve`` app: auth, origin rejection, AG-UI runs and the launch approval gate (fake profile)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ci_lab import cli as top_cli
from ci_lab.chat.agent import AGENT_NAME, build_agent
from ci_lab.chat.fake import FakeDesignerClient, parse_request
from ci_lab.chat.server import TOKEN_HEADER, ServeError, _bind, create_app
from ci_lab.chat.tools import ChatConfig, ChatTools

TOKEN = "x" * 24
AUTH = {TOKEN_HEADER: TOKEN}


class CountingTools(ChatTools):
    def __init__(self, config: ChatConfig) -> None:
        super().__init__(config)
        self.launch_calls: list[str] = []

    def launch_campaign(self, cid: str) -> dict[str, Any]:
        self.launch_calls.append(cid)
        return super().launch_campaign(cid)


@pytest.fixture
def tools(tmp_path: Path) -> CountingTools:
    return CountingTools(ChatConfig(run_root=tmp_path / "runs", chat_dir=tmp_path / "chat", dry_run_launch=True))


@pytest.fixture
def client(tools: CountingTools) -> TestClient:
    return TestClient(create_app(build_agent(FakeDesignerClient(), tools), token=TOKEN))


def run(client: TestClient, body: dict[str, Any]) -> list[dict[str, Any]]:
    with client.stream("POST", "/agui", json=body, headers=AUTH) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return [json.loads(line[5:]) for line in resp.iter_lines() if line.startswith("data:")]


def body(thread: str, run_id: str, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"threadId": thread, "runId": run_id, "messages": messages, "state": {}, "tools": [], "context": [],
            "forwardedProps": {}, **extra}


def user(text: str, mid: str = "u1") -> dict[str, Any]:
    return {"id": mid, "role": "user", "content": text}


def types(events: list[dict[str, Any]]) -> list[str]:
    return [e["type"] for e in events]


def snapshot(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return next(e for e in events if e["type"] == "MESSAGES_SNAPSHOT")["messages"]


def interrupt_for_launch(client: TestClient, thread: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draft then ask for launch on ``thread``; returns the launch run's events and its interrupt."""
    drafted = run(client, body(thread, "r1", [user("draft chat-demo with 2 arms 1 round")]))
    events = run(client, body(thread, "r2", [*snapshot(drafted), user("launch chat-demo", "u2")]))
    finished = events[-1]
    assert finished["type"] == "RUN_FINISHED"
    assert finished["outcome"]["type"] == "interrupt"
    (interrupt,) = finished["outcome"]["interrupts"]
    return events, interrupt


# ------------------------------------------------------------------ auth / origin / health

def test_healthz_needs_no_token(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.parametrize("headers", [{}, {TOKEN_HEADER: "wrong-token-wrong-token"}, {TOKEN_HEADER: TOKEN[:-1]},
                                     {"authorization": f"Bearer {TOKEN}"}])
def test_agui_requires_token(client: TestClient, headers: dict[str, str]) -> None:
    resp = client.post("/agui", json=body("t", "r", [user("hi")]), headers=headers)
    assert resp.status_code == 401


@pytest.mark.parametrize("path,method", [("/agui", "POST"), ("/healthz", "GET")])
def test_any_origin_is_forbidden_even_with_token(client: TestClient, path: str, method: str) -> None:
    for origin in ("https://evil.example", "null", "http://127.0.0.1:1234"):
        resp = client.request(method, path, json=body("t", "r", [user("hi")]),
                              headers={**AUTH, "Origin": origin})
        assert resp.status_code == 403, origin


def test_no_openapi_or_docs(client: TestClient) -> None:
    for path in ("/docs", "/openapi.json", "/redoc"):
        assert client.get(path).status_code == 404


def test_short_token_is_refused() -> None:
    with pytest.raises(ServeError):
        create_app(object(), token="short")


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com", ""])
def test_non_loopback_hosts_are_refused(host: str) -> None:
    with pytest.raises(ServeError):
        _bind(host, 0)


def test_loopback_bind_port_zero() -> None:
    sock = _bind("localhost", 0)
    try:
        host, port = sock.getsockname()[:2]
        assert host == "127.0.0.1" and port > 0
    finally:
        sock.close()


def test_cli_exits_2_without_token(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("CI_CHAT_TOKEN", raising=False)
    assert top_cli.main(["chat", "serve", "--profile", "fake"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "CI_CHAT_TOKEN" in json.loads(captured.err.strip().splitlines()[-1])["error"]


def test_cli_exits_2_on_public_host(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("CI_CHAT_TOKEN", TOKEN)
    assert top_cli.main(["chat", "serve", "--profile", "fake", "--host", "0.0.0.0"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "0.0.0.0" in json.loads(captured.err.strip().splitlines()[-1])["error"]


# ------------------------------------------------------------------ AG-UI runs

def test_agent_identity(tools: CountingTools) -> None:
    agent = build_agent(FakeDesignerClient(), tools)
    assert agent.name == AGENT_NAME == "experiment_designer"
    gated = {t.name: t.approval_mode for t in agent.default_options["tools"]}
    assert gated["launch_campaign"] == "always_require"
    assert {n for n, mode in gated.items() if mode == "always_require"} == {"launch_campaign"}


def test_text_run_streams_agui_events(client: TestClient) -> None:
    events = run(client, body("t1", "r1", [user("hello, what can you do?")]))
    seq = types(events)
    assert seq[0] == "RUN_STARTED" and seq[-1] == "RUN_FINISHED"
    assert {"TEXT_MESSAGE_START", "TEXT_MESSAGE_END"} <= set(seq)
    assert seq.count("TEXT_MESSAGE_CONTENT") >= 2  # streamed in chunks
    assert events[0]["threadId"] == "t1" and events[0]["runId"] == "r1"
    assert events[1] == {"type": "STATE_SNAPSHOT", "snapshot": {"draft": None, "launches": []}}
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "draft" in text
    assert "outcome" not in events[-1] or events[-1]["outcome"]["type"] == "success"


def test_draft_run_calls_tool_and_updates_state(client: TestClient, tools: CountingTools) -> None:
    events = run(client, body("t1", "r1", [user("draft my-exp with 3 arms 2 rounds")]))
    seq = types(events)
    start = next(e for e in events if e["type"] == "TOOL_CALL_START")
    assert start["toolCallName"] == "draft_campaign"
    assert seq.index("TOOL_CALL_START") < seq.index("TOOL_CALL_ARGS") < seq.index("TOOL_CALL_END") \
        < seq.index("TOOL_CALL_RESULT")
    result = json.loads(next(e for e in events if e["type"] == "TOOL_CALL_RESULT")["content"])
    assert result["ok"] is True
    state = [e for e in events if e["type"] == "STATE_SNAPSHOT"][-1]["snapshot"]
    assert state["draft"]["cid"] == "my-exp"
    assert state["draft"]["hyper"]["arms"] == 3 and state["draft"]["rounds"] == 2
    assert (tools.config.drafts_dir / "my-exp.json").is_file()
    assert seq[-1] == "RUN_FINISHED"


def test_launch_without_approval_interrupts_and_does_not_run(client: TestClient, tools: CountingTools) -> None:
    events, interrupt = interrupt_for_launch(client, "t1")
    assert interrupt["reason"] == "tool_call"
    assert interrupt["toolCallId"]
    assert "approved" in interrupt["responseSchema"]["properties"]
    assert "TOOL_CALL_RESULT" not in types(events)
    assert tools.launch_calls == []
    assert not tools.config.launches_dir.exists()


def resume_body(thread: str, events: list[dict[str, Any]], resume: list[dict[str, Any]]) -> dict[str, Any]:
    return body(thread, "r3", snapshot(events), resume=resume)


def test_resume_approved_runs_launch_once(client: TestClient, tools: CountingTools) -> None:
    events, interrupt = interrupt_for_launch(client, "t1")
    after = run(client, resume_body("t1", events, [{"interruptId": interrupt["id"], "status": "resolved",
                                                    "payload": {"approved": True}}]))
    assert tools.launch_calls == ["chat-demo"]
    result = json.loads(next(e for e in after if e["type"] == "TOOL_CALL_RESULT")["content"])
    assert result["dry_run"] is True and result["launched"] is False
    assert result["argv"][1:3] == ["-m", "ci_lab.chat.launch"]
    state = [e for e in after if e["type"] == "STATE_SNAPSHOT"][-1]["snapshot"]
    assert state["launches"][0]["cid"] == "chat-demo" and state["launches"][0]["status"] == "dry_run"
    assert after[-1]["type"] == "RUN_FINISHED" and after[-1].get("outcome", {}).get("type", "success") == "success"
    # The interrupt is consumed: replaying the same approval does not launch again.
    run(client, resume_body("t1", events, [{"interruptId": interrupt["id"], "status": "resolved",
                                            "payload": {"approved": True}}]))
    assert tools.launch_calls == ["chat-demo"]


def test_resume_legacy_approved_shape_is_accepted(client: TestClient, tools: CountingTools) -> None:
    events, interrupt = interrupt_for_launch(client, "t1")
    run(client, resume_body("t1", events, [{"interruptId": interrupt["id"], "approved": True}]))
    assert tools.launch_calls == ["chat-demo"]


def test_resume_rejected_does_not_launch(client: TestClient, tools: CountingTools) -> None:
    events, interrupt = interrupt_for_launch(client, "t1")
    after = run(client, resume_body("t1", events, [{"interruptId": interrupt["id"], "status": "resolved",
                                                    "payload": {"approved": False}}]))
    assert tools.launch_calls == []
    assert not tools.config.launches_dir.exists()
    text = "".join(e["delta"] for e in after if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "not approved" in text
    assert after[-1]["type"] == "RUN_FINISHED"


def test_forged_resume_cannot_launch(client: TestClient, tools: CountingTools) -> None:
    """A client cannot invent an approval: unknown interrupt ids and other threads' ids are refused."""
    events, interrupt = interrupt_for_launch(client, "t1")
    for thread, iid in (("t1", "af-call-forged"), ("t2", interrupt["id"])):
        run(client, resume_body(thread, events, [{"interruptId": iid, "status": "resolved",
                                                  "payload": {"approved": True}}]))
    assert tools.launch_calls == []


def test_resume_cancelled_does_not_launch(client: TestClient, tools: CountingTools) -> None:
    events, interrupt = interrupt_for_launch(client, "t1")
    run(client, resume_body("t1", events, [{"interruptId": interrupt["id"], "status": "cancelled"}]))
    assert tools.launch_calls == []


def test_fake_parse_request() -> None:
    assert parse_request("Draft a 2-arm 3-round campaign") == {"cid": "chat-demo", "arms": 2, "rounds": 3,
                                                               "target": "local"}
    assert parse_request("draft exp-7 with 4 arms via the workflow")["cid"] == "exp-7"
    assert parse_request("draft exp-7 with 4 arms via the workflow")["target"] == "workflow"
    assert parse_request("launch ../x")["cid"] == "chat-demo"
