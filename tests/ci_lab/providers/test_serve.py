import asyncio
import json
import os
import socket
import threading
import time
from contextlib import contextmanager

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from fastapi.testclient import TestClient  # noqa: E402

from ci_lab import cli, obs  # noqa: E402
from ci_lab.providers.copilot import CopilotChatClient  # noqa: E402
from ci_lab.providers.fake_sdk import FakeCopilotClient, Hang  # noqa: E402
from ci_lab.providers.serve import check_loopback, create_app, write_key_file  # noqa: E402

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def make_app(script=("hello there",), **kw):
    sdk = FakeCopilotClient(script if callable(script) else list(script))
    client = CopilotChatClient(model="gpt-5-mini", sdk_client=sdk, **kw)
    return sdk, create_app(client, api_key=KEY, close_client=True)


def chat(tc, **body):
    body.setdefault("model", "gpt-5-mini")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return tc.post("/v1/chat/completions", json=body, headers=AUTH)


def test_auth_is_required():
    _, app = make_app()
    with TestClient(app) as tc:
        assert tc.get("/v1/models").status_code == 401
        r = tc.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
        models = tc.get("/v1/models", headers=AUTH).json()
        assert models["data"][0]["id"] == "gpt-5-mini"


def test_chat_completion_openai_shape():
    sdk, app = make_app()
    with TestClient(app) as tc:
        r = chat(tc, messages=[{"role": "developer", "content": "Be brief."},
                               {"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["model"] == "gpt-fake-served"
    assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "hello there"},
                                "finish_reason": "stop", "logprobs": None}]
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert "x-ci-ignored-params" not in r.headers
    s = sdk.sessions[0]
    assert s.kwargs["system_message"]["content"] == "Be brief."
    assert s.kwargs["tools"] == []
    assert s.prompts == ["hi"]


def test_multi_turn_messages_are_replayed_as_transcript():
    sdk, app = make_app()
    with TestClient(app) as tc:
        r = chat(tc, messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                               {"role": "user", "content": "c"}])
    assert r.status_code == 200
    assert "[assistant]\nb" in sdk.sessions[0].prompts[0]


def test_sampling_params_are_ignored_and_reported():
    _, app = make_app()
    with TestClient(app) as tc:
        r = chat(tc, temperature=0, seed=3, top_p=0.5, max_tokens=10)
    assert r.status_code == 200
    assert r.headers["x-ci-ignored-params"] == "max_tokens,seed,temperature,top_p"


@pytest.mark.parametrize("extra,param", [
    ({"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}, "tools"),
    ({"stream": True}, "stream"),
    ({"n": 2}, "n"),
])
def test_unsupported_features_get_openai_400(extra, param):
    sdk, app = make_app()
    with TestClient(app) as tc:
        r = chat(tc, **extra)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == param and err["message"]
    assert sdk.sessions == []


def test_bad_requests():
    _, app = make_app()
    with TestClient(app) as tc:
        assert chat(tc, messages=[{"role": "tool", "content": "x"}]).status_code == 400
        assert chat(tc, messages=[]).status_code == 400
        assert chat(tc, model="other").status_code == 404
        assert tc.post("/v1/chat/completions", content=b"nope", headers=AUTH).status_code == 400


def test_response_format_json_schema_maps_to_structured_output():
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    sdk, app = make_app(['```json\n{"score": 3}\n```'])
    with TestClient(app) as tc:
        r = chat(tc, response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": schema}})
    assert r.status_code == 200
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"score": 3}
    assert '"score"' in sdk.sessions[0].kwargs["system_message"]["content"]

    sdk, app = make_app(['{"a": 1}'])
    with TestClient(app) as tc:
        r = chat(tc, response_format={"type": "json_object"})
    assert r.json()["choices"][0]["message"]["content"] == '{"a": 1}'


def test_timeout_maps_to_504():
    sdk, app = make_app([Hang()], timeout_s=0.05)
    with TestClient(app) as tc:
        r = chat(tc)
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"
    assert sdk.sessions[0].aborted


def test_loopback_only_and_key_file(tmp_path):
    for ok in ("127.0.0.1", "::1", "localhost", "127.0.0.2"):
        assert check_loopback(ok) == ok
    for bad in ("0.0.0.0", "192.168.1.10", "::", "example.com"):
        with pytest.raises(ValueError):
            check_loopback(bad)
    key = write_key_file(tmp_path / "sub" / "key")
    assert (tmp_path / "sub" / "key").read_text() == key and len(key) >= 32
    assert write_key_file(tmp_path / "sub" / "key") != key


def test_cli_refuses_non_loopback(tmp_path, capsys):
    rc = cli.main(["copilot-serve", "--host", "0.0.0.0", "--port", "1", "--model", "m",
                   "--key-file", str(tmp_path / "k")])
    assert rc == 2 and "loopback" in capsys.readouterr().out
    assert not (tmp_path / "k").exists()


@contextmanager
def running(app):
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "uvicorn did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(10)


def test_litellm_openai_provider_against_copilot_serve():
    import litellm

    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"],
              "additionalProperties": False}
    scripts = iter([["plain answer"], ['{"verdict": "pass"}']])
    sdk, app = make_app(lambda kw: next(scripts))
    with running(app) as base:
        r1 = litellm.completion(model="openai/gpt-5-mini", api_base=base, api_key=KEY, temperature=1,
                                messages=[{"role": "system", "content": "judge"},
                                          {"role": "user", "content": "grade"}])
        r2 = litellm.completion(model="openai/gpt-5-mini", api_base=base, api_key=KEY, seed=1,
                                messages=[{"role": "user", "content": "grade as json"}],
                                response_format={"type": "json_schema",
                                                 "json_schema": {"name": "v", "schema": schema, "strict": True}})
        with pytest.raises(Exception):
            litellm.completion(model="openai/gpt-5-mini", api_base=base, api_key="wrong",
                               messages=[{"role": "user", "content": "x"}], num_retries=0)
    assert r1.choices[0].message.content == "plain answer"
    assert r1.model.endswith("gpt-fake-served")
    assert r1.usage.prompt_tokens == 10 and r1.usage.completion_tokens == 5
    assert json.loads(r2.choices[0].message.content) == {"verdict": "pass"}
    assert sdk.sessions[0].kwargs["system_message"]["content"] == "judge"


def test_offline_profile_client_talks_to_copilot_serve():
    from ci_lab.providers.factory import make_chat_client

    _, app = make_app(["via openai client"])
    with running(app) as base:
        client = make_chat_client(profile="offline", model="gpt-5-mini", purpose="judge", base_url=base,
                                  api_key=KEY)
        resp = asyncio.run(client.get_response("hi"))
    assert resp.text == "via openai client"

class _RecordingClient:
    model = "gpt-5-mini"

    def __init__(self):
        self.seen = []

    async def get_response(self, messages, options=None):
        from agent_framework import ChatResponse, Message

        self.seen.append(obs.current_ids())
        return ChatResponse(messages=[Message("assistant", ["ok"])], model="served")


def test_trace_context_propagates_per_request():
    from opentelemetry.sdk.trace import TracerProvider

    from ci_lab.providers.factory import make_chat_client

    stub = _RecordingClient()
    app = create_app(stub, api_key=KEY)
    tracer = TracerProvider().get_tracer("test")
    with running(app) as base:
        client = make_chat_client(profile="offline", model="gpt-5-mini", purpose="judge", base_url=base, api_key=KEY)

        async def call(name):
            with tracer.start_as_current_span(name) as s:
                await client.get_response("hi")
                return format(s.get_span_context().trace_id, "032x")

        t1 = asyncio.run(call("a"))
        t2 = asyncio.run(call("b"))
        asyncio.run(client.get_response("no span"))
    assert t1 != t2
    assert [ids[0] if ids else None for ids in stub.seen] == [t1, t2, None]


def test_service_env_strips_pinned_trace_context(monkeypatch):
    from ci_lab.providers.copilot import service_env

    monkeypatch.setenv("TRACEPARENT", "00-" + "1" * 32 + "-" + "2" * 16 + "-01")
    monkeypatch.setenv("TRACESTATE", "x=y")
    env = service_env()
    assert "TRACEPARENT" not in env and "TRACESTATE" not in env and env["PATH"] == os.environ["PATH"]


def test_spawn_starts_loopback_child(tmp_path, monkeypatch):
    from ci_lab.providers.serve import spawn

    import httpx

    monkeypatch.setenv("TRACEPARENT", "00-" + "1" * 32 + "-" + "2" * 16 + "-01")
    with pytest.raises(ValueError):
        spawn(model="gpt-5-mini", key_file=tmp_path / "k", host="0.0.0.0")
    with spawn(model="gpt-5-mini", key_file=tmp_path / "key", ready_timeout_s=180) as srv:
        assert srv.base_url.startswith("http://127.0.0.1:") and srv.proc.poll() is None
        assert srv.api_key == (tmp_path / "key").read_text()
        r = httpx.get(f"{srv.base_url}/models", headers={"Authorization": "Bearer " + srv.api_key})
        assert r.json()["data"][0]["id"] == "gpt-5-mini"
        assert httpx.get(f"{srv.base_url}/models").status_code == 401
    assert srv.proc.poll() is not None
    assert srv.api_key not in srv.log_file.read_text(errors="replace")
