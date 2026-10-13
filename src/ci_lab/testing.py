"""Test doubles shared by every ci_lab module (no network, no Copilot, no .NET)."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal, Self

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)
from agent_framework.observability import ChatTelemetryLayer

from ci_lab.contracts import RolloutKey


@dataclass
class Call:
    """A scripted assistant tool call."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


Step = str | Sequence[Call] | Callable[[Sequence[Message], Mapping[str, Any]], "str | Sequence[Call]"]


class FakeChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    """Scriptable MAF chat client. Each ``get_response`` consumes the next step:
    a str (final text), a list of :class:`Call` (tool calls), or a callable
    ``(messages, options) -> str | list[Call]``. When the script is exhausted it
    returns ``default``. Records every request in ``requests``."""

    OTEL_PROVIDER_NAME = "fake"

    def __init__(self, script: Sequence[Step] = (), *, model: str = "fake-model",
                 default: str = "ok", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.script = list(script)
        self.default = default
        self.requests: list[tuple[list[Message], dict[str, Any]]] = []

    async def _respond(self, messages: Sequence[Message], options: Mapping[str, Any]) -> ChatResponse:
        self.requests.append((list(messages), dict(options)))
        step: Any = self.script.pop(0) if self.script else self.default
        if callable(step):
            step = step(messages, options)
        if isinstance(step, str):
            contents: list[Any] = [step]
        else:
            n = len(self.requests)
            contents = [Content.from_function_call(call_id=c.call_id or f"call-{n}-{i}", name=c.name,
                                                   arguments=dict(c.arguments)) for i, c in enumerate(step)]
        return ChatResponse(messages=[Message(role="assistant", contents=contents)], model=self.model)

    def _inner_get_response(self, *, messages: Sequence[Message], stream: bool,
                            options: Mapping[str, Any], **kwargs: Any) -> Awaitable[ChatResponse]:
        if stream:
            raise NotImplementedError("FakeChatClient does not stream")
        return self._respond(messages, options)


class MemoryOutbox:
    """In-memory :class:`ci_lab.contracts.Outbox` (durable version lives in ci_lab.ledger.outbox)."""

    def __init__(self) -> None:
        self.done: dict[str, Any] = {}
        self.calls: list[str] = []

    def run_once(self, op: str, fn: Callable[[], Any], *, reconcile: Callable[[], Any | None] | None = None) -> Any:
        if op in self.done:
            return self.done[op]
        if reconcile is not None and (found := reconcile()) is not None:
            self.done[op] = found
            return found
        self.calls.append(op)
        self.done[op] = result = fn()
        return result

    async def arun_once(self, op: str, fn: Callable[[], Awaitable[Any]], *,
                        reconcile: Callable[[], Awaitable[Any | None]] | None = None) -> Any:
        if op in self.done:
            return self.done[op]
        if reconcile is not None and (found := await reconcile()) is not None:
            self.done[op] = found
            return found
        self.calls.append(op)
        self.done[op] = result = await fn()
        return result


class MemoryJournal:
    """In-memory :class:`ci_lab.contracts.RolloutJournal` with event-id dedupe."""

    def __init__(self) -> None:
        self.rollouts: dict[str, dict[str, Any]] = {}

    def start(self, key: RolloutKey, input: Mapping[str, Any]) -> None:
        self.rollouts.setdefault(key.rollout_id, {"input": dict(input), "events": {}, "status": "running"})

    def event(self, key: RolloutKey, event_type: str, data: Mapping[str, Any], *, event_id: str) -> None:
        self.rollouts[key.rollout_id]["events"].setdefault(
            event_id, {"event_id": event_id, "event_type": event_type, "data": dict(data)})

    def finish(self, key: RolloutKey, status: Literal["succeeded", "failed"]) -> None:
        self.rollouts[key.rollout_id]["status"] = status

    def events(self, key: RolloutKey) -> list[dict[str, Any]]:
        return list(self.rollouts.get(key.rollout_id, {}).get("events", {}).values())


# ---------------------------------------------------------------- loopback model server

Reply = str | Mapping[str, Any]
Responder = Callable[[Mapping[str, Any]], Reply]


def s1_top_token(body: Mapping[str, Any], *, noul: str = "no", level: str = "max") -> str:
    """Deterministic llama-server first token for a ``ci_lab.judge.backends`` System-1 question.

    yes/no questions answer ``noul``; choice questions the first listed code; level questions the
    highest (``level="max"``) or lowest level.
    """
    text = str((body.get("messages") or [{}])[-1].get("content") or "")
    q = text.rsplit("QUESTION:", 1)[-1]
    if "Answer yes or no." in q:
        return noul
    if "OPTIONS:" in q:
        m = re.search(r"OPTIONS:\n([A-Z]):", q)
        return m.group(1) if m else "A"
    levels = [int(x) for x in re.findall(r"^(\d+): ", q.split("LEVELS:", 1)[-1], flags=re.MULTILINE)] or [0]
    return str(max(levels) if level == "max" else min(levels))


def tool_call(name: str, arguments: Mapping[str, Any], call_id: str | None = None) -> dict[str, Any]:
    """An OpenAI ``tool_calls`` reply for :class:`LoopbackLLM` responders."""
    return {"tool_calls": [{"id": call_id or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(dict(arguments))}}]}


class LoopbackLLM:
    """Deterministic OpenAI-compatible model server on ``127.0.0.1`` (plus llama-server ``/props``).

    Serves ``POST /v1/chat/completions`` (and ``/chat/completions``) from ``respond(body)``, which
    returns assistant text, a message mapping (``{"content"|"tool_calls": ...}``, see
    :func:`tool_call`) or a full completion (``{"choices": [...]}``). Requests asking for
    ``logprobs`` get the first-token ``top_logprobs`` llama-server returns, with the token from
    ``judge(body)`` (default :func:`s1_top_token`). Every request body is kept in ``requests``.
    Streaming requests are answered as a one-chunk SSE stream.
    """

    def __init__(self, respond: Responder | None = None, *, judge: Callable[[Mapping[str, Any]], str] | None = None,
                 model_path: str = "/models/loopback-judge.gguf") -> None:
        self.respond = respond or (lambda body: "ok")
        self.judge = judge or s1_top_token
        self.model_path = model_path
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: Any = None
        self._thread: Any = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def kind(self, body: Mapping[str, Any]) -> str:
        if body.get("logprobs"):
            return "judge"
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return "json_schema"
        return "tools" if body.get("tools") else "chat"

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [b for b in self.requests if self.kind(b) == kind]

    def _completion(self, body: Mapping[str, Any]) -> dict[str, Any]:
        model = str(body.get("model") or "loopback")
        usage = {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
        if body.get("logprobs"):
            tok = self.judge(body)
            return {"id": "cmpl-loopback", "object": "chat.completion", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": tok},
                                 "logprobs": {"content": [{"token": tok, "logprob": 0.0,
                                                           "top_logprobs": [{"token": tok, "logprob": 0.0}]}]}}],
                    "usage": usage}
        reply = self.respond(body)
        if isinstance(reply, Mapping) and "choices" in reply:
            return dict(reply)
        msg: dict[str, Any] = {"role": "assistant", "content": reply if isinstance(reply, str) else None}
        if isinstance(reply, Mapping):
            msg.update(reply)
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        return {"id": "cmpl-loopback", "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "finish_reason": finish, "message": msg}], "usage": usage}

    def start(self) -> Self:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, code: int, payload: Any, ctype: str = "application/json") -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                if path == "/props":
                    self._send(200, {"model_path": outer.model_path, "build_info": "loopback",
                                     "chat_template": "loopback"})
                elif path.endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": "loopback", "object": "model"}]})
                elif path in ("/health", "/healthz"):
                    self._send(200, {"status": "ok"})
                else:
                    self._send(404, {"error": {"message": f"no route {self.path}"}})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not path.endswith("/chat/completions"):
                    self._send(404, {"error": {"message": f"no route {self.path}"}})
                    return
                with outer._lock:
                    outer.requests.append(body)
                try:
                    out = outer._completion(body)
                except Exception as e:  # noqa: BLE001 - surfaced to the client as HTTP 500
                    self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
                    return
                if not body.get("stream"):
                    self._send(200, out)
                    return
                choice = out["choices"][0]
                delta = {k: v for k, v in choice["message"].items() if v is not None}
                if delta.get("tool_calls"):
                    delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(delta["tool_calls"])]
                chunks = [{**out, "object": "chat.completion.chunk",
                           "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {**out, "object": "chat.completion.chunk",
                           "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]}]
                sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                self._send(200, sse.encode(), "text/event-stream")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="loopback-llm", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
