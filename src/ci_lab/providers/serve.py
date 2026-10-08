"""``copilot-serve``: a tool-less, OpenAI-compatible endpoint backed by :class:`CopilotChatClient` (design C7).

It exists so that OpenAI-speaking components (LiteLLM ``openai/<model>`` with
``api_base``, e.g. ASSERT's judge) can use Copilot models. The server:

* binds to loopback only and refuses any other host;
* requires ``Authorization: Bearer <key>``. The random key is written to ``--key-file``
  and never logged;
* implements non-streaming ``POST /v1/chat/completions`` and ``GET /v1/models``;
* maps ``response_format`` ``json_schema`` / ``json_object`` to structured output;
* rejects ``tools`` / ``functions`` / ``stream=true`` / ``n>1`` with an OpenAI-style 400;
* accepts and ignores sampling parameters (temperature, seed, top_p, ...) and lists them
  in the ``x-ci-ignored-params`` response header;
* maps Copilot timeouts to 504.
"""

from __future__ import annotations

import ipaddress
import os
import secrets
import socket
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_framework import Message
from agent_framework.exceptions import ChatClientException, ChatClientInvalidResponseException
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ci_lab import obs
from ci_lab.providers.copilot import CopilotChatClient, CopilotTimeoutError, copilot_scope, service_env
from ci_lab.providers.models import ModelPreflightError

__all__ = ["IGNORED_PARAMS", "CopilotServeProcess", "check_loopback", "create_app", "serve", "spawn"]

IGNORED_PARAMS = frozenset({
    "temperature", "top_p", "seed", "frequency_penalty", "presence_penalty", "max_tokens",
    "max_completion_tokens", "stop", "logit_bias", "logprobs", "top_logprobs", "user", "n",
    "stream_options", "parallel_tool_calls", "service_tier", "store", "metadata", "reasoning_effort",
})
_REJECTED = {"tools": "tools", "functions": "tools", "tool_choice": "tools", "function_call": "tools"}


def check_loopback(host: str) -> str:
    """Return ``host`` if it is a loopback address. Otherwise raise ``ValueError``."""
    if host == "localhost":
        return host
    try:
        if ipaddress.ip_address(host.strip("[]")).is_loopback:
            return host
    except ValueError:
        pass
    raise ValueError(f"copilot-serve only binds to loopback addresses (got {host!r})")


def _error(status: int, message: str, *, param: str | None = None, code: str | None = None,
           type_: str = "invalid_request_error", headers: dict[str, str] | None = None) -> Any:
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": type_, "param": param, "code": code}})


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                parts.append(str(part.get("text") or ""))
            else:
                raise ValueError("only text content parts are supported")
        return "".join(parts)
    raise ValueError("unsupported message content")


def _messages(raw: Any) -> list[Message]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty list")
    out = []
    for m in raw:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported message role {role!r} (this endpoint has no tools)")
        if m.get("tool_calls") or m.get("function_call"):
            raise ValueError("tool calls are not supported by this endpoint")
        out.append(Message(role=role, contents=[_text(m.get("content"))]))
    return out


def create_app(client: Any, *, api_key: str, close_client: bool = False, check_model: bool = True) -> Any:
    """Build the FastAPI app. ``client`` is a MAF chat client (normally :class:`CopilotChatClient`).

    With ``check_model`` the served model is checked against the Copilot account's model list at
    startup (``client.check_model()``), so a stale ``--model`` fails before the first request."""

    @asynccontextmanager
    async def lifespan(_app: Any) -> Any:
        try:
            if check_model and hasattr(client, "check_model"):
                try:
                    await client.check_model()
                except ModelPreflightError as exc:
                    print(f"copilot-serve: {exc}\n  override: ci-lab copilot-serve --model <id>", file=sys.stderr,
                          flush=True)
                    raise
            yield
        finally:
            if close_client and hasattr(client, "close"):
                await client.close()

    app = FastAPI(title="ci-lab copilot-serve", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    model_id = str(getattr(client, "model", "copilot"))

    def authorized(request: Request) -> bool:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and secrets.compare_digest(token.strip().encode(), api_key.encode())

    def unauthorized() -> Any:
        return _error(401, "Invalid or missing API key.", type_="authentication_error", code="invalid_api_key")

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        if not authorized(request):
            return unauthorized()
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0,
                                            "owned_by": "github-copilot"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        if not authorized(request):
            return unauthorized()
        try:
            body = await request.json()
        except Exception:
            return _error(400, "Request body must be JSON.")
        if not isinstance(body, dict):
            return _error(400, "Request body must be a JSON object.")
        if body.get("stream"):
            return _error(400, "Streaming is not supported by copilot-serve.", param="stream",
                          code="unsupported_parameter")
        for key, param in _REJECTED.items():
            if body.get(key) not in (None, [], "none"):
                return _error(400, "Tools are not supported by copilot-serve.", param=param,
                              code="unsupported_parameter")
        if body.get("n") not in (None, 1):
            return _error(400, "Only n=1 is supported.", param="n", code="unsupported_parameter")
        requested = body.get("model")
        if requested and requested != model_id:
            return _error(404, f"The model {requested!r} is not served here (serving {model_id!r}).",
                          param="model", code="model_not_found")
        try:
            messages = _messages(body.get("messages"))
        except ValueError as exc:
            return _error(400, str(exc), param="messages")

        options: dict[str, Any] = {}
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_schema", "json_object"):
            options["response_format"] = fmt
        elif fmt not in (None, {}) and not (isinstance(fmt, dict) and fmt.get("type") == "text"):
            return _error(400, "Unsupported response_format.", param="response_format")
        known = {"model", "messages", "response_format", "stream", *_REJECTED}
        ignored = sorted(k for k in body if k not in known and body[k] is not None)
        headers = {"x-ci-ignored-params": ",".join(ignored)} if ignored else {}

        try:
            # Long-lived service: join the caller's trace per request (W3C traceparent header).
            with obs.use_carrier(request.headers), copilot_scope(f"copilot-serve-{uuid.uuid4().hex}"):
                response = await client.get_response(messages, options=options)
        except (CopilotTimeoutError, TimeoutError) as exc:
            return _error(504, f"Upstream Copilot timeout: {exc}", type_="timeout_error", code="timeout",
                          headers=headers)
        except ChatClientInvalidResponseException as exc:
            return _error(502, str(exc), type_="upstream_error", code="invalid_upstream_response", headers=headers)
        except ChatClientException as exc:
            return _error(502, str(exc), type_="upstream_error", code="upstream_error", headers=headers)

        usage = response.usage_details or {}
        prompt_tokens = int(usage.get("input_token_count") or 0)
        completion_tokens = int(usage.get("output_token_count") or 0)
        finish = str(getattr(response.finish_reason, "value", response.finish_reason) or "stop")
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model or model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response.text},
                         "finish_reason": finish, "logprobs": None}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }
        return JSONResponse(payload, headers=headers)

    return app


def write_key_file(path: str | os.PathLike[str]) -> str:
    """Write a fresh random bearer key to ``path`` (owner-only where supported) and return it."""
    key = secrets.token_urlsafe(32)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key)
    return key


def serve(*, host: str = "127.0.0.1", port: int, model: str, key_file: str | os.PathLike[str],
          client: Any | None = None, check_model: bool = True, **client_kwargs: Any) -> None:
    """Run copilot-serve in the foreground (blocking). Exits non-zero at startup if ``model`` is
    not in the Copilot account's model list (``check_model=False`` skips that)."""
    import uvicorn

    check_loopback(host)
    key = write_key_file(key_file)
    owned = client is None
    client = client or CopilotChatClient(model=model, **client_kwargs)
    app = create_app(client, api_key=key, close_client=owned, check_model=check_model)
    print(f"copilot-serve: http://{host}:{port}/v1 model={model} key-file={Path(key_file).resolve()}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


@dataclass
class CopilotServeProcess:
    """A copilot-serve child process started by :func:`spawn`. ``api_key`` is read from the key file."""

    proc: subprocess.Popen[bytes]
    base_url: str
    api_key: str
    key_file: Path
    log_file: Path

    def close(self, timeout_s: float = 10.0) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout_s)

    def __enter__(self) -> CopilotServeProcess:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def spawn(*, model: str, key_file: str | os.PathLike[str], host: str = "127.0.0.1", port: int | None = None,
          reasoning_effort: str | None = None, timeout_s: float = 300.0, ready_timeout_s: float = 60.0,
          python: str = sys.executable) -> CopilotServeProcess:
    """Start ``ci-lab copilot-serve`` in a child process and wait until it answers ``/v1/models``.

    copilot-serve is long-lived, so it does not inherit a pinned ``TRACEPARENT``; callers
    propagate trace context per request with ``obs.carrier()`` headers. Child output goes to
    ``<key_file>.log``.
    """
    import httpx

    check_loopback(host)
    port = port or _free_port("127.0.0.1" if host == "localhost" else host.strip("[]"))
    key_path = Path(key_file).resolve()
    key_path.unlink(missing_ok=True)
    args = [python, "-c", "import sys; from ci_lab.cli import main; sys.exit(main())", "copilot-serve",
            "--host", host, "--port", str(port), "--model", model, "--key-file", str(key_path),
            "--timeout-s", str(timeout_s)]
    if reasoning_effort:
        args += ["--reasoning-effort", reasoning_effort]
    log_path = key_path.with_name(key_path.name + ".log")
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(args, env=service_env(), stdout=log, stderr=subprocess.STDOUT)
    url_host = f"[{host.strip('[]')}]" if ":" in host else host
    base_url = f"http://{url_host}:{port}/v1"
    deadline = time.monotonic() + ready_timeout_s
    try:
        while True:
            if proc.poll() is not None:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"copilot-serve exited with code {proc.returncode}: {tail}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"copilot-serve did not become ready within {ready_timeout_s}s")
            key = key_path.read_text(encoding="utf-8").strip() if key_path.exists() else ""
            if key:
                try:
                    r = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=2)
                    if r.status_code == 200:
                        return CopilotServeProcess(proc, base_url, key, key_path, log_path)
                except httpx.HTTPError:
                    pass
            time.sleep(0.2)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)
        raise
