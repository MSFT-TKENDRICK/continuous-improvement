"""FastAPI/AG-UI server for the experiment designer (``ci-lab chat serve``).

Security model (docs/chat.md): loopback bind only, a per-process bearer token in the
``x-ci-chat-token`` header (constant-time compare) on ``/agui``, any request carrying an
``Origin`` header is refused (browsers always send one on cross-origin fetches, the canvas proxy
strips it), and ``launch_campaign`` is approval-gated by MAF itself.

Process contract: exactly one stdout line ``{"event":"listening",...}`` once the socket is
bound; everything else goes to stderr. Stdin EOF triggers a graceful shutdown, so the server
dies with its parent.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import socket
import sys
import threading
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from ci_lab.chat.agent import AGENT_NAME, build_agent
from ci_lab.chat.tools import ChatConfig, ChatTools

__all__ = ["AGUI_PATH", "DEFAULT_STATE", "LOOPBACK_HOSTS", "MIN_TOKEN_LEN", "TOKEN_ENV", "TOKEN_HEADER",
           "ServeError", "create_app", "make_client", "serve"]

AGUI_PATH = "/agui"
TOKEN_ENV = "CI_CHAT_TOKEN"
TOKEN_HEADER = "x-ci-chat-token"
MIN_TOKEN_LEN = 16
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
DEFAULT_MODEL = "gpt-5-mini"  # the repo-wide copilot default (sleep/optim/specs)
DEFAULT_STATE: dict[str, Any] = {"draft": None, "launches": []}
STATE_SCHEMA: dict[str, Any] = {
    "draft": {"type": ["object", "null"], "description": "normalized campaign draft from draft_campaign"},
    "launches": {"type": "array", "description": "[{cid, target, status, pid?, log?, run_url?}]"},
}

log = logging.getLogger("ci_lab.chat.server")


class ServeError(Exception):
    """A startup error reported as ``{"error": ...}`` on stderr with exit code 2."""


class _RejectOrigin:
    """ASGI middleware: 403 for any HTTP request that carries an ``Origin`` header."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and any(k.lower() == b"origin" for k, _ in scope.get("headers") or ()):
            body = b'{"detail":"cross-origin requests are not allowed"}'
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def _token_dependency(token: str) -> Callable[..., None]:
    expected = token.encode("utf-8")

    def require_token(x_ci_chat_token: str | None = Header(default=None)) -> None:
        if x_ci_chat_token is None or not hmac.compare_digest(x_ci_chat_token.encode("utf-8"), expected):
            raise HTTPException(status_code=401, detail="missing or bad x-ci-chat-token")

    return require_token


def create_app(agent: Any, *, token: str) -> FastAPI:
    """FastAPI app exposing ``agent`` on ``POST /agui`` (token + no Origin) and ``GET /healthz``."""
    from agent_framework_ag_ui import (
        AgentFrameworkAgent,
        add_agent_framework_fastapi_endpoint,
    )

    if len(token) < MIN_TOKEN_LEN:
        raise ServeError(f"{TOKEN_ENV} must be at least {MIN_TOKEN_LEN} characters")
    app = FastAPI(title="ci-lab experiment chat", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(_RejectOrigin)

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    # require_confirmation=False: no extra `confirm_changes` tool call; approval-gated tools still
    # interrupt the run (RUN_FINISHED outcome "interrupt") and only run after an approved resume.
    runner = AgentFrameworkAgent(agent=agent, name=AGENT_NAME, state_schema=STATE_SCHEMA,
                                 require_confirmation=False)
    add_agent_framework_fastapi_endpoint(app, runner, AGUI_PATH, default_state=DEFAULT_STATE,
                                         dependencies=[Depends(_token_dependency(token))])
    app.state.agui_runner = runner
    return app


def make_client(profile: str, *, model: str | None = None) -> Any:
    """Chat client for the designer: ``fake`` (scripted, offline) or ``copilot`` (Copilot SDK)."""
    if profile == "fake":
        from ci_lab.chat.fake import FakeDesignerClient

        return FakeDesignerClient()
    if profile == "copilot":
        from ci_lab.providers.copilot import CopilotChatClient

        return CopilotChatClient(model=model or os.environ.get("CI_CHAT_MODEL") or DEFAULT_MODEL)
    raise ServeError(f"unknown chat profile {profile!r}; use copilot or fake")


def _bind(host: str, port: int) -> socket.socket:
    if host not in LOOPBACK_HOSTS:
        raise ServeError(f"refusing to bind {host!r}: only {', '.join(LOOPBACK_HOSTS)} are allowed")
    addr = "127.0.0.1" if host == "localhost" else host
    family = socket.AF_INET6 if ipaddress.ip_address(addr).version == 6 else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            sock.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", socket.SO_REUSEADDR), 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((addr, port))
    except OSError as exc:
        sock.close()
        raise ServeError(f"cannot bind {addr}:{port}: {exc}") from exc
    sock.set_inheritable(False)
    return sock


def _watch_stdin(on_eof: Callable[[], None]) -> threading.Thread:
    def run() -> None:
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        try:
            while stream is not None and stream.read(4096):
                pass
        except (OSError, ValueError):
            pass
        log.info("stdin closed; shutting down")
        on_eof()

    thread = threading.Thread(target=run, name="ci-chat-stdin-watch", daemon=True)
    thread.start()
    return thread


def serve(config: ChatConfig, *, profile: str, host: str = "127.0.0.1", port: int = 0, model: str | None = None,
          stdin_watch: bool = True, token: str | None = None, stdout: Any = None) -> int:
    """Run the server until stdin EOF or a signal. Raises :class:`ServeError` on bad startup input."""
    import uvicorn

    token = os.environ.get(TOKEN_ENV, "") if token is None else token
    if len(token) < MIN_TOKEN_LEN:
        raise ServeError(f"{TOKEN_ENV} must be set to at least {MIN_TOKEN_LEN} characters")
    sock = _bind(host, port)
    bound_host, bound_port = sock.getsockname()[:2]
    app = create_app(build_agent(make_client(profile, model=model), ChatTools(config)), token=token)
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False, lifespan="off",
                                           timeout_graceful_shutdown=5))
    out = stdout or sys.stdout

    async def main() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done():
                await task
                return
            await asyncio.sleep(0.01)
        out.write(json.dumps({"event": "listening", "host": bound_host, "port": bound_port, "path": AGUI_PATH},
                             separators=(",", ":")) + "\n")
        out.flush()
        log.info("experiment chat listening on %s:%s%s (profile %s)", bound_host, bound_port, AGUI_PATH, profile)
        if stdin_watch:
            _watch_stdin(lambda: loop.call_soon_threadsafe(setattr, server, "should_exit", True))
        await task

    try:
        asyncio.run(main())
    finally:
        sock.close()
    return 0
