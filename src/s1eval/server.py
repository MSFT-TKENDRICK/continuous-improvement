"""Minimal, hardened TypeSafe System One-compatible HTTP server wrapping any s1eval backend.

Endpoints: POST /v1/systemone, GET /v1/models, GET /health.

Security defaults (local evaluation tool, not a production gateway):
  * binds to loopback; a non-loopback bind requires --insecure-allow-remote AND a token
  * Host header must match an allowed host (DNS-rebinding defence); no CORS headers at all
  * JSON content type required, Content-Length required, body size capped
  * bounded concurrency (busy -> 429 + Retry-After) because the local model is serial
  * access log contains method, path, status and latency only - never headers or bodies

Non-answers: TypeSafe answers have no "abstain" representation. By default a request with any
abstained/refused question fails with HTTP 502 ``judge_abstained`` (never a fabricated answer).
Clients that understand s1eval may send ``X-S1Eval-Allow-Abstain: 1`` to receive the remaining
answers plus a top-level ``s1eval.abstained`` list; ``X-S1Eval-Diagnostics: 1`` adds per-answer
diagnostics under an ``s1eval`` key (ignored by the official SDK, which uses extra="ignore").
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backends.base import BackendError
from .types import WireError, questions_from_wire

log = logging.getLogger("s1eval.server")

MAX_BODY_BYTES = 1_000_000
DRAIN_LIMIT_BYTES = 4 * MAX_BODY_BYTES
DRAIN_TIMEOUT_S = 2.0
MAX_QUESTIONS = 32


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class S1Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, backend, *, token: str | None, model_name: str, allowed_hosts: set[str],
                 max_concurrency: int = 1, busy_wait_s: float = 600.0):
        self.backend = backend
        self.token = token
        self.model_name = model_name
        self.allowed_hosts = allowed_hosts
        self.sem = threading.BoundedSemaphore(max_concurrency)
        self.busy_wait_s = busy_wait_s
        super().__init__(addr, Handler)


class Handler(BaseHTTPRequestHandler):
    server: S1Server
    server_version = "s1eval"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default (it logs request lines)
        pass

    def _discard_unread_body(self) -> None:
        """Drain (bounded) an unread request body before an early error reply.

        Closing a socket with unread input makes the TCP stack send RST, which can destroy the
        error response before the client reads it (seen as WinError 10053 / ECONNRESET).
        """
        self.close_connection = True
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = None
        budget = DRAIN_LIMIT_BYTES if length is None or length < 0 else min(length, DRAIN_LIMIT_BYTES)
        try:
            self.connection.settimeout(DRAIN_TIMEOUT_S)
            while budget > 0:
                chunk = self.rfile.read1(min(65536, budget)) if hasattr(self.rfile, "read1") else self.rfile.read(min(65536, budget))
                if not chunk:
                    break
                budget -= len(chunk)
        except OSError:
            pass

    def _send(self, status: int, payload: Any, extra_headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode()
        if getattr(self, "_body_pending", False):
            self._body_pending = False
            self._discard_unread_body()
        self.send_response(status)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
        log.info("%s %s %d", self.command, self.path.split("?")[0], status)

    def _error(self, status: int, etype: str, msg: str, extra_headers: dict[str, str] | None = None) -> None:
        self._send(status, {"error": {"type": etype, "message": msg}}, extra_headers)

    def _validation_error(self, loc: list[Any], msg: str) -> None:
        self._send(422, {"detail": [{"loc": loc, "msg": msg, "type": "value_error"}]})

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in self.server.allowed_hosts

    def _auth_ok(self) -> bool:
        if not self.server.token:
            return True
        got = self.headers.get("Authorization") or ""
        expected = f"Bearer {self.server.token}"
        return hmac.compare_digest(got.encode(), expected.encode())

    def _guard(self) -> bool:
        if not self._host_ok():
            self._error(421, "misdirected_request", "Host header not allowed")
            return False
        if not self._auth_ok():
            self._error(401, "authentication_error", "invalid or missing bearer token")
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?")[0]
        if path == "/health":
            self._send(200, {"status": "ok"})
        elif path == "/v1/systemone":
            self._error(405, "method_not_allowed", "method not allowed")
        elif path == "/v1/models":
            self._send(200, {"models": [{
                "name": self.server.model_name,
                "description": "s1eval local System One-style approximation (not Jev).",
                "release_date": date.today().isoformat(),
            }]})
        else:
            self._error(404, "not_found", "unknown path")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._error(405, "method_not_allowed", "CORS is not supported")

    def do_PUT(self) -> None:  # noqa: N802
        self._body_pending = True
        self._error(405, "method_not_allowed", "method not allowed")

    do_DELETE = do_PATCH = do_PUT

    def do_POST(self) -> None:  # noqa: N802
        self._body_pending = True
        if not self._guard():
            return
        if self.path.split("?")[0] != "/v1/systemone":
            self._error(404, "not_found", "unknown path")
            return
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._error(415, "unsupported_media_type", "Content-Type must be application/json")
            return
        if self.headers.get("Transfer-Encoding"):
            self._error(411, "length_required", "chunked bodies are not supported")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "length_required", "Content-Length required")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._error(413, "payload_too_large", f"body must be <= {MAX_BODY_BYTES} bytes")
            return
        raw = self.rfile.read(length)
        self._body_pending = False
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._validation_error(["body"], "body is not valid JSON")
            return
        if not isinstance(body, dict):
            self._validation_error(["body"], "body must be an object")
            return
        for f in ("state", "model", "questions"):
            if f not in body:
                self._validation_error(["body", f], "Field required")
                return
        if not isinstance(body["state"], (str, dict, list)):
            self._validation_error(["body", "state"], "state must be a string, object or array")
            return
        if not isinstance(body["model"], str):
            self._validation_error(["body", "model"], "model must be a string")
            return
        try:
            questions = questions_from_wire(body["questions"])
        except WireError as e:
            self._validation_error(["body", "questions"], str(e))
            return
        if len(questions) > MAX_QUESTIONS:
            self._validation_error(["body", "questions"], f"at most {MAX_QUESTIONS} questions per request")
            return

        if not self.server.sem.acquire(timeout=self.server.busy_wait_s):
            self._error(429, "rate_limit_error", "judge busy", {"Retry-After": "5"})
            return
        try:
            t0 = time.perf_counter()
            decision = self.server.backend.decide(body["state"], questions)
        except BackendError as e:
            self._error(502, "backend_error", str(e)[:300])
            return
        except WireError as e:
            self._validation_error(["body", "questions"], str(e))
            return
        finally:
            self.server.sem.release()

        allow_abstain = self.headers.get("X-S1Eval-Allow-Abstain") == "1"
        diagnostics = self.headers.get("X-S1Eval-Diagnostics") == "1"
        non_ok = {k: a.status for k, a in decision.answers.items() if not a.ok}
        if non_ok and not allow_abstain:
            self._send(502, {"error": {
                "type": "judge_abstained",
                "message": "the judge abstained or refused; no answer is fabricated",
                "questions": non_ok,
            }})
            return
        out: dict[str, Any] = {
            "model": self.server.model_name,
            "answers": {k: a.to_wire(include_diagnostics=diagnostics) for k, a in decision.answers.items() if a.ok},
            "usage": {"input_tokens": int(decision.usage.get("input_tokens") or 0),
                      "output_tokens": int(decision.usage.get("output_tokens") or 0)},
        }
        if non_ok:
            out["s1eval"] = {"abstained": non_ok}
        if diagnostics:
            out.setdefault("s1eval", {})["latency_s"] = round(time.perf_counter() - t0, 3)
            out["s1eval"]["model_calls"] = decision.model_calls
        self._send(200, out)


def make_server(backend, host: str = "127.0.0.1", port: int = 8090, *, token: str | None = None,
                model_name: str | None = None, insecure_allow_remote: bool = False,
                extra_allowed_hosts: set[str] | None = None, max_concurrency: int = 1) -> S1Server:
    if not is_loopback(host):
        if not insecure_allow_remote:
            raise ValueError(f"refusing to bind non-loopback host {host!r} without --insecure-allow-remote")
        if not token:
            raise ValueError("a non-loopback bind requires --token")
    srv = S1Server((host, port), backend, token=token, model_name=model_name or "s1eval-local",
                   allowed_hosts=set(), max_concurrency=max_concurrency)
    real_port = srv.server_address[1]
    hosts = {f"127.0.0.1:{real_port}", f"localhost:{real_port}", f"[::1]:{real_port}"}
    if not is_loopback(host):
        hosts.add(f"{host}:{real_port}")
    hosts |= {h.lower() for h in (extra_allowed_hosts or set())}
    srv.allowed_hosts = hosts
    return srv
