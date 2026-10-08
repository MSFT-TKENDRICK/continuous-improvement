"""Launch and manage a loopback ``agl-server`` (Agent Lightning 1.0.2) subprocess.

The server is started with ``sys.executable -m agentlightning.server`` and Hydra
overrides (host, free port, key, ``default_proxy.*``). The random API key is
passed through the child's environment (``key=${oc.env:CI_LAB_AGL_KEY}``) so
it never appears on a command line or in logs. Native Windows is supported
(only ``agl-controller``'s local runner refuses Windows).
"""

from __future__ import annotations

import logging
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from ci_lab import obs
from ci_lab.agl.client import AglClient, ProxyMode, proxy_base_url
from ci_lab.contracts import RolloutKey

log = logging.getLogger(__name__)

KEY_ENV = "CI_LAB_AGL_KEY"
_SAFE_OVERRIDE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./")


class AglServerError(RuntimeError):
    pass


class _EarlyExit(AglServerError):
    pass


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _hydra_value(value: Any) -> str:
    text = str(value)
    if text and set(text) <= _SAFE_OVERRIDE:
        return text
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


class AglServer:
    """Context manager owning one ``agl-server`` process plus an authenticated :class:`AglClient`.

    ``endpoints`` are OpenAI-compatible upstream base URLs (including ``/v1``) registered for
    ``model_name`` once the server is healthy; the proxy forwards to them.
    """

    def __init__(self, model_name: str = "local", endpoints: Sequence[str] = (), *,
                 host: str = "127.0.0.1", port: int | None = None, key: str | None = None,
                 train_temperature: float = 1.0, val_temperature: float = 0.0,
                 include_log_probs: bool = False, startup_timeout: float = 30.0,
                 log_path: Path | str | None = None, cwd: Path | str | None = None) -> None:
        self.model_name = model_name
        self.endpoints = list(endpoints)
        self.host = host
        self.port = port
        self._key = key or secrets.token_urlsafe(32)
        self.train_temperature = train_temperature
        self.val_temperature = val_temperature
        self.include_log_probs = include_log_probs
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path) if log_path else None
        self.cwd = Path(cwd) if cwd else None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh: IO[bytes] | None = None
        self._client: AglClient | None = None

    def __repr__(self) -> str:
        return f"AglServer(model_name={self.model_name!r}, base_url={self.base_url!r}, running={self.running})"

    # ------------------------------------------------------------ properties

    @property
    def key(self) -> str:
        return self._key

    @property
    def base_url(self) -> str | None:
        return f"http://{self.host}:{self.port}" if self.port else None

    @property
    def client(self) -> AglClient:
        if self._client is None:
            raise AglServerError("server not started")
        return self._client

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def proxy_base_url(self, rollout: RolloutKey | str, mode: ProxyMode = "val", *,
                       attempt_id: str | None = None) -> str:
        if self.base_url is None:
            raise AglServerError("server not started")
        return proxy_base_url(self.base_url, rollout, mode, attempt_id=attempt_id)

    # ------------------------------------------------------------ lifecycle

    def command(self) -> list[str]:
        overrides = {
            "host": self.host,
            "port": self.port,
            "default_proxy.model_name": self.model_name,
            "default_proxy.include_log_probs": self.include_log_probs,
            "default_proxy.train.temperature": self.train_temperature,
            "default_proxy.val.temperature": self.val_temperature,
        }
        args = [f"{k}={_hydra_value(v)}" for k, v in overrides.items()]
        args.append("key=${oc.env:" + KEY_ENV + "}")
        # No Hydra output dir / log files in the caller's cwd.
        args += ["hydra.run.dir=.", "hydra.output_subdir=null",
                 "hydra/job_logging=disabled", "hydra/hydra_logging=disabled"]
        return [sys.executable, "-m", "agentlightning.server", *args]

    def start(self) -> AglServer:
        if self.running:
            return self
        last_error: Exception | None = None
        for _ in range(3 if self.port is None else 1):
            explicit_port = self.port is not None
            if not explicit_port:
                self.port = free_port(self.host)
            try:
                self._launch()
                return self
            except _EarlyExit as exc:  # likely lost a free-port race: retry on a new port
                last_error = exc
                self.stop()
                if explicit_port:
                    break
                self.port = None
            except BaseException:
                self.stop()
                raise
        raise last_error or AglServerError("agl-server failed to start")

    def child_env(self) -> dict[str, str]:
        """Server environment. Long-lived service: never inherits a startup ``TRACEPARENT``;
        trace context travels per request via ``obs.carrier()`` headers sent by AglClient
        (design §12.5)."""
        env = {k: v for k, v in os.environ.items() if k not in (obs.TRACEPARENT_ENV, "TRACESTATE")}
        env[KEY_ENV] = self._key
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _launch(self) -> None:
        env = self.child_env()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "ab")  # noqa: SIM115 - closed in stop()
            out: Any = self._log_fh
        else:
            out = subprocess.DEVNULL
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        log.info("starting agl-server on %s (model_name=%s)", self.base_url, self.model_name)
        self._proc = subprocess.Popen(self.command(), env=env, cwd=self.cwd, stdin=subprocess.DEVNULL,
                                      stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                                      start_new_session=os.name != "nt")
        assert self.base_url is not None
        self._client = AglClient(self.base_url, self._key)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise _EarlyExit(f"agl-server exited early with code {self._proc.returncode}")
            if self._client.healthz():
                break
            time.sleep(0.2)
        else:
            raise AglServerError(f"agl-server not healthy within {self.startup_timeout:.0f}s")
        if self.endpoints:
            self._client.register_models(
                [{"model": self.model_name, "endpoint": ep, "version": 0} for ep in self.endpoints])

    def register_endpoints(self, endpoints: Sequence[str], *, version: int = 0) -> None:
        self.client.register_models([{"model": self.model_name, "endpoint": ep, "version": version}
                                     for ep in endpoints])
        self.endpoints.extend(ep for ep in endpoints if ep not in self.endpoints)

    def stop(self, timeout: float = 10.0) -> None:
        proc, self._proc = self._proc, None
        if self._client is not None:
            self._client.close()
            self._client = None
        try:
            if proc is not None and proc.poll() is None:
                _terminate_tree(proc, timeout)
        finally:
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None

    def __enter__(self) -> AglServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _terminate_tree(proc: subprocess.Popen[bytes], timeout: float) -> None:
    if os.name == "nt":
        # A venv python.exe is a launcher that spawns the real interpreter: kill the whole tree.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)
