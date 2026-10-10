"""MCP **code mode**: one ``run_code(code) -> str`` tool instead of one tool per MCP tool.

The model writes a short Python snippet against generated typed stubs (``tools.<server>.<tool>(**kwargs)``),
so one turn can chain several tool calls and print only what it needs.

This is an **isolation layer, not an OS sandbox**. Its layers are:

1. a static AST allowlist (:func:`ci_lab.mcp.codecheck.check_code`): only imports from the exposure's
   ``imports`` and ``tools``; no ``os``/``sys``/``subprocess``/``socket``/``ctypes``/``importlib``; no
   ``open``/``exec``/``eval``/``compile``; no names or attributes starting with ``_``; literal ``getattr`` only;
2. a separate ``sys.executable -I bootstrap.py`` process in a fresh temp cwd with a scrubbed env, restricted
   builtins and a guarded ``__import__``;
3. a wall-clock timeout that kills the whole process tree (a Win32 Job Object via pywin32, else
   ``proc.kill()``; a new session + ``killpg`` on POSIX);
4. every tool call is a JSON-lines RPC on the child's ORIGINAL fd 1 / fd 0 (``{"rpc": ...}`` envelopes) that
   the parent forwards through :class:`~ci_lab.mcp.client.McpHub`, so exposure allowlists and the
   ``before_call`` governance hook still apply. The child's fds 0/1 are re-pointed at ``os.devnull`` and
   ``sys.stdout``/``sys.stderr`` at a capped buffer, so user prints never reach the channel.

It does not restrict CPU/memory beyond the timeout, filesystem reads by allowed modules, or a determined
escape from CPython itself. Run untrusted code inside an OS sandbox/container if that matters.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ci_lab.mcp.codecheck import check_code
from ci_lab.mcp.registry import CodeModeConfig
from ci_lab.mcp.stubs import bootstrap_source, describe, generate_stubs, usable

OnToolCall = Callable[[str], Any]
_ENV_KEEP = ("SYSTEMROOT",)


class ProcessTree:
    """Kill a child and all of its descendants: Job Object on Windows (pywin32), process group on POSIX."""

    def __init__(self, proc: subprocess.Popen[Any]):
        self.proc = proc
        self.job: Any = None
        if sys.platform == "win32":
            self.job = _assign_job(proc.pid)

    def kill(self) -> None:
        if self.job is not None:
            import win32job

            with contextlib.suppress(Exception):  # already gone
                win32job.TerminateJobObject(self.job, 1)
        elif sys.platform != "win32" and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 9)
            except OSError:
                pass
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self) -> None:
        if self.job is not None:
            import win32api

            win32api.CloseHandle(self.job)  # KILL_ON_JOB_CLOSE reaps any survivor
            self.job = None


def _assign_job(pid: int) -> Any:
    try:
        import win32api
        import win32con
        import win32job
    except ImportError:
        return None
    job = None
    try:
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, pid)
        try:
            win32job.AssignProcessToJobObject(job, handle)
        finally:
            win32api.CloseHandle(handle)
        return job
    except Exception:  # noqa: BLE001 - fall back to proc.kill()
        if job is not None:
            win32api.CloseHandle(job)
        return None


def _cap(text: str, limit: int, total: int | None = None) -> str:
    total = len(text) if total is None else total
    if total <= limit and len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated: {limit} of {total} chars shown]"


class CodeMode:
    """The single ``run_code`` tool over a hub's ``mode: code`` tools.

    ``on_tool_call(name)`` is invoked once per tool call made inside a snippet (``name`` = ``server.tool``),
    e.g. ``RunMeter.tool_call``; ``self.tool_calls`` counts them regardless.
    """

    def __init__(self, hub: Any, cfg: CodeModeConfig | None = None, on_tool_call: OnToolCall | None = None):
        self.hub = hub
        self.cfg = cfg or hub.exposure.code_mode
        self.on_tool_call = on_tool_call
        self.tool_calls = 0
        self.last_pid: int | None = None
        self._tools = usable(hub.tools(mode="code"))
        self._keys = {(t.server, t.name) for t in self._tools}

    @property
    def description(self) -> str:
        return describe(self._tools, timeout_s=self.cfg.timeout_s, max_output_chars=self.cfg.max_output_chars,
                        imports=self.cfg.imports)

    def maf_tool(self) -> Any:
        from agent_framework import FunctionTool

        async def run_code(code: str) -> str:
            return await self.run_code(code)

        return FunctionTool(name="run_code", description=self.description, func=run_code)

    async def run_code(self, code: str) -> str:
        if errs := check_code(code, self.cfg.imports):
            return "error: code rejected:\n" + "\n".join(f"- {e}" for e in errs)
        with tempfile.TemporaryDirectory(prefix="ci-codemode-") as tmp:
            return await self._run(Path(tmp), code)

    def _prepare(self, tmp: Path, code: str) -> Path:
        stub_dir = tmp / "stubs"
        for rel, src in generate_stubs(self._tools).items():
            p = stub_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src, encoding="utf-8")
        (tmp / "user_code.py").write_text(code, encoding="utf-8")
        boot = tmp / "bootstrap.py"
        boot.write_text(bootstrap_source(str(stub_dir), allowed_imports=self.cfg.imports,
                                         max_output_chars=self.cfg.max_output_chars), encoding="utf-8")
        return boot

    @staticmethod
    def _spawn(tmp: Path, boot: Path) -> subprocess.Popen[bytes]:
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ} | {"TEMP": str(tmp), "TMP": str(tmp)}
        kwargs: dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" \
            else {"start_new_session": True}
        with open(tmp / "stderr.txt", "wb") as errf:
            return subprocess.Popen([sys.executable, "-I", str(boot)], cwd=tmp, env=env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=errf, **kwargs)

    async def _run(self, tmp: Path, code: str) -> str:
        proc = self._spawn(tmp, self._prepare(tmp, code))
        tree = ProcessTree(proc)
        self.last_pid = proc.pid
        try:
            return await asyncio.wait_for(self._serve(proc, tmp), timeout=self.cfg.timeout_s)
        except TimeoutError:
            tree.kill()
            return f"error: timed out after {self.cfg.timeout_s:g}s; process tree killed"
        finally:
            tree.kill()
            try:
                await asyncio.to_thread(proc.wait, 10)
            finally:
                tree.close()
                for s in (proc.stdin, proc.stdout):
                    try:
                        s.close()
                    except OSError:
                        pass

    async def _serve(self, proc: subprocess.Popen[bytes], tmp: Path) -> str:
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:
                rc = await asyncio.to_thread(proc.wait)
                tail = (tmp / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-2000:]
                return _cap(f"error: code-mode process exited ({rc}) without a result\n{tail}".rstrip(),
                            self.cfg.max_output_chars)
            try:
                msg = json.loads(line).get("rpc")
            except (json.JSONDecodeError, AttributeError):
                msg = None
            if not isinstance(msg, dict):
                continue
            if msg.get("method") == "done":
                return self._result(msg)
            if msg.get("method") == "call":
                reply = await self._forward(msg)
                await asyncio.to_thread(self._send, proc, reply)

    @staticmethod
    def _send(proc: subprocess.Popen[bytes], reply: dict[str, Any]) -> None:
        proc.stdin.write((json.dumps({"rpc": reply}, default=str) + "\n").encode("utf-8"))
        proc.stdin.flush()

    async def _forward(self, msg: dict[str, Any]) -> dict[str, Any]:
        server, tool, args = msg.get("server"), msg.get("tool"), msg.get("args") or {}
        self.tool_calls += 1
        if self.on_tool_call is not None:
            self.on_tool_call(f"{server}.{tool}")
        if (server, tool) not in self._keys or not isinstance(args, dict):
            return {"id": msg.get("id"), "ok": False, "error": f"{server}.{tool} is not available in code mode"}
        try:
            result = await self.hub.call_tool(server, tool, args)
        except Exception as e:  # noqa: BLE001 - surfaced to the snippet as tools.ToolError
            return {"id": msg.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"id": msg.get("id"), "ok": True, "result": result}

    def _result(self, msg: dict[str, Any]) -> str:
        out, err = str(msg.get("output") or ""), msg.get("error")
        text = _cap(out, self.cfg.max_output_chars, int(msg.get("output_chars") or len(out)))
        if err:
            text = (text + "\n" if text else "") + str(err).rstrip()[-2000:]
        return text or "(no output)"
