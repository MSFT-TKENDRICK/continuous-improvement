"""``McpHub``: start the frozen-registry MCP servers that the exposure overlay selects, and call their tools.

Every call is checked against the exposed tool set (exposure ∩ frozen ``tools_allow``) and then routed
through a ``before_call(server, tool, args)`` governance hook (AGT/ACS bind it), before it reaches the
server. ``maf_tools(hub)`` wraps direct-mode tools as MAF ``FunctionTool``s, one per MCP tool.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ci_lab.mcp.registry import Exposure, McpConfigError, Registry, load_registry


class McpDenied(PermissionError):
    """A call refused by the exposure allowlist or the ``before_call`` governance hook."""

    def __init__(self, server: str, tool: str, reason: str):
        super().__init__(f"MCP call {server}.{tool} denied: {reason}")
        self.server, self.tool, self.reason = server, tool, reason


class McpToolError(RuntimeError):
    """The server reported a tool error (``isError``)."""


@dataclass(frozen=True)
class Denial:
    reason: str


# Returns None/True to allow; False, a reason string or a Denial to deny; may also raise. May be async.
BeforeCall = Callable[[str, str, dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolInfo:
    server: str
    name: str
    description: str
    input_schema: Mapping[str, Any]
    mode: str
    output_schema: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def qualified(self) -> str:
        return f"{self.server}.{self.name}"


def _contains(eg: BaseExceptionGroup, exc: BaseException) -> bool:
    return any(e is exc or (isinstance(e, BaseExceptionGroup) and _contains(e, exc)) for e in eg.exceptions)


def _result_value(result: Any, info: ToolInfo) -> Any:
    sc = result.structuredContent
    if sc is not None:
        wrapped = bool(info.output_schema and info.output_schema.get("x-fastmcp-wrap-result"))
        return sc["result"] if wrapped and isinstance(sc, dict) and "result" in sc else sc
    text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "type", None) == "text")
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


class McpHub:
    """Async context manager owning one stdio session per exposed server.

    ``roots`` overrides registry root defaults by root name (e.g. ``{"harness_root": candidate_dir}``);
    unspecified roots resolve relative to ``base_dir`` (default: the current directory).
    """

    def __init__(self, registry: Registry | None = None, exposure: Exposure | None = None, *,
                 roots: Mapping[str, str | Path] | None = None, base_dir: str | Path | None = None,
                 before_call: BeforeCall | None = None, environ: Mapping[str, str] | None = None,
                 servers: Sequence[str] | None = None, errlog: TextIO | None = None):
        self.registry = registry or load_registry()
        self.exposure = exposure or Exposure.everything(self.registry)
        names = list(self.exposure.servers) if servers is None else list(servers)
        if missing := [n for n in names if n not in self.exposure.servers]:
            raise McpConfigError([f"servers not exposed: {missing}"])
        roots = dict(roots or {})
        declared = {r for n in names for r in self.registry.servers[n].roots}
        if unknown := sorted(set(roots) - declared):
            raise McpConfigError([f"unknown roots {unknown}"])
        self._names = names
        self._roots = roots
        self._base_dir = base_dir
        self.before_call = before_call
        self._environ = environ
        self._errlog = errlog
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[tuple[str, str], ToolInfo] = {}
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> Self:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            for name in self._names:
                await self._start(stack, name)
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                        tb: TracebackType | None) -> bool:
        stack, self._stack = self._stack, None
        self._sessions.clear()
        self._tools.clear()
        if stack is None:
            return False
        try:
            await stack.__aexit__(exc_type, exc, tb)
        except BaseExceptionGroup as eg:
            # The SDK's task groups wrap the body's exception; surface the original (e.g. McpDenied).
            if exc is not None and _contains(eg, exc):
                return False
            raise
        return False

    async def _start(self, stack: AsyncExitStack, name: str) -> None:
        sdef, exp = self.registry.servers[name], self.exposure.servers[name]
        resolved = sdef.resolve_roots({k: v for k, v in self._roots.items() if k in sdef.roots}, self._base_dir)
        argv = sdef.argv(resolved)
        params = StdioServerParameters(command=argv[0], args=argv[1:], env=sdef.env(self._environ))
        read, write = await stack.enter_async_context(stdio_client(params, errlog=self._errlog or sys.stderr))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[name] = session
        for t in (await session.list_tools()).tools:
            if t.name in sdef.tools_allow and t.name in exp.tools:
                self._tools[(name, t.name)] = ToolInfo(name, t.name, exp.tools[t.name] or t.description or "",
                                                       t.inputSchema or {"type": "object", "properties": {}},
                                                       exp.mode, t.outputSchema)

    def tools(self, server: str | None = None, mode: str | None = None) -> list[ToolInfo]:
        return [t for t in self._tools.values()
                if (server is None or t.server == server) and (mode is None or t.mode == mode)]

    async def list_tools(self, server: str | None = None) -> list[ToolInfo]:
        return self.tools(server)

    async def _govern(self, server: str, tool: str, args: dict[str, Any]) -> None:
        if self.before_call is None:
            return
        verdict = self.before_call(server, tool, args)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        if verdict is None or verdict is True:
            return
        reason = verdict.reason if isinstance(verdict, Denial) else verdict if isinstance(verdict, str) else "policy"
        raise McpDenied(server, tool, reason)

    async def call_tool(self, server: str, tool: str, args: Mapping[str, Any] | None = None) -> Any:
        """Call an exposed tool through governance; returns structured content (or parsed/plain text)."""
        info = self._tools.get((server, tool))
        if info is None:
            raise McpDenied(server, tool, "tool is not exposed")
        args = dict(args or {})
        await self._govern(server, tool, args)
        result = await self._sessions[server].call_tool(tool, args)
        if result.isError:
            raise McpToolError("; ".join(getattr(c, "text", "") for c in result.content) or "tool error")
        return _result_value(result, info)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def maf_tools(hub: McpHub, *, mode: str = "direct") -> list[Any]:
    """MAF ``FunctionTool``s for the hub's ``mode`` tools, named ``<server>__<tool>``; results are JSON text."""
    from agent_framework import FunctionTool

    def bind(info: ToolInfo) -> Callable[..., Awaitable[str]]:
        async def call(**kwargs: Any) -> str:
            return _as_text(await hub.call_tool(info.server, info.name, kwargs))
        return call

    return [FunctionTool(name=f"{t.server}__{t.name}", description=t.description,
                         input_model=dict(t.input_schema), func=bind(t))
            for t in hub.tools(mode=mode)]
