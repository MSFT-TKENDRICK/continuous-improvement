"""MAF host for the packaged ACS policies and the single governed agent factory.

:class:`Governance` binds one packaged policy to one agent and
yields MAF middleware for six ACS intervention points: agent ``input``/``output``, chat
``pre_model_call``/``post_model_call`` and function ``pre_tool_call``/``post_tool_call`` (points
absent from the manifest are skipped). In ``enforce`` a deny sets a refusal result and raises
``MiddlewareTermination``; transforms rewrite the message / arguments / result; in
``evaluate_only`` decisions are only recorded. Every decision is appended (content-free) to the
:class:`~ci_lab.governance.audit.AuditTrail`. Every MAF agent in ``src/`` must be built by
:func:`governed_agent`, :func:`governed_declarative` or :func:`governed_harness_agent`
(lint rule ``agt.governed-agent-factory``).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    MiddlewareTermination,
)
from pydantic import BaseModel

from ci_lab.governance.acs import AgentControlSuspended
from ci_lab.governance.acs.host import AgentControlInterruption
from ci_lab.governance.acs.runtime import Decision, InterventionPointResult
from ci_lab.governance.approvals import FileApprovalQueue
from ci_lab.governance.audit import AuditTrail
from ci_lab.governance.hypervisor import KillSwitchAdapter
from ci_lab.governance.policies import governance_mode, load_policy
from ci_lab.guards.recorder import json_args, structured_result

__all__ = ["Governance", "default_did", "govern", "governed_agent", "governed_declarative",
           "governed_harness_agent", "harness_mcp_before_call"]

GOVERNED_KEY = "ci_lab.governance"
HISTORY_KEY = "ci_lab.governance.history"
HISTORY_MAX = 200
_REASON = re.compile(r"^[A-Za-z0-9_.:/ -]{0,160}$")
_trails: dict[Path, AuditTrail] = {}


def _trail(path: Path | str | None) -> AuditTrail:
    t = AuditTrail(path)
    return _trails.setdefault(t.path.resolve(), t)


def _hex(identity: str | None) -> str | None:
    return identity.removeprefix("sha256:") if identity else None


def default_did(name: str) -> str:
    return ("did:ci-lab:agent:" + re.sub(r"[^A-Za-z0-9._-]", "-", name or "agent"))[:128]


def _refusal(exc: AgentControlInterruption) -> str:
    v = exc.result.verdict
    if isinstance(exc, AgentControlSuspended):
        return (f"Action held for approval ({v.reason}); run `ci-lab governance approve "
                f"{exc.result.enforced_identity}` and retry.")
    return f"Blocked by governance policy ({v.reason}): {v.message or 'not permitted'}"


class Governance:
    """One policy bound to one agent; see the module docstring."""

    def __init__(self, policy: str, *, agent_name: str = "agent", role: str | None = None,
                 model: str | None = None, mode: str | None = None, agent_did: str | None = None,
                 audit: AuditTrail | Path | str | None = None,
                 approvals: Callable[..., Any] | None = None,
                 kill_switch: KillSwitchAdapter | None = None,
                 allowed_models: Collection[str] | None = None) -> None:
        self.policy, self.mode = policy, mode or governance_mode()
        self.allowed_models = None if allowed_models is None else frozenset(allowed_models)
        self.audit = audit if isinstance(audit, AuditTrail) else _trail(audit)
        self.kill_switch = kill_switch or KillSwitchAdapter()
        self.agent = {"name": agent_name, "role": role or agent_name}
        self.model, self.did = model, agent_did or default_did(agent_name)
        self.control = load_policy(policy, mode=self.mode, decision_log=self._log,
                                   approval_resolver=approvals or FileApprovalQueue())
        self.points = frozenset(self.control.runtime.manifest.points)
        self._history: list[dict[str, Any]] = []

    def _log(self, entry: Mapping[str, Any]) -> None:
        reason, decision, approval = entry.get("reason") or "", entry["decision"], entry.get("approval")
        reason = reason if _REASON.fullmatch(reason) else "invalid_reason"
        if approval:  # a liftable deny routed to the approval queue
            reason, decision = f"{reason} approval:{approval}"[:160], "allow" if approval == "allow" else decision
        self.audit.append({
            "agent_did": self.did, "intervention_point": entry["point"],
            "decision": decision, "reason": reason,
            "input_identity": _hex(entry.get("input_identity")),
            "enforced_identity": _hex(entry.get("enforced_identity")),
            "mode": entry["mode"], "ts": datetime.fromtimestamp(entry["ts"], UTC)})

    async def check(self, point: str, snapshot: Mapping[str, Any],
                    tool: str | None = None) -> InterventionPointResult | None:
        """Guard one point; ``None`` if unconfigured. Raises ``AgentControlBlocked``/``Suspended``."""
        if point not in self.points:
            return None
        snap = {**snapshot, "agent": self.agent,
                "governance": {"kill_switch": self.kill_switch.engaged()}}
        return await self.control.guard(point, snap, tool)

    def transformed(self, result: InterventionPointResult | None) -> Any:
        """The enforced replacement target, or ``None`` when nothing is to be rewritten."""
        if result is None or self.mode != "enforce" or result.verdict.decision is not Decision.TRANSFORM:
            return None
        return result.transformed_policy_target

    def history(self, session: Any) -> list[dict[str, Any]]:
        state = getattr(session, "state", None)
        if isinstance(state, dict):
            return state.setdefault(HISTORY_KEY, [])
        return self._history

    def middleware(self) -> list[Any]:
        return [_AgentGate(self), _ChatGate(self), _ToolGate(self)]


def harness_mcp_before_call(
    *,
    agent_name: str,
    allowed_tools: Collection[str] | None = None,
    audit: AuditTrail | Path | str | None = None,
) -> Callable[[str, str, dict[str, Any]], Awaitable[str | None]]:
    """Build the fail-closed ACS hook used by harness MCP calls."""
    gov = Governance("harness", agent_name=agent_name, audit=audit)
    case_allow = None if allowed_tools is None else frozenset(allowed_tools)

    async def before_call(server: str, tool: str, args: dict[str, Any]) -> str | None:
        if case_allow is not None and tool not in case_allow:
            return "tool is outside the frozen case exposure"
        name = f"{server}.{tool}"
        try:
            await gov.check("pre_tool_call", {"call": {"name": name, "arguments": args}})
        except AgentControlInterruption as exc:
            return _refusal(exc)
        except Exception:  # noqa: BLE001 - MCP governance must fail closed
            return "harness governance adapter failure"
        return None

    return before_call


def _last_user(messages: list[Message]) -> int | None:
    return next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), None)


class _AgentGate(AgentMiddleware):
    def __init__(self, gov: Governance) -> None:
        self.gov = gov

    async def _output(self, response: AgentResponse) -> AgentResponse:
        try:
            res = await self.gov.check("output", {"output": {"text": response.text}})
        except AgentControlInterruption as exc:
            return AgentResponse(messages=[Message("assistant", [_refusal(exc)])])
        new = self.gov.transformed(res)
        if isinstance(new, str):
            return AgentResponse(messages=[Message("assistant", [new])], response_id=response.response_id,
                                 usage_details=response.usage_details)
        return response

    async def process(self, context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
        i = _last_user(context.messages)
        text = context.messages[i].text if i is not None else ""
        try:
            new = self.gov.transformed(await self.gov.check("input", {"input": {"text": text}}))
        except AgentControlInterruption as exc:
            context.result = AgentResponse(messages=[Message("assistant", [_refusal(exc)])])
            raise MiddlewareTermination(str(exc), result=context.result) from None
        if isinstance(new, str) and i is not None:
            context.messages[i] = Message("user", [new])
        if context.stream:
            context.stream_result_transforms.append(self._output)
            await call_next()
            return
        await call_next()
        if isinstance(context.result, AgentResponse):
            context.result = await self._output(context.result)


class _ChatGate(ChatMiddleware):
    def __init__(self, gov: Governance) -> None:
        self.gov = gov

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        model: dict[str, Any] = {"id": self.gov.model} if self.gov.model else {}
        if model and self.gov.allowed_models is not None:
            model["allowed"] = sorted(self.gov.allowed_models)
        try:
            await self.gov.check("pre_model_call", {"model": model, "messages": len(context.messages)})
        except AgentControlInterruption as exc:
            context.result = ChatResponse(messages=[Message("assistant", [_refusal(exc)])])
            raise MiddlewareTermination(str(exc), result=context.result) from None
        await call_next()
        if context.stream or not isinstance(context.result, ChatResponse):
            return
        try:
            res = await self.gov.check("post_model_call", {"response": {"text": context.result.text}})
        except AgentControlInterruption as exc:
            context.result = ChatResponse(messages=[Message("assistant", [_refusal(exc)])])
            raise MiddlewareTermination(str(exc), result=context.result) from None
        new = self.gov.transformed(res)
        if isinstance(new, str):
            context.result = ChatResponse(messages=[Message("assistant", [new])])


class _ToolGate(FunctionMiddleware):
    def __init__(self, gov: Governance) -> None:
        self.gov = gov

    async def process(self, context: FunctionInvocationContext,
                      call_next: Callable[[], Awaitable[None]]) -> None:
        name, args = context.function.name, json_args(context.arguments)
        call_id = (context.metadata or {}).get("call_id")
        hist = self.gov.history(context.session)
        step = {"kind": "tool_call", "tool": name, "call_id": call_id, "args": args}
        try:
            res = await self.gov.check("pre_tool_call", {"call": {"name": name, "arguments": args},
                                                         "history": hist[-HISTORY_MAX:]})
        except AgentControlInterruption as exc:
            hist.append({**step, "status": "blocked"})
            context.result = f"ERROR: {_refusal(exc)}"
            raise MiddlewareTermination(str(exc)) from None
        new = self.gov.transformed(res)
        if isinstance(new, dict):
            args = new
            context.arguments = (type(context.arguments).model_validate(new)
                                 if isinstance(context.arguments, BaseModel) else new)
        hist.append({**step, "args": args})
        try:
            await call_next()
        except Exception:
            hist.append({"kind": "tool_result", "tool": name, "call_id": call_id, "status": "error"})
            raise
        result = structured_result(context.result)
        hist.append({"kind": "tool_result", "tool": name, "call_id": call_id, "result": result,
                     "status": "ok"})
        del hist[:-HISTORY_MAX]
        try:
            post = await self.gov.check("post_tool_call", {"call": {"name": name, "arguments": args},
                                                           "result": result})
        except AgentControlInterruption as exc:
            context.result = f"ERROR: {_refusal(exc)}"
            raise MiddlewareTermination(str(exc)) from None
        if (new := self.gov.transformed(post)) is not None:
            context.result = new


def govern(agent: Any, policy: str = "meta_agents", **kwargs: Any) -> Any:
    """Install :class:`Governance` middleware (outermost) on a built agent; idempotent."""
    props = agent.additional_properties
    if GOVERNED_KEY in props:
        return agent
    kwargs.setdefault("agent_name", getattr(agent, "name", None) or "agent")
    gov = Governance(policy, **kwargs)
    agent.middleware = [*gov.middleware(), *(agent.middleware or [])]
    props[GOVERNED_KEY] = {"policy": policy, "mode": gov.mode, "agent_did": gov.did}
    return agent


def governed_agent(*args: Any, policy: str = "meta_agents", governance: Mapping[str, Any] | None = None,
                   **kwargs: Any) -> Agent:
    """``agent_framework.Agent(*args, **kwargs)`` under ``policy``."""
    return govern(Agent(*args, **kwargs), policy, **dict(governance or {}))


def governed_declarative(factory: Any, doc: Mapping[str, Any], *, policy: str = "meta_agents",
                         governance: Mapping[str, Any] | None = None) -> Any:
    """``AgentFactory.create_agent_from_dict(doc)`` under ``policy``."""
    return govern(factory.create_agent_from_dict(doc), policy, **dict(governance or {}))


def governed_harness_agent(*args: Any, policy: str = "harness",
                           governance: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
    """``agent_framework.create_harness_agent(*args, **kwargs)`` under ``policy``."""
    from agent_framework import create_harness_agent

    return govern(create_harness_agent(*args, **kwargs), policy, **dict(governance or {}))
