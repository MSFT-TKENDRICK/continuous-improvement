"""Shared fixtures for guard tests: real MAF Agent + FakeChatClient, real ``ci_lab.rules`` engine
(spy wrapper for failure injection), order-support tools with call counters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from agent_framework import (
    Agent,
    ChatResponse,
    ChatResponseUpdate,
    ResponseStream,
    tool,
)

from ci_lab import rules
from ci_lab.guards import install_guards
from ci_lab.guards.domains.order_support import (
    EXTRACTORS,
    SEED_RULES,
    TOOL_POLICIES,
    verify_identity_tool,
)
from ci_lab.testing import FakeChatClient
from order_support import tools as os_tools


class SpyEngine:
    """The real engine, recording every view it evaluates; ``fail`` injects evaluation errors."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.views: list[Any] = []

    def evaluate(self, bundle: Any, view: Any, *, on: str) -> list[Any]:
        self.views.append(view)
        if self.fail:
            raise RuntimeError("engine exploded")
        return rules.evaluate(bundle, view, on=on)

    def redact(self, text: str, matches: Sequence[Any], bundle: Any) -> str:
        return rules.redact(text, matches, bundle)


class StreamingFakeChatClient(FakeChatClient):
    """FakeChatClient that also streams (one update per content)."""

    def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any) -> Any:
        if not stream:
            return self._respond(messages, options)

        async def updates() -> Any:
            response = await self._respond(messages, options)
            for m in response.messages:
                for c in m.contents:
                    yield ChatResponseUpdate(contents=[c], role=m.role, model=self.model)

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)


class Tools:
    """Order-support tools as MAF FunctionTools (JSON str results) with call logs."""

    def __init__(self, refund_delay: float = 0.0) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.active_side_effects = 0
        self.max_parallel_side_effects = 0
        self.refund_delay = refund_delay

    def _log(self, name: str, **args: Any) -> None:
        self.calls.append((name, args))

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    def all(self) -> list[Any]:
        def lookup_order(order_id: str) -> str:
            self._log("lookup_order", order_id=order_id)
            return json.dumps(os_tools.lookup_order(order_id))

        def search_kb(query: str) -> str:
            self._log("search_kb", query=query)
            return json.dumps(os_tools.search_kb(query))

        async def issue_refund(order_id: str, amount: float) -> str:
            self.active_side_effects += 1
            self.max_parallel_side_effects = max(self.max_parallel_side_effects, self.active_side_effects)
            try:
                await asyncio.sleep(self.refund_delay)
                self._log("issue_refund", order_id=order_id, amount=amount)
                return json.dumps(os_tools.issue_refund(order_id, amount))
            finally:
                self.active_side_effects -= 1

        def escalate_to_human(reason: str, order_id: str | None = None) -> str:
            self._log("escalate_to_human", reason=reason, order_id=order_id)
            return json.dumps(os_tools.escalate_to_human(reason, order_id))

        verify = verify_identity_tool()

        def verify_identity(order_id: str, full_name: str, email_or_phone: str) -> str:
            self._log("verify_identity", order_id=order_id)
            return verify.func(order_id, full_name, email_or_phone)

        return [tool(lookup_order), tool(search_kb), tool(issue_refund), tool(escalate_to_human),
                tool(verify_identity, name="verify_identity", description=verify.description)]


def seed_bundle() -> Any:
    return rules.load_bundle([SEED_RULES], [EXTRACTORS])


def make_agent(script: Sequence[Any], *, mode: str | None = "enforce", bundle: Any = None, engine: Any = None,
               sink: Any = None, streaming: bool = False, tools: Tools | None = None, **kw: Any) -> Any:
    client = (StreamingFakeChatClient if streaming else FakeChatClient)(script)
    tools = tools or Tools()
    mw = install_guards(bundle=seed_bundle() if bundle is None else bundle, tool_policies=TOOL_POLICIES,
                        sink=sink, mode_override=mode, engine=engine or SpyEngine(), **kw)
    agent = Agent(client=client, tools=tools.all(), middleware=mw)
    return agent, client, tools, mw.runtime
