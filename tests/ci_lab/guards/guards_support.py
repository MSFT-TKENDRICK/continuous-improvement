"""Shared fixtures for guard tests using neutral tools and rule data."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
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
from ci_lab.testing import FakeChatClient

FIXTURES = Path(__file__).with_name("fixtures")
EXTRACTORS = FIXTURES / "extractors.yaml"
SEED_RULES = FIXTURES / "rules.yaml"
TOOL_POLICIES = {
    "inspect_item": False,
    "search_docs": False,
    "verify_access": False,
    "apply_change": True,
    "escalate": True,
}
ITEMS = {
    "item-a": {"item_id": "item-a", "approved": True, "limit": 100.0,
               "owner": "reviewer@example.com", "phone": "+1-206-555-0141"},
    "item-b": {"item_id": "item-b", "approved": False, "limit": 5.0,
               "owner": "other@example.com", "phone": "+1-503-555-0177"},
}


def verify_access(item_id: str, principal: str) -> dict[str, Any]:
    return {"verified": item_id in ITEMS and principal.strip().lower() == "reviewer",
            "item_id": item_id}


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
    """Neutral MAF FunctionTools with call logs."""

    def __init__(self, action_delay: float = 0.0) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.active_side_effects = 0
        self.max_parallel_side_effects = 0
        self.action_delay = action_delay

    def _log(self, name: str, **args: Any) -> None:
        self.calls.append((name, args))

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    def all(self) -> list[Any]:
        def inspect_item(item_id: str) -> str:
            self._log("inspect_item", item_id=item_id)
            return json.dumps(ITEMS.get(item_id, {"error": "not_found", "item_id": item_id}))

        def search_docs(query: str) -> str:
            self._log("search_docs", query=query)
            return json.dumps({"query": query, "text": "Use the approved workflow."})

        async def apply_change(item_id: str, amount: float) -> str:
            self.active_side_effects += 1
            self.max_parallel_side_effects = max(self.max_parallel_side_effects, self.active_side_effects)
            try:
                await asyncio.sleep(self.action_delay)
                self._log("apply_change", item_id=item_id, amount=amount)
                return json.dumps({"change_id": f"change-{item_id}", "applied": True})
            finally:
                self.active_side_effects -= 1

        def escalate(reason: str, item_id: str | None = None) -> str:
            self._log("escalate", reason=reason, item_id=item_id)
            return json.dumps({"escalated": True, "item_id": item_id})

        def verify_access(item_id: str, principal: str) -> str:
            self._log("verify_access", item_id=item_id)
            return json.dumps(globals()["verify_access"](item_id, principal))

        return [tool(inspect_item), tool(search_docs), tool(apply_change), tool(escalate),
                tool(verify_access)]


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
