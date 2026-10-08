"""Shared fixtures for ``ci_lab.rules`` golden tests (order-support tool names, §13.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ci_lab.rulespec import TrajectoryStep

FIXTURES = Path(__file__).parent / "fixtures"
SEEDS = FIXTURES / "seeds.yaml"
EXTRACTORS = FIXTURES / "extractors.yaml"


class TB:
    """Tiny trajectory builder: ``call()`` / ``result()`` / ``respond()`` append typed steps."""

    def __init__(self) -> None:
        self.steps: list[TrajectoryStep] = []
        self._n = 0

    def _add(self, **kw: Any) -> TrajectoryStep:
        s = TrajectoryStep(i=len(self.steps), **kw)
        self.steps.append(s)
        return s

    def call(self, tool: str, status: str | None = None, call_id: str | None = None, **args: Any) -> TrajectoryStep:
        self._n += 1
        cid = call_id if call_id is not None else f"c{self._n}"
        return self._add(kind="tool_call", tool=tool, call_id=cid or None, args=args, status=status)

    def result(self, result: dict[str, Any] | None, status: str = "ok", tool: str | None = None,
               call_id: str | None = None) -> TrajectoryStep:
        last = next(s for s in reversed(self.steps) if s.kind == "tool_call")
        cid = call_id if call_id is not None else last.call_id
        return self._add(kind="tool_result", tool=tool or last.tool, call_id=cid or None, result=result, status=status)

    def ok(self, tool: str, result: dict[str, Any], **args: Any) -> TrajectoryStep:
        self.call(tool, **args)
        return self.result(result)

    def respond(self, text: str) -> TrajectoryStep:
        return self._add(kind="response", text=text)

    def lookup(self, order_id: str, total: float = 50.0, eligible: bool = True, exceeded: bool = False,
               status: str = "delivered", pii: bool = False) -> TrajectoryStep:
        res: dict[str, Any] = {"order_id": order_id, "status": status, "total": total,
                               "refund_eligible": eligible, "refund_limit_exceeded": exceeded}
        if pii:
            res["customer"] = {"email": "jo@example.com", "phone": "+1 555 010 0000"}
        return self.ok("lookup_order", res, order_id=order_id)

    def verify(self, order_id: str, verified: bool = True) -> TrajectoryStep:
        return self.ok("verify_identity", {"verified": verified, "order_id": order_id},
                       order_id=order_id, full_name="Jo Doe", email_or_phone="x")


@pytest.fixture
def tb() -> TB:
    return TB()


@pytest.fixture(scope="session")
def seeds():
    from ci_lab.rules import load_bundle

    return load_bundle([SEEDS], [EXTRACTORS])
