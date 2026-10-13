"""Shared fixtures for ``ci_lab.rules`` golden tests (harness-agent tool names, §13.7)."""

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

    def pending(self, tool: str, **args: Any) -> TrajectoryStep:
        """A tool_call step that is NOT appended (the call under evaluation)."""
        return TrajectoryStep(i=len(self.steps), kind="tool_call", tool=tool, call_id="pending", args=args)

    def pending_response(self, text: str) -> TrajectoryStep:
        return TrajectoryStep(i=len(self.steps), kind="response", text=text)

    def inspect(self, resource_id: str, total: float = 50.0, eligible: bool = True, exceeded: bool = False,
               status: str = "delivered", pii: bool = False) -> TrajectoryStep:
        res: dict[str, Any] = {"resource_id": resource_id, "status": status, "total": total,
                               "edit_allowed": eligible, "edit_limit_exceeded": exceeded}
        if pii:
            res["metadata"] = {"email": "jo@example.com", "phone": "+1 555 010 0000"}
        return self.ok("read_file", res, resource_id=resource_id)

    def authorize(self, resource_id: str, verified: bool = True) -> TrajectoryStep:
        return self.ok("verify_access", {"verified": verified, "resource_id": resource_id},
                       resource_id=resource_id, full_name="Jo Doe", email_or_phone="x")


@pytest.fixture
def tb() -> TB:
    return TB()


@pytest.fixture(scope="session")
def seeds():
    from ci_lab.rules import load_bundle

    return load_bundle([SEEDS], [EXTRACTORS])


IDV = {"flag": "access_verified", "tool": "verify_access", "result_path": "result.verified",
       "subject": "args.resource_id"}
_ON = {"R1": "tool_call", "R2": "tool_call", "R3": "response", "R4": "trajectory"}


def make_rule(**kw: Any):
    from ci_lab.rulespec import RuleSpec

    rung = kw.get("rung", "R2")
    base: dict[str, Any] = {"id": "t.rule", "version": 1, "rung": rung, "on": _ON[rung], "target": "write_file",
                            "action": "block" if rung == "R2" else "warn", "template": "count.exceeded",
                            "slots": {"tool": "write_file"}}
    base.update(kw)
    return RuleSpec.model_validate(base)


@pytest.fixture
def mk():
    """``mk(rule_kwargs..., extractors=[...])`` -> compiled Bundle (default extractor: access_verified)."""
    from ci_lab.rules import build_bundle
    from ci_lab.rulespec import ExtractorSpec

    def _mk(*rules: dict[str, Any], extractors: list[dict[str, Any]] | None = None):
        ex = [ExtractorSpec.model_validate(e) for e in (extractors if extractors is not None else [IDV])]
        return build_bundle([make_rule(**{"id": f"t.rule{n}", **r}) for n, r in enumerate(rules)], ex)

    return _mk


@pytest.fixture
def fires():
    """``fires(bundle, steps, pending, on=...)`` -> sorted ids of matching rules."""
    from ci_lab.rules import evaluate
    from ci_lab.rulespec import GuardView

    def _fires(bundle, steps, pending, on="tool_call"):
        return [m.rule.id for m in evaluate(bundle, GuardView(steps=tuple(steps), pending=pending), on=on)]

    return _fires
