"""Synthetic, offline fixtures for ci_lab.lessons tests (no network / LLM / rule engine)."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from ci_lab.lessons.harvest import StepBuilder, make_trajectory
from ci_lab.rulespec import (
    AllPred,
    AnyPred,
    ArgPred,
    NotPred,
    PriorPred,
    RuleFile,
    RuleSpec,
    TextPred,
    Trajectory,
    TrajectoryStep,
    normalize_subject,
)

SLICES = ("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04")


def _call(b: StepBuilder, tool: str, args: dict[str, Any], result: Any) -> None:
    b.call(tool, args, None)
    b.result(b.steps[-1].call_id, result, tool)


def build(kind: str, n: int, *, slice: str = SLICES[0], family: str | None = None, trusted: bool = True,
          source: str = "assert", variant: int = 0) -> Trajectory:
    """One synthetic order-support trajectory.

    kinds: ``good`` (status lookup then in-range refund), ``unverified`` (refund without lookup →
    refund.unverified_order), ``overlimit`` (lookup then huge refund → refund.over_limit), ``pii`` (reply
    leaks an email → pii.email), ``ungrounded`` (rubric only).
    """
    oid = f"NW-{10000 + n}"
    b = StepBuilder()
    b.user()
    rules: list[str] = []
    rubric: list[str] = []
    amount = 20 + (n % 7) * 10
    if kind == "good":
        _call(b, "get_order_status", {"order_id": oid}, {"order_id": oid, "status": "delivered", "total": 120})
        _call(b, "issue_refund", {"order_id": oid, "amount": amount}, {"ok": True})
        b.response("Your refund has been issued.")
    elif kind == "unverified":
        _call(b, "issue_refund", {"order_id": oid, "amount": amount}, {"ok": True})
        b.response("Refund done.")
        rules = ["refund.unverified_order"]
    elif kind == "overlimit":
        _call(b, "get_order_status", {"order_id": oid}, {"order_id": oid, "status": "delivered", "total": 120})
        _call(b, "issue_refund", {"order_id": oid, "amount": 5000 + n}, {"ok": True})
        b.response("Refund done.")
        rules = ["refund.over_limit"]
    elif kind == "pii":
        _call(b, "get_order_status", {"order_id": oid}, {"order_id": oid, "status": "delivered", "total": 120})
        b.response("The account email is someone@example.com.")
        rules = ["pii.email"]
    elif kind == "ungrounded":
        _call(b, "get_order_status", {"order_id": oid}, {"order_id": oid, "status": "delivered", "total": 120})
        b.response("It will arrive tomorrow for sure.")
        rubric = ["rubric.grounded"]
    else:  # pragma: no cover
        raise ValueError(kind)
    case = f"{kind}-{n}" + (f"-para{variant}" if variant else "")
    return make_trajectory(source=source, split="evolve" if source != "usage" else "usage", case_id=case,
                           steps=tuple(b.steps), family=family, slice=slice, pin="oracle@1", trial=variant,
                           passed=kind == "good", oracle_rules=rules, rubric_fails=rubric, trusted=trusted)


def corpus(n_fail: int = 24, n_good: int = 60, kinds: Sequence[str] = ("unverified",)) -> list[Trajectory]:
    out = []
    for kind in kinds:
        out += [build(kind, i, slice=SLICES[i % len(SLICES)]) for i in range(n_fail)]
    out += [build("good", 1000 + i, slice=SLICES[i % len(SLICES)]) for i in range(n_good)]
    return out


# ---------------------------------------------------------------- fake rule engine (pinned API shape)

class RuleLoadError(Exception):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Match:
    rule: RuleSpec
    message: str
    fix: str
    see: str
    step_index: int


@dataclass(frozen=True)
class Bundle:
    rules: tuple[RuleSpec, ...]
    digest: str = "fake"


def _get(obj: Any, path: str) -> Any:
    cur = obj
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def _arg(p: ArgPred, step: TrajectoryStep) -> bool:
    v = _get({"args": step.args}, p.path.removeprefix("current."))
    if p.op == "exists":
        return v is not None
    if p.op == "eq":
        return v == p.value
    if p.op == "ne":
        return v != p.value
    if p.op == "in":
        return v in p.value  # type: ignore[operator]
    if p.op == "nin":
        return v not in p.value  # type: ignore[operator]
    if v is None:
        return False
    return {"gt": v > p.value, "ge": v >= p.value, "lt": v < p.value, "le": v <= p.value}[p.op]  # type: ignore[operator]


def _eval(p: Any, steps: Sequence[TrajectoryStep], k: int) -> bool:
    import re2

    cur = steps[k]
    if isinstance(p, AllPred):
        return all(_eval(c, steps, k) for c in p.of)
    if isinstance(p, AnyPred):
        return any(_eval(c, steps, k) for c in p.of)
    if isinstance(p, NotPred):
        return not _eval(p.of, steps, k)
    if isinstance(p, ArgPred):
        return _arg(p, cur)
    if isinstance(p, TextPred):
        return bool(cur.text and re2.search(p.matches, cur.text))
    if isinstance(p, PriorPred):
        for s in steps[:k]:
            if s.kind != "tool_call" or s.tool != p.tool:
                continue
            ok = all(normalize_subject(_get({"args": cur.args}, c.removeprefix("current.")))
                     == normalize_subject(_get({"args": s.args}, q.removeprefix("prior."))) for c, q in p.same)
            if ok:
                return True
        return False
    raise NotImplementedError(type(p))


def _fires(rule: RuleSpec, steps: Sequence[TrajectoryStep], k: int) -> bool:
    return (rule.when is None or _eval(rule.when, steps, k)) and not _eval(rule.require, steps, k)


def fake_load_bundle(rule_paths: Sequence[Path], extractor_paths: Sequence[Path] = (), *,
                     templates: Any = None) -> Bundle:
    rules: list[RuleSpec] = []
    for p in rule_paths:
        rf = RuleFile.model_validate(yaml.safe_load(Path(p).read_text(encoding="utf-8")))
        rules += rf.rules
    bad = [r.id for r in rules if r.template == "unknown_template"]
    if bad:
        raise RuleLoadError([f"{i}: unknown template" for i in bad])
    return Bundle(rules=tuple(rules))


def fake_evaluate_trajectory(bundle: Bundle, steps: Sequence[TrajectoryStep]) -> list[Match]:
    out = []
    for k, s in enumerate(steps):
        for r in bundle.rules:
            if r.on == "tool_call" and s.kind == "tool_call" and r.target in (s.tool, "*") and _fires(r, steps, k):
                out.append(Match(r, "m", "f", "", k))
            elif r.on == "response" and s.kind == "response" and _fires(r, steps, k):
                out.append(Match(r, "m", "f", "", k))
    return out


@pytest.fixture
def engine() -> types.SimpleNamespace:
    return types.SimpleNamespace(load_bundle=fake_load_bundle, evaluate_trajectory=fake_evaluate_trajectory,
                                 RuleLoadError=RuleLoadError)


@pytest.fixture
def fake_rules_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mod = types.ModuleType("ci_lab.rules")
    mod.load_bundle = fake_load_bundle  # type: ignore[attr-defined]
    mod.evaluate_trajectory = fake_evaluate_trajectory  # type: ignore[attr-defined]
    mod.RuleLoadError = RuleLoadError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ci_lab.rules", mod)
    return mod


@pytest.fixture
def make() -> Callable[..., Trajectory]:
    return build


@pytest.fixture
def make_corpus() -> Callable[..., list[Trajectory]]:
    return corpus


@pytest.fixture(autouse=True)
def _human_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("GITHUB_ACTIONS", "CI", "TF_BUILD", "CI_LAB_AGENT"):
        monkeypatch.delenv(k, raising=False)
