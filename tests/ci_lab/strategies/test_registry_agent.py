import asyncio
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_EXPERIMENT,
    ATTR_STRATEGY,
    ATTR_VARIANT,
    SPAN_OPTIMIZER,
    STRATEGIES,
    ArmContext,
    ArmDirective,
    Edit,
    Profile,
)
from ci_lab.strategies import (
    AgentStrategy,
    EditBudgetExceeded,
    GepaStrategy,
    SkillOptStrategy,
    UnknownStrategy,
    get_strategy,
    register_strategy,
)


@pytest.fixture
def spans(monkeypatch):
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    return exp


def ctx(tmp_path, budget=1, strategy="agent", focus=()):
    return ArmContext("exp-1", ArmDirective("a1", strategy, tuple(focus), budget), tmp_path, "base", [],
                      Profile.FAKE, tmp_path / "run")


def edit(i=0):
    return Edit("prompt", f"h{i}", ("p.md",), f"sha{i}")


def test_registry_builds_each_strategy_from_shared_deps(domain):
    async def proposer(c):
        return []

    deps = {"proposer": proposer, "domain": domain, "lm": object(), "committer": lambda *a: "x"}
    built = {n: get_strategy(n, **deps) for n in ("agent", "gepa", "skillopt")}
    assert isinstance(built["agent"], AgentStrategy) and built["agent"].proposer is proposer
    assert isinstance(built["gepa"], GepaStrategy) and built["gepa"].domain is domain
    assert isinstance(built["skillopt"], SkillOptStrategy) and built["skillopt"].lm is deps["lm"]
    assert {n: s.name for n, s in built.items()} == {n: n for n in built}


def test_registry_external_guard_strategy(monkeypatch):
    import ci_lab.strategies as reg
    from ci_lab.lessons_arm.strategy import GuardStrategy

    monkeypatch.setattr(reg, "_FACTORIES", dict(reg._FACTORIES))
    reg._FACTORIES.pop("guard", None)
    assert "guard" in STRATEGIES and reg.EXTERNAL["guard"] == "ci_lab.lessons_arm.strategy"
    built = get_strategy("guard", write_mode="shadow", proposer=None, lm=object())
    assert isinstance(built, GuardStrategy) and built.name == "guard"
    assert reg._FACTORIES["guard"] is GuardStrategy
    assert set(reg.available()) == set(STRATEGIES)

    class Guard:
        name = "guard"

        def __init__(self, *, bundle):
            self.bundle = bundle

    register_strategy("guard", Guard)
    assert get_strategy("guard", bundle="b", proposer=None).bundle == "b"
    with pytest.raises(UnknownStrategy):
        register_strategy("evolution", Guard)


def test_registry_external_import_failure(monkeypatch):
    import ci_lab.strategies as reg

    monkeypatch.setattr(reg, "_FACTORIES", {k: v for k, v in reg._FACTORIES.items() if k != "guard"})
    monkeypatch.setitem(reg.EXTERNAL, "guard", "ci_lab.no_such_module_xyz")
    with pytest.raises(UnknownStrategy, match="no_such_module_xyz"):
        get_strategy("guard")
    assert "guard" not in reg.available()


def test_registry_errors():
    with pytest.raises(UnknownStrategy):
        get_strategy("evolution")
    with pytest.raises(TypeError):
        get_strategy("agent")
    with pytest.raises(TypeError):
        get_strategy("gepa", lm=object())


def test_agent_strategy_span_and_budget(tmp_path, spans):
    seen = []

    async def proposer(c):
        seen.append(c)
        return [edit()]

    s = AgentStrategy(proposer)
    c = ctx(tmp_path)
    assert asyncio.run(s.propose(c)) == [edit()]
    assert seen == [c]
    (sp,) = spans.get_finished_spans()
    assert sp.name == SPAN_OPTIMIZER
    assert sp.attributes[ATTR_STRATEGY] == "agent"
    assert sp.attributes[ATTR_EXPERIMENT] == "exp-1" and sp.attributes[ATTR_VARIANT] == "a1"
    assert sp.attributes["ci.edits"] == 1


def test_agent_strategy_rejects_over_budget_and_bad_types(tmp_path, spans):
    async def two(c):
        return [edit(0), edit(1)]

    async def bad(c):
        return ["not an edit"]

    with pytest.raises(EditBudgetExceeded):
        asyncio.run(AgentStrategy(two).propose(ctx(tmp_path, budget=1)))
    assert asyncio.run(AgentStrategy(two).propose(ctx(tmp_path, budget=2))) == [edit(0), edit(1)]
    with pytest.raises(TypeError):
        asyncio.run(AgentStrategy(bad).propose(ctx(tmp_path)))
    statuses = [s.status.status_code for s in spans.get_finished_spans()]
    assert statuses == [StatusCode.ERROR, StatusCode.UNSET, StatusCode.ERROR]


def test_agent_zero_budget_skips_proposer(tmp_path):
    called = []

    async def proposer(c):
        called.append(1)
        return [edit()]

    assert asyncio.run(AgentStrategy(proposer).propose(ctx(tmp_path, budget=0))) == []
    assert called == []
    with pytest.raises(TypeError):
        AgentStrategy(None)


def test_paths_are_paths(tmp_path):
    assert isinstance(ctx(tmp_path).worktree, Path)


@pytest.mark.parametrize(("strategy", "files", "bad"), [
    ("gepa", ["harness/prompt/x.md", "src/order_support/harness/guards/a.yaml"],
     ["src/order_support/harness/guards/a.yaml"]),
    ("agent", ["harness\\guards\\a.yaml"], ["harness\\guards\\a.yaml"]),
    ("skillopt", ["harness/guardsx/a.md", "harness/guards"], []),
    ("guard", ["harness/guards/a.yaml", "harness/guards/a.yaml"], []),
    ("guard", ["harness/prompt/x.md", "harness/guards/BUNDLE.lock"],
     ["harness/prompt/x.md", "harness/guards/BUNDLE.lock"]),
])
def test_edit_scope_violations(strategy, files, bad):
    from ci_lab.strategies.base import edit_scope_violations

    assert edit_scope_violations(strategy, files) == bad


def test_evolve_cases_prefer_campaign_evolve_set(tmp_path):
    from ci_lab.contracts import FailureRecord
    from ci_lab.strategies.base import evolve_cases_for

    class D:
        def splits(self):
            return {"evolve": ["d1"], "heldout": ["h1"]}

    fail = FailureRecord("f1", "s", "c", (), {}, "")
    ctx = ArmContext("e", ArmDirective("v1", "gepa", ("prompt",)), tmp_path, "b", [fail], Profile.FAKE, tmp_path,
                     evolve_case_ids=("c1", "c2"))
    assert evolve_cases_for(ctx, explicit=["x"], domain=D()) == ["x"]
    assert evolve_cases_for(ctx, domain=D()) == ["c1", "c2"]
    ctx.evolve_case_ids = ()
    assert evolve_cases_for(ctx, domain=D()) == ["d1"]
    assert evolve_cases_for(ctx) == ["f1"]
