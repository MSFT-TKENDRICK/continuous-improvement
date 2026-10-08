"""The ASSERT wrapper subprocess joins the harness trace and wraps each case in a ``ci.case`` span."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_CASE,
    ATTR_EXPERIMENT,
    ATTR_ROLLOUT,
    ATTR_SPLIT,
    ATTR_TRIAL,
    ATTR_VARIANT,
    SPAN_CASE,
    RolloutKey,
)
from ci_lab.testing import Call
from order_support import agent, assert_wrapper, cli

IDENTITY_ENVS = (assert_wrapper.EXPERIMENT_ENV, assert_wrapper.VARIANT_ENV, assert_wrapper.TRIAL_ENV,
                 assert_wrapper.SPLIT_ENV, obs.TRACEPARENT_ENV, "TRACESTATE")


@pytest.fixture
def clean_env(monkeypatch):
    for name in IDENTITY_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(assert_wrapper, "_attach_token", None)
    yield monkeypatch
    token = assert_wrapper._attach_token
    if token is not None:
        otel_context.detach(token)


def _fake_inference() -> types.ModuleType:
    module = types.ModuleType("fake_inference")

    async def _run_prompt_test_case(*, test_case, **_):
        return agent.chat(test_case["seed"]["message"])

    async def _run_scenario_test_case(*, test_case, **_):
        return test_case["test_case_id"]

    module._run_prompt_test_case = _run_prompt_test_case
    module._run_scenario_test_case = _run_scenario_test_case
    return module


def test_wrapper_env_carries_trace_context_and_identity(clean_env, captured):
    with obs.span("parent"):
        env = assert_wrapper.wrapper_env(experiment_id="exp1", variant="arm-a", trial=2, split="train",
                                         base={"KEEP": "1"})
        trace_id, span_id = obs.current_ids()
    assert env["KEEP"] == "1" and env[obs.TRACEPARENT_ENV].startswith(f"00-{trace_id}-{span_id}-")
    assert env[assert_wrapper.EXPERIMENT_ENV] == "exp1" and env[assert_wrapper.VARIANT_ENV] == "arm-a"
    assert env[assert_wrapper.TRIAL_ENV] == "2" and env[assert_wrapper.SPLIT_ENV] == "train"
    assert assert_wrapper.EXPERIMENT_ENV not in assert_wrapper.wrapper_env(base={})


def test_command_runs_the_cli_module():
    argv = assert_wrapper.command("evals/x.yaml", "--concurrency", "1")
    assert argv == [sys.executable, "-m", "order_support.cli", "run", "evals/x.yaml", "--concurrency", "1"]


def test_case_attributes_derive_rollout_id_from_contracts():
    env = {assert_wrapper.EXPERIMENT_ENV: "exp1", assert_wrapper.VARIANT_ENV: "arm-a",
           assert_wrapper.TRIAL_ENV: "3", assert_wrapper.SPLIT_ENV: "dev"}
    attrs = assert_wrapper.case_attributes("case-7", env)
    assert attrs[ATTR_CASE] == "case-7" and attrs[ATTR_TRIAL] == 3 and attrs[ATTR_SPLIT] == "dev"
    assert attrs[ATTR_EXPERIMENT] == "exp1" and attrs[ATTR_VARIANT] == "arm-a"
    assert attrs[ATTR_ROLLOUT] == RolloutKey("exp1", "arm-a", "case-7", 3).rollout_id
    bare = assert_wrapper.case_attributes("case-7", {})
    assert bare[ATTR_TRIAL] == 0 and bare[ATTR_ROLLOUT] is None and bare[ATTR_SPLIT] is None


def test_install_wraps_case_runners_idempotently(clean_env):
    module = _fake_inference()
    original = module._run_prompt_test_case
    assert_wrapper.install([module])
    wrapped = module._run_prompt_test_case
    assert wrapped is not original and wrapped.__wrapped__ is original
    assert_wrapper.install([module])
    assert module._run_prompt_test_case is wrapped


def test_case_span_parents_agent_spans_and_joins_parent_trace(clean_env, captured, use_client):
    use_client([Call("lookup_order", {"order_id": "NW-10007"}, "t1")], "Shipped.")
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("harness.parent") as parent:
        clean_env.setenv(obs.TRACEPARENT_ENV, obs.child_env({})[obs.TRACEPARENT_ENV])
        parent_ctx = parent.get_span_context()
    clean_env.setenv(assert_wrapper.EXPERIMENT_ENV, "exp1")
    clean_env.setenv(assert_wrapper.VARIANT_ENV, "arm-a")
    clean_env.setenv(assert_wrapper.SPLIT_ENV, "train")
    module = _fake_inference()
    assert_wrapper.install([module])
    assert assert_wrapper._attach_token is not None

    case = {"test_case_id": "c1", "seed": {"message": "Where is NW-10007?"}}
    assert asyncio.run(module._run_prompt_test_case(test_case=case)) == "Shipped."

    spans = captured()
    by_name = {s.name: s for s in spans}
    case_span = by_name[SPAN_CASE]
    assert case_span.trace_id == f"{parent_ctx.trace_id:032x}"
    assert case_span.parent_span_id == f"{parent_ctx.span_id:016x}"
    assert case_span.attributes[ATTR_CASE] == "c1" and case_span.attributes[ATTR_SPLIT] == "train"
    assert case_span.attributes[ATTR_ROLLOUT] == RolloutKey("exp1", "arm-a", "c1", 0).rollout_id
    assert by_name["agent.chat"].parent_span_id == case_span.span_id
    assert {s.trace_id for s in spans} == {case_span.trace_id}


def test_cmd_run_installs_wrapper_before_assert(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "stage_replay_inference_set", lambda *a, **k: None)
    for name in (cli.TIMEOUT_ENV, cli.TEST_SET_CONCURRENCY_ENV, cli.ASSERT_CONCURRENCY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(assert_wrapper, "install", lambda *a, **k: order.append("install"))
    from assert_ai.cli import cli as assert_cli

    monkeypatch.setattr(assert_cli, "main", lambda args, **_: order.append("assert"))
    path = cli.REPO_ROOT / "evals" / "assert" / "grounding" / "eval_config.yaml"
    assert cli.main(["run", str(path)]) == 0
    assert order == ["install", "assert"]
