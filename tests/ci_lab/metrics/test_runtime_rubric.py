"""ci_lab.metrics.runtime / rubric / maf: RunMeter, runtime_summary, metric checks, MAF metering middleware."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from ci_lab.contracts import TaskScore
from ci_lab.metrics import (
    METRIC_NAMES,
    RunMeter,
    runtime_summary,
    score_metric,
    validate_metric_check,
)
from ci_lab.metrics.runtime import percentile


def test_run_meter_counts_and_times() -> None:
    m = RunMeter()
    assert m.wall_ms == 0.0
    with m:
        m.llm_call(tokens_in=10, tokens_out=3)
        m.llm_call()
        m.tool_call("lookup")
        m.tool_call("lookup")
        time.sleep(0.01)
    frozen = m.wall_ms
    assert frozen >= 5.0
    time.sleep(0.005)
    assert m.wall_ms == frozen
    assert m.as_dict() == {"wall_ms": frozen, "llm_calls": 2.0, "tool_calls": 2.0, "tokens_in": 10.0,
                           "tokens_out": 3.0, "tokens": 13.0}
    assert m.tool_names["lookup"] == 2


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([5], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([0, 10], 95) == pytest.approx(9.5)


def test_runtime_summary() -> None:
    assert runtime_summary([]) == dict.fromkeys(
        ("wall_ms_mean", "wall_ms_p50", "wall_ms_p95", "tokens_per_task", "llm_calls_per_task",
         "tool_calls_per_task", "n"), 0.0)
    scores = [TaskScore("a", 0, "s", 1.0, tokens_in=10, tokens_out=10, wall_ms=100.0, llm_calls=2, tool_calls=1),
              TaskScore("b", 0, "s", 0.0, tokens_in=0, tokens_out=40, wall_ms=300.0, llm_calls=4, tool_calls=3),
              TaskScore("c", 0, "s", None, tokens_in=999, wall_ms=1e6, llm_calls=99)]
    got = runtime_summary(scores)
    assert got == pytest.approx({"wall_ms_mean": 200.0, "wall_ms_p50": 200.0, "wall_ms_p95": 290.0,
                                 "tokens_per_task": 30.0, "llm_calls_per_task": 3.0, "tool_calls_per_task": 2.0,
                                 "n": 2.0})


LE = {"kind": "metric", "metric": "wall_ms", "op": "le", "target": 100, "worst": 300}
GE = {"kind": "metric", "metric": "output_lines", "op": "ge", "target": 10, "worst": 0}


@pytest.mark.parametrize(("check", "value", "score"), [
    (LE, 50, 1.0), (LE, 100, 1.0), (LE, 200, 0.5), (LE, 300, 0.0), (LE, 1000, 0.0),
    (GE, 20, 1.0), (GE, 10, 1.0), (GE, 5, 0.5), (GE, 0, 0.0), (GE, -3, 0.0)])
def test_score_metric(check: dict, value: float, score: float) -> None:
    assert score_metric(check, value) == pytest.approx(score)


@pytest.mark.parametrize("bad", [
    {**LE, "kind": "assert"}, {**LE, "metric": "vibes"}, {**LE, "op": "lt"}, {**LE, "target": 300, "worst": 100},
    {**LE, "target": 100, "worst": 100}, {**GE, "target": 0, "worst": 10}, {**LE, "target": "1"},
    {**LE, "worst": float("nan")}, {**LE, "target": True}, {**LE, "extra": 1},
    {k: v for k, v in LE.items() if k != "worst"}, ["kind", "metric"]])
def test_bad_metric_checks_raise(bad) -> None:
    with pytest.raises(ValueError):
        validate_metric_check(bad)
    with pytest.raises(ValueError):
        score_metric(bad, 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "3", True])
def test_bad_metric_value_raises(value) -> None:
    with pytest.raises(ValueError):
        score_metric(LE, value)  # type: ignore[arg-type]


def test_metric_names() -> None:
    assert METRIC_NAMES == ("wall_ms", "tokens", "tokens_in", "tokens_out", "llm_calls", "tool_calls",
                            "output_chars", "output_lines", "complexity_delta", "edit_lines")


def test_maf_metering_middleware() -> None:
    from agent_framework import ChatResponse

    from ci_lab.metrics.maf import metering_middleware, usage_tokens

    meter = RunMeter()
    chat, fn = metering_middleware(meter)

    async def go() -> None:
        ctx = SimpleNamespace(result=None)

        async def respond() -> None:
            ctx.result = ChatResponse(messages=[], usage_details={"input_token_count": 7, "output_token_count": 2})
        await chat.process(ctx, respond)

        async def noop() -> None:
            return None
        await chat.process(SimpleNamespace(result=None), noop)
        await fn.process(SimpleNamespace(function=SimpleNamespace(name="lookup")), noop)

    with meter:
        asyncio.run(go())
    assert (meter.llm_calls, meter.tokens_in, meter.tokens_out, meter.tool_calls) == (2, 7, 2, 1)
    assert usage_tokens(None) == (0, 0)
    assert usage_tokens(SimpleNamespace(input_token_count=1, output_token_count=None)) == (1, 0)
