"""Judge admission wrapper (queue wait outside ASSERT's timeout) and the ``run-suites`` queue."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from ci_lab.judge import admission
from order_support import cli

S1_MODEL = "s1/llamacpp/qwen3.5-4b"


@pytest.fixture
def judge_module(monkeypatch):
    """The real ``assert_ai.core.judge`` with its judge-call symbol restored after the test."""
    module = importlib.import_module(cli._JUDGE_MODULE)
    current = getattr(module, cli._JUDGE_CALL)
    # Start unwrapped even if an earlier in-process ``cli.main(["run", ...])`` installed the wrapper.
    monkeypatch.setattr(module, cli._JUDGE_CALL, getattr(current, "__wrapped__", current))
    monkeypatch.setenv("CI_S1_LLAMA_URL", "http://127.0.0.1:9")  # never the live server
    monkeypatch.setenv(admission.MAX_INFLIGHT_ENV, "1")
    return module


@pytest.fixture
def fake_completion(monkeypatch):
    """Replace only LiteLLM's network call; ASSERT's generate/_with_retries/_await_with_timeout stay real."""
    import litellm

    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0.2)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": '{"score": 1}'}}],
                                     model=kwargs.get("model"))

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


def _request(module, timeout_s: float):
    options, system, user = module._build_judge_request(system_prompt="s", user_message="u",
                                                        judge_temperature=0.0, judge_max_tokens=8)
    return dataclasses.replace(options, timeout_s=timeout_s), system, user


def _hold_elsewhere(seconds: float) -> threading.Thread:
    """Another holder (fresh thread context) takes the only lease for ``seconds``."""
    url = admission.s1_local_url(S1_MODEL)
    got = threading.Event()

    def run() -> None:
        lease = admission.acquire(url)
        got.set()
        time.sleep(seconds)
        lease._owner_exit()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert got.wait(10)
    return t


def test_judge_call_symbol_is_pinned():
    """The wrapper only works while ASSERT routes every judge verdict through this module global."""
    module = importlib.import_module(cli._JUDGE_MODULE)
    fn = getattr(module, cli._JUDGE_CALL)
    assert inspect.iscoroutinefunction(fn)
    assert list(inspect.signature(fn).parameters)[:2] == ["judge_model", "options"]
    assert cli._JUDGE_CALL in module._run_judge_attempts.__code__.co_names
    assert "timeout_s" in {f.name for f in dataclasses.fields(module.GenerateOptions)}


def test_missing_judge_symbol_fails_loudly(judge_module, monkeypatch):
    monkeypatch.delattr(judge_module, cli._JUDGE_CALL)
    with pytest.raises(RuntimeError, match="assert-ai internals changed"):
        cli.set_judge_admission()


def test_judge_timeout_must_be_positive(judge_module):
    with pytest.raises(ValueError):
        cli.set_judge_admission(0)


def test_wrapper_is_idempotent(judge_module):
    original = getattr(judge_module, cli._JUDGE_CALL)
    cli.set_judge_admission()
    cli.set_judge_admission(5)
    wrapped = getattr(judge_module, cli._JUDGE_CALL)
    assert wrapped is not original and wrapped.__wrapped__ is original


def test_queue_wait_is_excluded_from_assert_timeout(judge_module, fake_completion):
    cli.set_judge_admission(0.5)
    options, system, user = _request(judge_module, 300)
    holder = _hold_elsewhere(0.9)
    t0 = time.monotonic()
    verdict, _ = asyncio.run(getattr(judge_module, cli._JUDGE_CALL)(S1_MODEL, options, system, user, None, ["score"]))
    elapsed = time.monotonic() - t0
    holder.join(5)
    assert verdict == {"score": 1}
    assert elapsed >= 0.8  # queued behind the other holder for longer than the 0.5 s call budget
    assert len(fake_completion) == 1


def test_waiting_inside_the_timeout_is_the_bug_being_fixed(judge_module, monkeypatch):
    """Control: the same wait *inside* ASSERT's timeout (unwrapped) raises TimeoutError."""
    import litellm

    async def queued_completion(**kwargs):
        async with admission.hold_async(admission.s1_local_url(S1_MODEL)):
            await asyncio.sleep(0.2)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "{}"}}])

    monkeypatch.setattr(litellm, "acompletion", queued_completion)
    options, system, user = _request(judge_module, 0.5)
    holder = _hold_elsewhere(0.9)
    with pytest.raises(TimeoutError):
        asyncio.run(getattr(judge_module, cli._JUDGE_CALL)(S1_MODEL, options, system, user, None, ["score"]))
    holder.join(5)


def test_lease_is_held_across_the_call_and_reentrant_inside(judge_module, monkeypatch):
    import litellm

    seen: list[admission.Lease | None] = []

    async def completion(**kwargs):
        url = admission.s1_local_url(S1_MODEL)
        seen.append(admission.current_lease(url))
        async with admission.hold_async(url) as inner:  # provider-side acquisition must not deadlock
            seen.append(inner)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "{}"}}])

    monkeypatch.setattr(litellm, "acompletion", completion)
    cli.set_judge_admission(5)
    options, system, user = _request(judge_module, 300)
    asyncio.run(asyncio.wait_for(
        getattr(judge_module, cli._JUDGE_CALL)(S1_MODEL, options, system, user, None, ["score"]), 10))
    assert seen[0] is not None and seen[1] is seen[0] and seen[0].index == 0


@pytest.mark.parametrize(("given", "current", "expected"), [(None, 300.0, cli.DEFAULT_LOCAL_JUDGE_TIMEOUT_S),
                                                            (None, 5000.0, 5000.0), (42.0, 300.0, 42.0)])
def test_s1_judge_budget(judge_module, monkeypatch, given, current, expected):
    seen: list[float] = []

    async def original(judge_model, options, *args, **kwargs):
        seen.append(options.timeout_s)
        return {"score": 1}, ""

    monkeypatch.setattr(judge_module, cli._JUDGE_CALL, original)
    cli.set_judge_admission(given)
    options, system, user = _request(judge_module, current)
    asyncio.run(getattr(judge_module, cli._JUDGE_CALL)(S1_MODEL, options, system, user, None, ["score"]))
    assert seen == [expected]


def test_non_s1_judges_pass_through_untouched(judge_module, monkeypatch):
    seen: list[tuple[float, admission.Lease | None]] = []

    async def original(judge_model, options, *args, **kwargs):
        seen.append((options.timeout_s, admission.current_lease()))
        return {"score": 1}, ""

    monkeypatch.setattr(judge_module, cli._JUDGE_CALL, original)
    cli.set_judge_admission(42)
    options, system, user = _request(judge_module, 300)
    for model in ("openai/gpt-5-mini", "s1/openai/gpt-4.1"):
        asyncio.run(getattr(judge_module, cli._JUDGE_CALL)(model, options, system, user, None, ["score"]))
    assert seen == [(300, None), (300, None)]


def test_resolve_judge_timeout(monkeypatch):
    monkeypatch.delenv(cli.JUDGE_TIMEOUT_ENV, raising=False)
    assert cli.resolve_judge_timeout(None) is None
    monkeypatch.setenv(cli.JUDGE_TIMEOUT_ENV, "900")
    assert cli.resolve_judge_timeout(None) == 900.0
    assert cli.resolve_judge_timeout(60) == 60.0
    with pytest.raises(ValueError):
        cli.resolve_judge_timeout(-1)


def test_run_installs_judge_admission(monkeypatch):
    applied: list[float | None] = []
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "stage_replay_inference_set", lambda *a, **k: None)
    monkeypatch.setattr(cli, "set_judge_admission", applied.append)
    for name in (cli.TIMEOUT_ENV, cli.TEST_SET_CONCURRENCY_ENV, cli.ASSERT_CONCURRENCY_ENV, cli.JUDGE_TIMEOUT_ENV):
        monkeypatch.delenv(name, raising=False)
    from assert_ai.cli import cli as assert_cli

    from order_support import assert_wrapper

    monkeypatch.setattr(assert_wrapper, "install", lambda *a, **k: None)
    monkeypatch.setattr(assert_cli, "main", lambda args, **_: None)
    path = cli.SUITES_DIR / "grounding" / "eval_config.yaml"
    assert cli.main(["run", str(path), "--judge-timeout", "777"]) == 0
    assert cli.main(["run", str(path)]) == 0
    assert applied == [777.0, None]


# --- run-suites -----------------------------------------------------------------------------

_FAKE_SUITE = """
import json, os, sys, time
out, delay, code, rows = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
start = time.time(); time.sleep(delay)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "inference_set.jsonl"), "w") as f:
    f.write("".join(json.dumps({"i": i}) + "\\n" for i in range(rows)))
with open(os.path.join(out, "scores.jsonl"), "w") as f:
    f.write("".join(json.dumps({"judge_status": "ok"}) + "\\n" for i in range(rows - (code == 3))))
with open(os.path.join(out, "span.json"), "w") as f:
    json.dump([start, time.time()], f)
print("fake suite on stdout"); print("fake suite on stderr", file=sys.stderr)
sys.exit(code)
"""


def _fake_job(tmp_path: Path, name: str, *, delay: float = 0.6, code: int = 0, rows: int = 3) -> cli.SuiteJob:
    script = tmp_path / "fake_suite.py"
    if not script.exists():
        script.write_text(_FAKE_SUITE, encoding="utf-8")
    out = tmp_path / "runs" / name
    return cli.SuiteJob(name=name, argv=[sys.executable, str(script), str(out), str(delay), str(code), str(rows)],
                        log_path=tmp_path / "logs" / f"{name}.log", scores_path=out / "scores.jsonl",
                        inference_path=out / "inference_set.jsonl")


def _max_overlap(spans: list[tuple[float, float]]) -> int:
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans], key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


@pytest.mark.parametrize("parallel", [1, 2])
def test_run_suite_queue_bounds_parallelism(tmp_path, parallel):
    jobs = [_fake_job(tmp_path, f"s{i}") for i in range(4)]
    results = cli.run_suite_queue(jobs, parallel, echo=io.StringIO())
    spans = [tuple(json.loads((tmp_path / "runs" / j.name / "span.json").read_text())) for j in jobs]
    assert _max_overlap(spans) == parallel
    assert [r["suite"] for r in results] == [j.name for j in jobs]
    assert all(r["exit_code"] == 0 and r["complete"] and r["scored_rows"] == 3 for r in results)
    log = jobs[0].log_path.read_text()
    assert "fake suite on stdout" in log and "fake suite on stderr" in log


def test_run_suite_queue_reports_failures_and_incomplete_scores(tmp_path):
    jobs = [_fake_job(tmp_path, "ok", delay=0.1), _fake_job(tmp_path, "boom", delay=0.1, code=2),
            _fake_job(tmp_path, "short", delay=0.1, code=3)]
    by = {r["suite"]: r for r in cli.run_suite_queue(jobs, 3, echo=io.StringIO())}
    assert by["ok"]["exit_code"] == 0 and by["ok"]["complete"] is True
    assert by["boom"]["exit_code"] == 2
    assert by["short"]["scored_rows"] == 2 and by["short"]["inference_rows"] == 3
    assert by["short"]["complete"] is False
    assert by["ok"]["scores_path"].endswith("scores.jsonl") and by["ok"]["duration_s"] >= 0.1


def test_select_suites(tmp_path):
    for name in ("a", "b", "judge_replay"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "eval_config.yaml").write_text("x: 1\n")
    assert cli.select_suites("all", tmp_path) == ["a", "b"]
    assert cli.select_suites("b, a,b", tmp_path) == ["b", "a"]
    assert cli.select_suites("judge_replay", tmp_path) == ["judge_replay"]
    with pytest.raises(ValueError, match="unknown"):
        cli.select_suites("a,nope", tmp_path)
    assert "judge_replay" not in cli.select_suites("all") and "grounding" in cli.select_suites("all")


def test_judge_model_resolves_pipeline_config_and_overrides():
    cfg = cli.SUITES_DIR / "grounding" / "eval_config.yaml"
    assert cli._judge_model(cfg, []) == S1_MODEL
    assert cli._judge_model(cfg, ["--override", "judge.model.name=openai/x"]) == "openai/x"
    assert cli._judge_model(Path("missing.yaml"), []) is None


def test_suite_argv_is_a_plain_argv_list():
    argv = cli.suite_argv(Path("cfg.yaml"), ["--judge-timeout", "900"], ["--override", "a=b c"])
    assert argv == [sys.executable, "-m", "order_support.cli", "run", "cfg.yaml", "--judge-timeout", "900",
                    "--override", "a=b c"]


def test_admission_stats(tmp_path):
    (tmp_path / "admission-1.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"wait_s": 0.0, "held_s": 80.0, "slot": 0, "capacity": 1},
        {"wait_s": 120.0, "held_s": 100.0, "slot": 0, "capacity": 1}]) + "\n")
    stats = cli.admission_stats(tmp_path)
    assert stats["leases"] == 2 and stats["max_wait_s"] == 120.0 and stats["mean_held_s"] == 90.0
    assert stats["capacity"] == [1] and stats["slots"] == [0]


def test_cmd_run_suites_end_to_end(tmp_path, monkeypatch, capsys):
    """CLI wiring: suites -> child argv/env/logs -> JSON summary on stdout -> exit code."""
    suites = tmp_path / "suites"
    for name in ("alpha", "beta"):
        (suites / name).mkdir(parents=True)
        (suites / name / "eval_config.yaml").write_text("judge:\n  model:\n    name: openai/gpt-5-mini\n")
    monkeypatch.setattr(cli, "SUITES_DIR", suites)
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run_paths", lambda config, passthrough: (None, None))
    codes = {"alpha": 0, "beta": 4}
    argvs: list[list[str]] = []
    real_argv = cli.suite_argv

    def fake_argv(config, opts, passthrough):
        argvs.append(real_argv(config, opts, passthrough))
        code = codes[config.parent.name]
        return [sys.executable, "-c", f"import os,sys; print(os.environ['CI_S1_ADMISSION_LOG']); sys.exit({code})"]

    monkeypatch.setattr(cli, "suite_argv", fake_argv)
    rc = cli.main(["run-suites", "--suites", "all", "--parallel", "2", "--judge-timeout", "900",
                   "--log-dir", str(tmp_path / "out"), "--override", "x=y"])
    summary = json.loads(capsys.readouterr().out)
    assert rc == 1 and summary["ok"] is False and summary["parallel"] == 2
    assert {r["suite"]: r["exit_code"] for r in summary["suites"]} == codes
    assert all(a[-4:] == ["--judge-timeout", "900.0", "--override", "x=y"] for a in argvs)
    assert "alpha" in (tmp_path / "out" / "alpha.log").read_text()
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["ok"] is False

    codes["beta"] = 0
    assert cli.main(["run-suites", "--suites", "beta", "--log-dir", str(tmp_path / "out2")]) == 0
    assert cli.main(["run-suites", "--suites", "nope"]) == 2


def test_cmd_run_suites_banner_reports_s1_admission(tmp_path, monkeypatch, capsys):
    """Real suite configs judge via s1/llamacpp: the banner and summary must name the admission queue."""
    monkeypatch.setenv("CI_S1_LLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("CI_S1_MAX_INFLIGHT", "1")
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run_paths", lambda config, passthrough: (None, None))
    monkeypatch.setattr(cli, "suite_argv", lambda config, opts, passthrough: [sys.executable, "-c", "pass"])
    assert cli.main(["run-suites", "--suites", "grounding,tool_selection", "--log-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    banner = next(line for line in captured.err.splitlines() if "s1 admission" in line)
    assert "n/a" not in banner and "http://127.0.0.1:9" in banner
    s1 = json.loads(captured.out)["s1_admission"]
    assert len(s1) == 1 and s1[0]["url"] == "http://127.0.0.1:9" and s1[0]["capacity"] == 1
