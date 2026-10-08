import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from order_support import cli, replay


def test_artifacts_root_injected_unless_overridden(tmp_path):
    out = cli.with_artifacts_root(["--concurrency", "1"], tmp_path)
    assert out[-2:] == ["--override", f"artifacts_root={tmp_path.resolve()}"]
    given = ["--override", "artifacts_root=/x"]
    assert cli.with_artifacts_root(given, tmp_path) == given
    assert cli.with_artifacts_root(["--override=artifacts_root=/x"], tmp_path) == ["--override=artifacts_root=/x"]


@pytest.fixture
def restore_timeouts(monkeypatch):
    """Snapshot every global set_model_timeout touches so the patch can't leak between tests."""
    import importlib

    import litellm

    modules = [importlib.import_module(m) for m in cli._TIMEOUT_MODULES]
    for m in modules:
        monkeypatch.setattr(m, "DEFAULT_MODEL_TIMEOUT_S", m.DEFAULT_MODEL_TIMEOUT_S)
    client = importlib.import_module(cli._MODEL_CLIENT)
    monkeypatch.setattr(client, cli._AWAIT_HELPER, getattr(client, cli._AWAIT_HELPER))
    monkeypatch.setattr(litellm, "request_timeout", litellm.request_timeout)
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", litellm.request_timeout_explicitly_set)
    return modules


def test_model_timeout_patch_targets_exist(restore_timeouts):
    cli.set_model_timeout(1234)
    assert all(m.DEFAULT_MODEL_TIMEOUT_S == 1234.0 for m in restore_timeouts)


def test_model_timeout_must_be_positive(restore_timeouts):
    with pytest.raises(ValueError):
        cli.set_model_timeout(0)


def test_model_timeout_bounds_calls_assert_leaves_unbounded(restore_timeouts):
    """test_set / stratification / systematize pass timeout_s=None; the patch must bound them."""
    import asyncio
    import importlib

    client = importlib.import_module(cli._MODEL_CLIENT)
    cli.set_model_timeout(0.05)
    cli.set_model_timeout(0.05)  # idempotent: wraps the original helper, not the wrapper
    helper = getattr(client, cli._AWAIT_HELPER)
    assert not hasattr(helper.__wrapped__, "__wrapped__")
    with pytest.raises(TimeoutError):
        asyncio.run(helper(asyncio.sleep(1), timeout_s=None))
    # An explicit per-call timeout (e.g. test_set.timeout_s in YAML) still wins.
    assert asyncio.run(helper(asyncio.sleep(0.1, result="ok"), timeout_s=5)) == "ok"


def test_model_client_routes_every_call_through_patched_helper():
    """The helper patch only works if model_client looks the name up at call time."""
    import ast
    import importlib
    import inspect

    client = importlib.import_module(cli._MODEL_CLIENT)
    tree = ast.parse(inspect.getsource(client))
    funcs = {f.name: f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ("generate", "generate_structured", "generate_with_tools", "_run_sync_with_timeout"):
        assert any(isinstance(n, ast.Name) and n.id == cli._AWAIT_HELPER for n in ast.walk(funcs[name])), name


def test_model_timeout_overrides_litellm_600s_fallback(restore_timeouts):
    from litellm.litellm_core_utils.completion_timeout import CompletionTimeout
    from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout

    def resolve() -> float:
        return CompletionTimeout.resolve(None, {}, "openai", global_timeout=get_configured_request_timeout(),
                                         supports_httpx_timeout=lambda _: True)

    cli.set_model_timeout(1800)
    assert resolve() == 1800.0
    cli.set_model_timeout(6000)  # LiteLLM's "unset" sentinel value must still be honoured
    assert resolve() == 6000.0


def test_timeout_is_read_at_call_time():
    """The patch only works if assert-ai reads the module global per call, not as a default arg."""
    import ast
    import importlib
    import inspect

    for name in cli._TIMEOUT_MODULES:
        tree = ast.parse(inspect.getsource(importlib.import_module(name)))
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "DEFAULT_MODEL_TIMEOUT_S"]
        assert uses, name
        defaults = {id(d) for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for d in [*f.args.defaults, *f.args.kw_defaults] if d is not None}
        assert not any(id(u) in defaults for u in uses), f"{name} binds the timeout as a default arg"


GROUNDING = replay.REPO_ROOT / "evals" / "assert" / "grounding" / "eval_config.yaml"


@pytest.fixture
def fake_assert_run(monkeypatch, restore_timeouts):
    """Run cmd_run up to the assert-ai call without running ASSERT; record its argv."""
    from assert_ai.cli import cli as assert_cli

    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "stage_replay_inference_set", lambda *a, **k: None)
    monkeypatch.setattr(assert_cli, "main", lambda args, **_: calls.append(list(args)))
    monkeypatch.delenv(cli.TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(cli.TEST_SET_CONCURRENCY_ENV, raising=False)
    monkeypatch.delenv(cli.ASSERT_CONCURRENCY_ENV, raising=False)
    test_set = importlib.import_module(cli._TEST_SET_MODULE)
    monkeypatch.setattr(test_set, cli._TEST_SET_MODEL_CALL, getattr(test_set, cli._TEST_SET_MODEL_CALL))
    return calls


def test_run_propagates_model_timeout_to_agent_env(fake_assert_run, monkeypatch):
    import os

    from order_support import agent

    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.delenv(agent.TIMEOUT_ENV, raising=False)
    assert cli.main(["run", str(GROUNDING), "--model-timeout", "1500"]) == 0
    assert fake_assert_run and os.environ[cli.TIMEOUT_ENV] == "1500.0"
    assert agent.agent_timeout() == 1500.0


def test_run_reads_model_timeout_from_dotenv(fake_assert_run, monkeypatch, tmp_path, restore_timeouts):
    (tmp_path / ".env").write_text(f"{cli.TIMEOUT_ENV}=777\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", str(GROUNDING)]) == 0
    assert all(m.DEFAULT_MODEL_TIMEOUT_S == 777.0 for m in restore_timeouts)


def test_run_without_model_timeout_leaves_assert_defaults(fake_assert_run, monkeypatch, restore_timeouts):
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    before = [m.DEFAULT_MODEL_TIMEOUT_S for m in restore_timeouts]
    assert cli.main(["run", str(GROUNDING)]) == 0
    assert [m.DEFAULT_MODEL_TIMEOUT_S for m in restore_timeouts] == before


class _InFlight:
    """Fake test_set.generate_structured that records the peak number of concurrent calls."""

    def __init__(self) -> None:
        self.now = self.peak = self.calls = 0

    async def __call__(self, model, prompt, *, schema_name, json_schema, options=None):
        self.now += 1
        self.calls += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(0.01)
        self.now -= 1
        count = json_schema["properties"]["test_set"]["minItems"]
        return SimpleNamespace(parsed={"test_set": [{"title": "t", "description": "d", "system_prompt": ""}] * count})


def _generate_prompt_and_scenario(out: Path) -> dict:
    """Drive ASSERT's real run_test_set (both kinds, 7 jobs each) on a fresh loop, as a stage does."""
    test_set = importlib.import_module(cli._TEST_SET_MODULE)
    kind = {"model": "openai/fake", "sample_size": 8}
    taxonomy = replay.REPO_ROOT / "evals" / "assert" / "grounding" / "taxonomy.json"
    return asyncio.run(test_set.run_test_set(taxonomy_path=str(taxonomy), save_path=str(out), context=None,
                                             prompt=dict(kind), scenario=dict(kind), target=None))


@pytest.mark.parametrize("limit", [1, 3])
def test_test_set_concurrency_bounds_both_kinds_together(fake_assert_run, tmp_path, limit):
    test_set = importlib.import_module(cli._TEST_SET_MODULE)
    fake = _InFlight()
    setattr(test_set, cli._TEST_SET_MODEL_CALL, fake)
    unbounded = _generate_prompt_and_scenario(tmp_path / "a.jsonl")
    assert fake.peak > 8  # ASSERT alone: up to 8 per kind, both kinds at once
    baseline_calls, fake.peak, fake.calls = fake.calls, 0, 0

    cli.set_test_set_concurrency(limit)
    cli.set_test_set_concurrency(limit)  # idempotent: wraps the original call, not the wrapper
    bounded = _generate_prompt_and_scenario(tmp_path / "b.jsonl")
    assert fake.peak == limit and fake.calls == baseline_calls
    assert bounded["saved_count"] == unbounded["saved_count"]
    # Each stage runs on a new event loop; the bound must hold (and not crash) there too.
    fake.peak = 0
    _generate_prompt_and_scenario(tmp_path / "c.jsonl")
    assert fake.peak == limit


def test_test_set_module_looks_up_model_call_at_call_time():
    import ast
    import inspect

    test_set = importlib.import_module(cli._TEST_SET_MODULE)
    tree = ast.parse(inspect.getsource(test_set))
    gen = next(f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef) and f.name == "_generate_records")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == cli._TEST_SET_MODEL_CALL
               for n in ast.walk(gen))


def test_test_set_concurrency_precedence(monkeypatch):
    monkeypatch.delenv(cli.TEST_SET_CONCURRENCY_ENV, raising=False)
    monkeypatch.delenv(cli.ASSERT_CONCURRENCY_ENV, raising=False)
    resolve = cli.resolve_test_set_concurrency
    assert resolve(None, []) is None
    monkeypatch.setenv(cli.ASSERT_CONCURRENCY_ENV, "4")
    assert resolve(None, []) == 4
    assert resolve(None, ["--concurrency", "2", "--concurrency=1"]) == 1  # last wins, as in click
    monkeypatch.setenv(cli.TEST_SET_CONCURRENCY_ENV, "3")
    assert resolve(None, ["--concurrency", "1"]) == 3
    assert resolve(5, ["--concurrency", "1"]) == 5
    assert resolve(None, ["--concurrency", "x"]) == 3
    monkeypatch.delenv(cli.TEST_SET_CONCURRENCY_ENV)
    assert resolve(None, ["--concurrency", "x"]) is None  # left for assert-ai to reject
    monkeypatch.setenv(cli.TEST_SET_CONCURRENCY_ENV, "0")
    with pytest.raises(ValueError):
        resolve(None, [])


def test_assert_concurrency_env_name_matches_click_prefix(monkeypatch):
    import click
    from assert_ai.cli import cli as assert_cli

    run_cmd = assert_cli.commands["run"]
    parent = click.Context(assert_cli, info_name="assert-ai", **assert_cli.context_settings)
    ctx = click.Context(run_cmd, parent=parent, info_name="run")
    option = next(p for p in run_cmd.params if p.name == "concurrency")
    monkeypatch.setenv(cli.ASSERT_CONCURRENCY_ENV, "3")
    assert option.resolve_envvar_value(ctx) == "3"


def test_run_applies_test_set_concurrency(fake_assert_run, monkeypatch):
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    applied = []
    monkeypatch.setattr(cli, "set_test_set_concurrency", applied.append)
    assert cli.main(["run", str(GROUNDING), "--concurrency", "1"]) == 0
    assert cli.main(["run", str(GROUNDING), "--test-set-concurrency", "2", "--concurrency", "1"]) == 0
    assert cli.main(["run", str(GROUNDING)]) == 0
    assert applied == [1, 2]
    # --concurrency is still forwarded to assert-ai; --test-set-concurrency is not.
    assert fake_assert_run[1][-4:-2] == ["--concurrency", "1"]
    assert "--test-set-concurrency" not in fake_assert_run[1]
    with pytest.raises(SystemExit):
        cli.main(["run", str(GROUNDING), "--test-set-concurrency", "0"])


def test_replay_inference_set_is_staged_into_run_root(tmp_path):
    passthrough = cli.with_artifacts_root([], tmp_path)
    dest = cli.stage_replay_inference_set(cli.REPLAY_CONFIG, passthrough)
    expected = tmp_path / "results" / "order_support_judge_replay" / "baseline" / "inference_set.jsonl"
    assert dest == expected
    assert dest.read_bytes() == replay.INFERENCE_SET_PATH.read_bytes()


def test_live_suites_are_not_staged(tmp_path):
    config = replay.REPO_ROOT / "evals" / "assert" / "grounding" / "eval_config.yaml"
    assert cli.stage_replay_inference_set(config, cli.with_artifacts_root([], tmp_path)) is None


def test_calibrate_command_without_scores(tmp_path, capsys):
    assert cli.main(["calibrate", "--scores", str(tmp_path / "none.jsonl")]) == 2
    assert "no scores" in capsys.readouterr().err


def test_replay_check(capsys):
    assert cli.main(["replay", "check"]) == 0


def test_unknown_args_rejected_outside_run():
    with pytest.raises(SystemExit):
        cli.main(["replay", "check", "--bogus"])
