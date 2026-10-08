from pathlib import Path

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
