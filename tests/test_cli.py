from pathlib import Path

import pytest

from order_support import cli, replay


def test_artifacts_root_injected_unless_overridden(tmp_path):
    out = cli.with_artifacts_root(["--concurrency", "1"], tmp_path)
    assert out[-2:] == ["--override", f"artifacts_root={tmp_path.resolve()}"]
    given = ["--override", "artifacts_root=/x"]
    assert cli.with_artifacts_root(given, tmp_path) == given
    assert cli.with_artifacts_root(["--override=artifacts_root=/x"], tmp_path) == ["--override=artifacts_root=/x"]


def test_model_timeout_patch_targets_exist(monkeypatch):
    import importlib

    modules = [importlib.import_module(m) for m in cli._TIMEOUT_MODULES]
    for m in modules:
        monkeypatch.setattr(m, "DEFAULT_MODEL_TIMEOUT_S", m.DEFAULT_MODEL_TIMEOUT_S)
    cli.set_model_timeout(1234)
    assert all(m.DEFAULT_MODEL_TIMEOUT_S == 1234.0 for m in modules)


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
