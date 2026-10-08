"""``order-support-evals``: thin helpers around the ``assert-ai`` CLI.

    order-support-evals run CONFIG [--model-timeout S] [assert-ai run options...]
    order-support-evals replay build | check
    order-support-evals calibrate [--scores PATH] [--json OUT]

``run`` forwards to ``assert-ai run`` in-process after three local fixes:
an absolute ``artifacts_root`` (a relative one resolves inside site-packages
when assert-ai is installed as a wheel), staging the committed replay
inference set into the run directory (ASSERT's viewer build expects it
there), and an optional per-call model timeout for slow local models.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from order_support import calibrate as calibrate_mod
from order_support import replay

REPO_ROOT = replay.REPO_ROOT
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
REPLAY_CONFIG = replay.REPLAY_DIR / "eval_config.yaml"
TIMEOUT_ENV = "ORDER_EVALS_MODEL_TIMEOUT_S"
# Modules that bind assert_ai.core.config_model.DEFAULT_MODEL_TIMEOUT_S at import time.
_TIMEOUT_MODULES = ("assert_ai.core.judge", "assert_ai.stages.inference")
# Every assert-ai model call (generate, generate_structured, generate_with_tools) awaits
# litellm through this helper; timeout_s=None (test-set generation, stratification,
# systematize, simulated tools) means no ASSERT-side bound.
_MODEL_CLIENT = "assert_ai.core.model_client"
_AWAIT_HELPER = "_await_with_timeout"


def _overrides(passthrough: list[str]) -> list[str]:
    out: list[str] = []
    it = iter(passthrough)
    for arg in it:
        if arg == "--override":
            out.append(next(it, ""))
        elif arg.startswith("--override="):
            out.append(arg.split("=", 1)[1])
    return out


def with_artifacts_root(passthrough: list[str], artifacts: Path = DEFAULT_ARTIFACTS) -> list[str]:
    if any(o.split("=", 1)[0].strip() == "artifacts_root" for o in _overrides(passthrough)):
        return list(passthrough)
    return [*passthrough, "--override", f"artifacts_root={artifacts.resolve()}"]


def set_model_timeout(seconds: float) -> None:
    """Bound every assert-ai model call by ``seconds``.

    * judge / tester / hosted target: patch the imported ``DEFAULT_MODEL_TIMEOUT_S``;
    * calls ASSERT makes with ``timeout_s=None`` (test-set generation, stratification,
      systematize, simulated tools): default the shared await helper to ``seconds``
      (an explicit ``timeout_s``, e.g. ``test_set.timeout_s`` in YAML, still wins);
    * LiteLLM: without an explicit timeout ``completion()`` falls back to a 600 s HTTP
      deadline, which would cut longer calls first, so set ``litellm.request_timeout``.
    """
    import importlib

    import litellm

    seconds = float(seconds)
    if seconds <= 0:
        raise ValueError("model timeout must be > 0")
    for name in _TIMEOUT_MODULES:
        module = importlib.import_module(name)
        if not hasattr(module, "DEFAULT_MODEL_TIMEOUT_S"):
            raise RuntimeError(f"{name}.DEFAULT_MODEL_TIMEOUT_S not found; assert-ai internals changed")
        module.DEFAULT_MODEL_TIMEOUT_S = seconds

    client = importlib.import_module(_MODEL_CLIENT)
    current = getattr(client, _AWAIT_HELPER, None)
    if current is None:
        raise RuntimeError(f"{_MODEL_CLIENT}.{_AWAIT_HELPER} not found; assert-ai internals changed")
    original = getattr(current, "__wrapped__", current)

    async def _await_with_default_timeout(awaitable: Any, *, timeout_s: float | None) -> Any:
        return await original(awaitable, timeout_s=seconds if timeout_s is None else timeout_s)

    _await_with_default_timeout.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(client, _AWAIT_HELPER, _await_with_default_timeout)

    litellm.request_timeout = seconds
    litellm.request_timeout_explicitly_set = True


def stage_replay_inference_set(config: Path, passthrough: list[str]) -> Path | None:
    """For judge-only configs, copy ``judge.inference_set_path`` into the run directory."""
    from assert_ai.config import resolve_stage_paths
    from assert_ai.runner import _load_context

    ctx: dict[str, Any] = _load_context(config=str(config), overrides=_overrides(passthrough))
    stages = dict(ctx["stages"])
    judge_cfg = stages.get("judge")
    if judge_cfg is None or "inference" in stages or not judge_cfg.get("inference_set_path"):
        return None
    src = Path(resolve_stage_paths({"inference_set_path": judge_cfg["inference_set_path"]},
                                   cfg_path=ctx["config_path"],
                                   artifacts_root=ctx["artifacts_root"])["inference_set_path"])
    dest = Path(ctx["run_root"]) / "inference_set.jsonl"
    if src.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    return dest


def cmd_run(args: argparse.Namespace, passthrough: list[str]) -> int:
    config = Path(args.config).resolve()
    if config == REPLAY_CONFIG.resolve() and not replay.is_current():
        print("inference_set.jsonl is stale; run 'order-support-evals replay build'", file=sys.stderr)
        return 2
    passthrough = with_artifacts_root(passthrough)
    timeout = args.model_timeout or os.environ.get(TIMEOUT_ENV)
    if timeout:
        set_model_timeout(float(timeout))
    stage_replay_inference_set(config, passthrough)
    from assert_ai.cli import cli

    try:
        cli.main(args=["run", "--config", str(config), *passthrough], prog_name="assert-ai",
                 standalone_mode=True)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    if args.action == "build":
        print(f"wrote {replay.build()} rows to {replay.INFERENCE_SET_PATH.relative_to(REPO_ROOT)}")
        return 0
    if replay.is_current():
        print("inference_set.jsonl is up to date")
        return 0
    print("inference_set.jsonl is stale; run 'order-support-evals replay build'", file=sys.stderr)
    return 1


def default_scores_path(run: str = "baseline") -> Path:
    return DEFAULT_ARTIFACTS / "results" / "order_support_judge_replay" / run / "scores.jsonl"


def cmd_calibrate(args: argparse.Namespace) -> int:
    path = Path(args.scores) if args.scores else default_scores_path(args.run)
    if not path.exists():
        print(f"no scores at {path}; run the judge_replay suite first", file=sys.stderr)
        return 2
    result = calibrate_mod.calibrate(calibrate_mod.load_scores(path))
    print(calibrate_mod.format_report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="order-support-evals", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an ASSERT eval config (extra options go to assert-ai run)")
    run.add_argument("config")
    run.add_argument("--model-timeout", type=float, default=None,
                     help="per-call timeout in seconds for every ASSERT model call (judge, tester, "
                          f"test-set generation, ...; default: ASSERT's 300 for judge/tester; env {TIMEOUT_ENV})")
    rp = sub.add_parser("replay", help="build or check the judge-replay inference set")
    rp.add_argument("action", choices=["build", "check"])
    cal = sub.add_parser("calibrate", help="compare judge_replay scores with reference labels")
    cal.add_argument("--scores", help="path to scores.jsonl (default: artifacts/results/.../<run>/scores.jsonl)")
    cal.add_argument("--run", default="baseline")
    cal.add_argument("--json", help="also write the full report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, passthrough = build_parser().parse_known_args(argv)
    if args.command == "run":
        return cmd_run(args, passthrough)
    if passthrough:
        build_parser().error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "replay":
        return cmd_replay(args)
    return cmd_calibrate(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
