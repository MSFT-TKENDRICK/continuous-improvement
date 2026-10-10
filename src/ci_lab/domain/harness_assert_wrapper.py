"""ASSERT adapter for the isolated ``harness_*`` suites.

It registers the existing ``ci_lab.judge`` provider, binds the current frozen case id for
``harness_runner.chat`` and otherwise leaves ASSERT's installed API untouched.
"""

from __future__ import annotations

import argparse
import contextvars
import functools
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

EXPERIMENT_ENV = "CI_EXPERIMENT_ID"
VARIANT_ENV = "CI_VARIANT"
TRIAL_ENV = "CI_TRIAL"
SPLIT_ENV = "CI_SPLIT"
_CASE_RUNNERS = ("_run_prompt_test_case", "_run_scenario_test_case")
_case_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("ci_harness_case_id", default=None)
_attach_token: object | None = None
_installed = False


def current_case_id() -> str | None:
    return _case_id.get()


def wrapper_env(*, experiment_id: str | None = None, variant: str | None = None,
                trial: int | None = None, split: str | None = None,
                base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = obs.child_env(base)
    for name, value in ((EXPERIMENT_ENV, experiment_id), (VARIANT_ENV, variant),
                        (TRIAL_ENV, trial), (SPLIT_ENV, split)):
        if value is not None:
            env[name] = str(value)
    return env


def case_attributes(case_id: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = os.environ if env is None else env
    trial = int(values.get(TRIAL_ENV) or 0)
    experiment, variant = values.get(EXPERIMENT_ENV), values.get(VARIANT_ENV)
    rollout = (RolloutKey(experiment, variant or "", case_id, trial).rollout_id if experiment else None)
    return {
        "openinference.span.kind": "CHAIN",
        ATTR_CASE: case_id,
        ATTR_TRIAL: trial,
        ATTR_SPLIT: values.get(SPLIT_ENV),
        ATTR_EXPERIMENT: experiment,
        ATTR_VARIANT: variant,
        ATTR_ROLLOUT: rollout,
    }


def _with_case_span(run: Any) -> Any:
    @functools.wraps(run)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        test_case = kwargs.get("test_case") or (args[0] if args else {})
        case_id = str(test_case.get("test_case_id") or "?")
        token = _case_id.set(case_id)
        try:
            with obs.span(SPAN_CASE, case_attributes(case_id)):
                return await run(*args, **kwargs)
        finally:
            _case_id.reset(token)

    wrapped.__ci_harness_case_span__ = True
    return wrapped


def register_judge() -> None:
    from ci_lab.judge.provider import register

    register()


def install(modules: Sequence[Any] | None = None) -> None:
    """Install once against the real ``assert-ai`` inference module."""
    global _attach_token, _installed
    if _attach_token is None:
        _attach_token = obs.attach_from_env()
    register_judge()
    if modules is None:
        from assert_ai.stages import inference

        modules = (inference,)
    for module in modules:
        for name in _CASE_RUNNERS:
            run = getattr(module, name, None)
            if run is not None and not getattr(run, "__ci_harness_case_span__", False):
                setattr(module, name, _with_case_span(run))
    _installed = True


def command(config: str | os.PathLike[str], *args: str) -> list[str]:
    return [sys.executable, "-m", __name__, "run", os.fspath(config), *args]


def launch(config: str | os.PathLike[str], *args: str, env: Mapping[str, str] | None = None,
           **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(command(config, *args), env=wrapper_env(base=env), check=False, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness-assert")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("config")
    args, rest = parser.parse_known_args(argv)
    config = Path(args.config).resolve()
    if not config.parent.name.startswith("harness_"):
        parser.error("the harness wrapper accepts only harness_* suites")
    install()
    from assert_ai.cli import cli

    try:
        cli.main(args=["run", "--config", str(config), *rest], prog_name="assert-ai",
                 standalone_mode=True)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
