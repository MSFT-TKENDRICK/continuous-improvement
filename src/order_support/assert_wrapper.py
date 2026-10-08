"""The ASSERT wrapper process (``order-support-evals run``) as a traced child of the harness.

Parent side (RRSI/self-improve drivers): :func:`launch` / :func:`command` +
:func:`wrapper_env` start the wrapper with ``obs.child_env()`` (W3C trace
context) plus the case-identity variables below.

Child side: :func:`install` (called by ``order_support.cli.cmd_run`` before
ASSERT starts) sets up ``ci_lab.telemetry`` when ``$CI_TELEMETRY`` asks for it
(M12), joins the parent's trace via ``obs.attach_from_env()``, registers the
``s1`` judge provider (M11) and wraps
every ASSERT case run in a ``ci.case`` span (design §12.3) carrying
``ci.case_id``, ``ci.trial``, ``ci.split`` and, when the experiment is known,
``agl.rollout_id`` (``contracts.RolloutKey``). Without a tracer provider the
spans are no-ops.
"""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
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
TELEMETRY_ENV = "CI_TELEMETRY"  # opt-in for ci_lab.telemetry.setup (M12): auto | on | off (default)
RUN_DIR_ENV = "CI_RUN_DIR"
PROFILE_ENV = "ORDER_AGENT_PROFILE"
_CASE_RUNNERS = ("_run_prompt_test_case", "_run_scenario_test_case")
_attach_token: object | None = None
_installed = False
_telemetry: Any = None
log = logging.getLogger(__name__)


def wrapper_env(*, experiment_id: str | None = None, variant: str | None = None, trial: int | None = None,
                split: str | None = None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """``obs.child_env(base)`` plus the case-identity variables :func:`install` reads."""
    env = obs.child_env(base)
    for name, value in ((EXPERIMENT_ENV, experiment_id), (VARIANT_ENV, variant), (TRIAL_ENV, trial),
                        (SPLIT_ENV, split)):
        if value is not None:
            env[name] = str(value)
    return env


def command(config: str | os.PathLike[str], *args: str) -> list[str]:
    """argv for ``order-support-evals run <config> [args...]`` in this interpreter."""
    return [sys.executable, "-m", "order_support.cli", "run", os.fspath(config), *args]


def launch(config: str | os.PathLike[str], *args: str, experiment_id: str | None = None,
           variant: str | None = None, trial: int | None = None, split: str | None = None,
           env: Mapping[str, str] | None = None, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Run the ASSERT wrapper as a subprocess whose spans join the current trace."""
    return subprocess.run(command(config, *args), env=wrapper_env(experiment_id=experiment_id, variant=variant,
                                                                   trial=trial, split=split, base=env),
                          check=False, **kwargs)


def case_attributes(test_case_id: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    trial = int(env.get(TRIAL_ENV) or 0)
    experiment, variant = env.get(EXPERIMENT_ENV) or None, env.get(VARIANT_ENV) or None
    rollout = (RolloutKey(experiment_id=experiment, variant=variant or "", case_id=test_case_id,
                          trial=trial).rollout_id if experiment else None)
    # CHAIN keeps ASSERT's span validation quiet if this span lands in a turn's capture.
    return {"openinference.span.kind": "CHAIN", ATTR_CASE: test_case_id, ATTR_TRIAL: trial,
            ATTR_SPLIT: env.get(SPLIT_ENV) or None, ATTR_EXPERIMENT: experiment, ATTR_VARIANT: variant,
            ATTR_ROLLOUT: rollout}


def _with_case_span(run: Any) -> Any:
    @functools.wraps(run)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        test_case = kwargs.get("test_case") or (args[0] if args else {})
        with obs.span(SPAN_CASE, case_attributes(str(test_case.get("test_case_id", "?")))):
            return await run(*args, **kwargs)

    wrapped.__ci_case_span__ = True  # type: ignore[attr-defined]
    return wrapped


def register_judge() -> None:
    """M11: resolve ASSERT's ``s1/...`` judge models through ``ci_lab.judge.provider``.

    Idempotent and offline: it only adds a LiteLLM custom-provider entry; backends connect lazily.
    """
    from ci_lab.judge.provider import register

    register()


def telemetry_mode(env: Mapping[str, str] | None = None) -> str | None:
    """``$CI_TELEMETRY`` as a ``telemetry.setup`` ``aspire`` mode, or None (the default: no setup).

    ``auto``/``1``/``true`` -> ``"auto"`` (export to a running ``ci-lab dashboard``, else JSONL
    only), ``on`` -> ``"on"``; unset, empty, ``off``, ``0`` or ``false`` -> None.
    """
    raw = ((os.environ if env is None else env).get(TELEMETRY_ENV) or "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return None
    if raw in ("1", "true", "yes", "auto"):
        return "auto"
    if raw == "on":
        return "on"
    log.warning("ignoring %s=%r (expected auto, on or off)", TELEMETRY_ENV, raw)
    return None


def setup_telemetry() -> Any | None:
    """M12: ``ci_lab.telemetry.setup("order-support", ...)`` when ``$CI_TELEMETRY`` asks for it.

    Never by default (tests and plain suite runs stay offline). Spans go to the Aspire dashboard
    (``aspire`` mode) and, when ``$CI_RUN_DIR`` is set, to ``<run_dir>/telemetry/spans-<pid>.jsonl``.
    An existing SDK tracer provider is reused; any setup failure only logs a warning.
    """
    global _telemetry
    if _telemetry is not None:
        return _telemetry
    mode = telemetry_mode()
    if mode is None:
        return None
    try:
        from ci_lab import telemetry

        _telemetry = telemetry.setup("order-support", profile=os.environ.get(PROFILE_ENV) or None,
                                     run_dir=os.environ.get(RUN_DIR_ENV) or None, aspire=mode)
    except Exception as exc:  # noqa: BLE001 - telemetry must never break an eval run
        log.warning("order-support telemetry setup skipped (%s: %s)", type(exc).__name__, exc)
    return _telemetry


def install(modules: Sequence[Any] | None = None) -> None:
    """Child-side setup before ASSERT runs; idempotent."""
    global _attach_token, _installed
    setup_telemetry()  # M12: before attaching, so the parent's context lands on our provider
    if _attach_token is None:
        _attach_token = obs.attach_from_env()
    register_judge()  # M11: S1 judge provider, before ASSERT runs
    if modules is None:
        from assert_ai.stages import inference

        modules = [inference]
    for module in modules:
        for name in _CASE_RUNNERS:
            run = getattr(module, name, None)
            if run is not None and not getattr(run, "__ci_case_span__", False):
                setattr(module, name, _with_case_span(run))
    _installed = True
