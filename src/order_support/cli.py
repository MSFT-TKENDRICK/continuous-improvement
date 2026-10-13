"""``order-support-evals``: thin helpers around the ``assert-ai`` CLI.

    order-support-evals run CONFIG [--model-timeout S] [--judge-timeout S] [--test-set-concurrency N]
                                   [assert-ai run options...]
    order-support-evals run-suites [--suites all|a,b] [--parallel N] [--model-timeout S]
                                   [--judge-timeout S] [--log-dir DIR] [run options...]
    order-support-evals replay build | check
    order-support-evals calibrate [--scores PATH] [--json OUT]

``run`` forwards to ``assert-ai run`` in-process after local fixes:
an absolute ``artifacts_root`` (a relative one resolves inside site-packages
when assert-ai is installed as a wheel), staging the committed replay
inference set into the run directory (ASSERT's viewer build expects it
there), an optional per-call model timeout for slow local models, a
bound on concurrent test-set generation calls, and host-wide admission for
local ``s1/llamacpp`` judges: each judge call first queues for a slot lease
shared by every process on the machine (``ci_lab.judge.admission``;
``CI_S1_MAX_INFLIGHT``, ``CI_S1_LOCK_DIR``), and only then starts ASSERT's
per-call timeout (``--judge-timeout``).

``run-suites`` runs suite configs as ``run`` subprocesses from a bounded queue,
one log per suite, and prints a JSON summary; re-running the same run name
resumes each suite's completed stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
TEST_SET_CONCURRENCY_ENV = "ORDER_EVALS_TEST_SET_CONCURRENCY"
# assert-ai's click group uses auto_envvar_prefix="ASSERT_AI", so this env var sets `run --concurrency`.
ASSERT_CONCURRENCY_ENV = "ASSERT_AI_RUN_CONCURRENCY"
# Test-set generation runs up to 8 jobs per kind and gathers the prompt and scenario kinds
# concurrently; ASSERT has no setting for it, so the wrapper bounds this module's model call.
_TEST_SET_MODULE = "assert_ai.stages.test_set"
_TEST_SET_MODEL_CALL = "generate_structured"
# One judge verdict (ASSERT's judge stage -> multi_judge -> _run_judge_attempts looks it up
# as a module global per call); local s1 judges queue for a host-wide slot lease around it.
_JUDGE_MODULE = "assert_ai.core.judge"
_JUDGE_CALL = "_single_judge_call"
JUDGE_TIMEOUT_ENV = "ORDER_EVALS_JUDGE_TIMEOUT_S"
# Per-call budget for local CPU s1 judges once they hold a lease (service time only). Measured
# at 1 in flight: 75-153 s per verdict (~1.6K-token prompt x 9 decisions at ~55 tok/s prefill);
# 1200 s leaves ~8x headroom for long transcripts and CPU contention from other work.
DEFAULT_LOCAL_JUDGE_TIMEOUT_S = 1200.0
SUITES_DIR = REPO_ROOT / "evals" / "assert"
# Judge-only replay of committed transcripts; run it by name, it is not part of "all".
_NOT_IN_ALL = frozenset({"judge_replay"})
_admission_log = logging.getLogger("ci_lab.judge.admission")


def _option_values(passthrough: list[str], option: str) -> list[str]:
    out: list[str] = []
    it = iter(passthrough)
    for arg in it:
        if arg == option:
            out.append(next(it, ""))
        elif arg.startswith(f"{option}="):
            out.append(arg.split("=", 1)[1])
    return out


def _overrides(passthrough: list[str]) -> list[str]:
    return _option_values(passthrough, "--override")


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


def set_test_set_concurrency(limit: int) -> None:
    """Allow at most ``limit`` in-flight test-set generation model calls, across prompt and scenario.

    Waiting for a slot happens outside ASSERT's per-call timeout, so queued jobs can't time out.
    Each stage runs on its own event loop, hence one semaphore per loop.
    """
    import asyncio
    import importlib
    import weakref

    if limit < 1:
        raise ValueError("test-set concurrency must be >= 1")
    module = importlib.import_module(_TEST_SET_MODULE)
    current = getattr(module, _TEST_SET_MODEL_CALL, None)
    if current is None:
        raise RuntimeError(f"{_TEST_SET_MODULE}.{_TEST_SET_MODEL_CALL} not found; assert-ai internals changed")
    original = getattr(current, "__wrapped__", current)
    semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()

    async def _bounded_generate_structured(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        if (semaphore := semaphores.get(loop)) is None:
            semaphore = semaphores[loop] = asyncio.Semaphore(limit)
        async with semaphore:
            return await original(*args, **kwargs)

    _bounded_generate_structured.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(module, _TEST_SET_MODEL_CALL, _bounded_generate_structured)


def set_judge_admission(timeout_s: float | None = None) -> None:
    """Queue local s1 judge calls host-wide *before* ASSERT's per-call timeout starts.

    Wraps ``assert_ai.core.judge._single_judge_call`` (one judge verdict, including ASSERT's
    transport retries inside ``generate``/``generate_structured``). For ``s1/llamacpp/...``
    (alias ``s1/local/...``) judges the wrapper takes a cross-process slot lease from
    :mod:`ci_lab.judge.admission` with no deadline, holds it across ASSERT's retries, and
    only then calls ASSERT, so ``timeout_s`` measures service time, never queue time.
    The per-call budget for those judges is ``timeout_s`` if given, else at least
    :data:`DEFAULT_LOCAL_JUDGE_TIMEOUT_S`. Other judge models are passed through untouched.
    """
    import dataclasses
    import importlib

    from ci_lab.judge import admission

    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("judge timeout must be > 0")
    module = importlib.import_module(_JUDGE_MODULE)
    current = getattr(module, _JUDGE_CALL, None)
    if current is None:
        raise RuntimeError(f"{_JUDGE_MODULE}.{_JUDGE_CALL} not found; assert-ai internals changed")
    original = getattr(current, "__wrapped__", current)
    announced: set[str] = set()

    async def _admitted_judge_call(judge_model: str, options: Any, *args: Any, **kwargs: Any) -> Any:
        url = admission.s1_local_url(judge_model)
        if url is None:
            return await original(judge_model, options, *args, **kwargs)
        budget = timeout_s if timeout_s is not None else max(DEFAULT_LOCAL_JUDGE_TIMEOUT_S,
                                                             float(options.timeout_s or 0))
        options = dataclasses.replace(options, timeout_s=budget)
        async with admission.hold_async(url) as lease:
            if url not in announced:
                announced.add(url)
                _admission_log.info("s1 judge admission: %s lease %d/%d (server slots %s), "
                                    "per-call judge timeout %.0fs", lease.base_url, lease.index,
                                    lease.capacity, lease.total_slots, budget)
            return await original(judge_model, options, *args, **kwargs)

    _admitted_judge_call.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(module, _JUDGE_CALL, _admitted_judge_call)


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at emit time (survives stream swaps/capture)."""

    @property  # type: ignore[override]
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


def _enable_admission_logging() -> None:
    """Show queue-wait INFO lines on stderr (ASSERT's own logging config may not include them)."""
    logger = logging.getLogger("ci_lab.judge.admission")
    if not any(isinstance(h, _StderrHandler) for h in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _positive_int(raw: str, source: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{source} must be an integer >= 1, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{source} must be an integer >= 1, got {raw!r}")
    return value


def _arg_positive_int(raw: str) -> int:
    try:
        return _positive_int(raw, "--test-set-concurrency")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _arg_parallel(raw: str) -> int:
    try:
        return _positive_int(raw, "--parallel")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def resolve_test_set_concurrency(cli_value: int | None, passthrough: list[str]) -> int | None:
    """``--test-set-concurrency``, else its env var, else assert-ai's ``--concurrency`` (flag or env)."""
    if cli_value is not None:
        return _positive_int(str(cli_value), "--test-set-concurrency")
    if raw := os.environ.get(TEST_SET_CONCURRENCY_ENV, "").strip():
        return _positive_int(raw, TEST_SET_CONCURRENCY_ENV)
    if given := _option_values(passthrough, "--concurrency"):
        try:
            return _positive_int(given[-1], "--concurrency")
        except ValueError:
            return None  # let assert-ai's own option validation report it
    if raw := os.environ.get(ASSERT_CONCURRENCY_ENV, "").strip():
        try:
            return _positive_int(raw, ASSERT_CONCURRENCY_ENV)
        except ValueError:
            return None
    return None


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


def _load_dotenv() -> None:
    """Load the project ``.env`` now, as assert_ai.runner does on import, so it sets our env vars too."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))


def resolve_model_timeout(cli_value: float | None) -> float | None:
    if cli_value is not None:
        return float(cli_value)
    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    return float(raw) if raw else None


def resolve_judge_timeout(cli_value: float | None) -> float | None:
    """``--judge-timeout``, else ``$ORDER_EVALS_JUDGE_TIMEOUT_S``, else None (local s1 default)."""
    if cli_value is not None:
        value = float(cli_value)
    elif raw := os.environ.get(JUDGE_TIMEOUT_ENV, "").strip():
        value = float(raw)
    else:
        return None
    if value <= 0:
        raise ValueError("judge timeout must be > 0")
    return value


def cmd_run(args: argparse.Namespace, passthrough: list[str]) -> int:
    config = Path(args.config).resolve()
    if config == REPLAY_CONFIG.resolve() and not replay.is_current():
        print("inference_set.jsonl is stale; run 'order-support-evals replay build'", file=sys.stderr)
        return 2
    _load_dotenv()
    from order_support import assert_wrapper

    assert_wrapper.register_judge()  # M11: idempotent, no network
    assert_wrapper.setup_telemetry()  # M12: only when $CI_TELEMETRY asks for it
    passthrough = with_artifacts_root(passthrough)
    timeout = resolve_model_timeout(args.model_timeout)
    if timeout is not None:
        set_model_timeout(timeout)
        # The in-process callable agent reads this per call (order_support.agent.agent_timeout).
        os.environ[TIMEOUT_ENV] = str(timeout)
    test_set_concurrency = resolve_test_set_concurrency(args.test_set_concurrency, passthrough)
    if test_set_concurrency is not None:
        set_test_set_concurrency(test_set_concurrency)
    set_judge_admission(resolve_judge_timeout(args.judge_timeout))
    _enable_admission_logging()
    stage_replay_inference_set(config, passthrough)
    assert_wrapper.install()  # join the parent trace; ci.case spans (re-checks M11/M12, idempotent)
    from assert_ai.cli import cli

    try:
        cli.main(args=["run", "--config", str(config), *passthrough], prog_name="assert-ai",
                 standalone_mode=True)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


@dataclass
class SuiteJob:
    """One ``order-support-evals run`` subprocess in the ``run-suites`` queue."""

    name: str
    argv: list[str]
    log_path: Path
    scores_path: Path | None = None
    inference_path: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


def available_suites(suites_dir: Path | None = None) -> list[str]:
    return sorted(p.parent.name for p in (suites_dir or SUITES_DIR).glob("*/eval_config.yaml"))


def select_suites(spec: str, suites_dir: Path | None = None) -> list[str]:
    """``all`` (every suite except the judge-only replay) or a comma-separated list."""
    available = available_suites(suites_dir)
    if spec.strip().lower() == "all":
        return [s for s in available if s not in _NOT_IN_ALL]
    names = list(dict.fromkeys(s.strip() for s in spec.split(",") if s.strip()))
    unknown = [s for s in names if s not in available]
    if unknown or not names:
        raise ValueError(f"unknown suite(s) {unknown or spec!r}; available: {', '.join(available)}")
    return names


def run_paths(config: Path, passthrough: list[str]) -> tuple[Path | None, Path | None]:
    """(scores.jsonl, inference_set.jsonl) of the ASSERT run dir this config + overrides resolve to."""
    try:
        from assert_ai.runner import _load_context

        ctx = _load_context(config=str(config), overrides=_overrides(with_artifacts_root(passthrough)))
        root = Path(ctx["run_root"])
    except Exception:  # noqa: BLE001 - the summary is best effort; the run reports real errors
        return None, None
    return root / "scores.jsonl", root / "inference_set.jsonl"


def _jsonl_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({})
    return rows


def suite_result(job: SuiteJob, exit_code: int | None, duration_s: float) -> dict[str, Any]:
    scores = _jsonl_rows(job.scores_path)
    inference = _jsonl_rows(job.inference_path)
    failed = sum(1 for r in scores or [] if r.get("judge_status") == "judge_failed")
    scored = len(scores or []) - failed
    complete = (None if job.scores_path is None else
                scores is not None and failed == 0 and (inference is None or scored >= len(inference)))
    out: dict[str, Any] = {
        "suite": job.name, "exit_code": exit_code, "duration_s": round(duration_s, 1),
        "scored_rows": scored if scores is not None else None, "judge_failed_rows": failed,
        "inference_rows": len(inference) if inference is not None else None, "complete": complete,
        "scores_path": str(job.scores_path) if job.scores_path else None, "log": str(job.log_path),
    }
    if adm := job.env.get("CI_S1_ADMISSION_LOG"):
        out["s1_admission"] = admission_stats(Path(adm))
    return out


def admission_stats(directory: Path) -> dict[str, Any]:
    """Summarise ``$CI_S1_ADMISSION_LOG`` lease records (wait = queue time, held = service time)."""
    rows = [r for p in sorted(directory.glob("admission-*.jsonl")) for r in _jsonl_rows(p) or []]
    waits = [float(r.get("wait_s") or 0) for r in rows]
    held = [float(r.get("held_s") or 0) for r in rows]
    return {
        "leases": len(rows),
        "max_wait_s": round(max(waits), 1) if waits else 0.0,
        "total_wait_s": round(sum(waits), 1),
        "mean_held_s": round(sum(held) / len(held), 1) if held else 0.0,
        "max_held_s": round(max(held), 1) if held else 0.0,
        "capacity": sorted({r.get("capacity") for r in rows if r.get("capacity") is not None}),
        "slots": sorted({r.get("slot") for r in rows if r.get("slot") is not None}),
    }


def run_suite_queue(jobs: list[SuiteJob], parallel: int, *, base_env: dict[str, str] | None = None,
                    echo: Any = None) -> list[dict[str, Any]]:
    """Run ``jobs`` as subprocesses (argv lists, no shell), at most ``parallel`` at a time, FIFO.

    Each job's stdout+stderr go to its own log file. Ctrl-C terminates the running children
    and skips the queued ones."""
    if parallel < 1:
        raise ValueError("--parallel must be >= 1")
    echo = echo or sys.stderr
    env = dict(os.environ if base_env is None else base_env)
    running: dict[str, subprocess.Popen] = {}
    mu = threading.Lock()
    stop = threading.Event()

    def one(job: SuiteJob) -> dict[str, Any]:
        if stop.is_set():
            return {**suite_result(job, None, 0.0), "skipped": True}
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        with open(job.log_path, "wb") as log_file:
            proc = subprocess.Popen(job.argv, stdout=log_file, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env={**env, **job.env})
            with mu:
                running[job.name] = proc
            print(f"[run-suites] start {job.name} (pid {proc.pid}) -> {job.log_path}", file=echo, flush=True)
            code = proc.wait()
        with mu:
            running.pop(job.name, None)
        res = suite_result(job, code, time.monotonic() - t0)
        print(f"[run-suites] done {job.name}: exit {code} in {res['duration_s']:.0f}s, "
              f"scored {res['scored_rows']}/{res['inference_rows']}", file=echo, flush=True)
        return res

    import concurrent.futures as cf

    with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(jobs))), thread_name_prefix="suite") as ex:
        futures = [ex.submit(one, job) for job in jobs]
        try:
            pending = set(futures)
            while pending:
                _, pending = cf.wait(pending, timeout=1.0)
        except KeyboardInterrupt:
            stop.set()
            with mu:
                for proc in running.values():
                    proc.terminate()
            raise
    return [f.result() for f in futures]


def _judge_model(config: Path, passthrough: list[str]) -> str | None:
    """The judge model ASSERT will use for this config + ``--override``s."""
    try:
        from assert_ai.runner import _load_context

        ctx = _load_context(config=str(config), overrides=_overrides(with_artifacts_root(passthrough)))
        return str(dict(ctx["stages"])["judge"]["model"]["name"])
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def suite_argv(config: Path, opts: list[str], passthrough: list[str]) -> list[str]:
    return [sys.executable, "-m", "order_support.cli", "run", str(config), *opts, *passthrough]


def cmd_run_suites(args: argparse.Namespace, passthrough: list[str]) -> int:
    try:
        names = select_suites(args.suites)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _load_dotenv()
    run_dir = Path(args.log_dir) if args.log_dir else DEFAULT_ARTIFACTS / "run-suites" / time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_dir.resolve()
    opts: list[str] = []
    if args.model_timeout is not None:
        opts += ["--model-timeout", str(args.model_timeout)]
    if args.judge_timeout is not None:
        opts += ["--judge-timeout", str(args.judge_timeout)]
    jobs = []
    for name in names:
        config = (SUITES_DIR / name / "eval_config.yaml").resolve()
        scores, inference = run_paths(config, passthrough)
        jobs.append(SuiteJob(
            name=name, argv=suite_argv(config, opts, passthrough),
            log_path=run_dir / f"{name}.log", scores_path=scores, inference_path=inference,
            env={"CI_S1_ADMISSION_LOG": str(run_dir / "admission" / name), "PYTHONUNBUFFERED": "1"},
        ))
    from ci_lab.judge import admission

    urls = dict.fromkeys(admission.s1_local_url(_judge_model(SUITES_DIR / n / "eval_config.yaml", passthrough) or "")
                         for n in names)
    s1 = [admission.describe(u) for u in urls if u]
    print(f"[run-suites] {len(jobs)} suite(s), parallel {args.parallel}, logs in {run_dir}; "
          f"s1 admission {s1 or 'n/a'}", file=sys.stderr, flush=True)
    t0 = time.monotonic()
    results = run_suite_queue(jobs, args.parallel)
    ok = all(r["exit_code"] == 0 and r["complete"] is not False for r in results)
    summary = {"ok": ok, "wall_s": round(time.monotonic() - t0, 1), "parallel": args.parallel,
               "run_dir": str(run_dir), "s1_admission": s1, "suites": results}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


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
    run.add_argument("--test-set-concurrency", type=_arg_positive_int, default=None, metavar="N",
                     help="max concurrent test-set generation model calls across prompt and scenario kinds "
                          f"(env {TEST_SET_CONCURRENCY_ENV}; default: --concurrency if given, else ASSERT's "
                          "up to 8 per kind)")
    run.add_argument("--judge-timeout", type=float, default=None, metavar="S",
                     help="per-call timeout for local s1/llamacpp judge calls, counted only after the "
                          "host-wide judge queue admits the call (env "
                          f"{JUDGE_TIMEOUT_ENV}; default max({DEFAULT_LOCAL_JUDGE_TIMEOUT_S:.0f}, "
                          "--model-timeout)); other judges keep --model-timeout")
    rs = sub.add_parser("run-suites", allow_abbrev=False,
                        help="run several ASSERT suites from a bounded queue (extra options go to each run)")
    rs.add_argument("--suites", default="all",
                    help="'all' (every suite under evals/assert except judge_replay) or a comma-separated list")
    rs.add_argument("--parallel", type=_arg_parallel, default=2, metavar="N",
                    help="max suites running at once (judge calls still queue host-wide; default 2)")
    rs.add_argument("--model-timeout", type=float, default=None, metavar="S", help="passed to each run")
    rs.add_argument("--judge-timeout", type=float, default=None, metavar="S", help="passed to each run")
    rs.add_argument("--log-dir", default=None,
                    help="per-suite logs + summary.json (default artifacts/run-suites/<timestamp>)")
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
    if args.command == "run-suites":
        return cmd_run_suites(args, passthrough)
    if passthrough:
        build_parser().error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "replay":
        return cmd_replay(args)
    return cmd_calibrate(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
