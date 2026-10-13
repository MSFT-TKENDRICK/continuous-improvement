"""``ci-lab sleep run|dry-run|usage-gate|harvest-usage|redact-spans`` (registered by
``ci_lab.cli``). Heavy imports stay lazy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROFILES = ("copilot", "offline", "fake")


def register(sub: Any) -> None:
    sleep = sub.add_parser("sleep", help="SkillOpt-Sleep nightly skill consolidation")
    ssub = sleep.add_subparsers(dest="sleep_command", required=True)

    run = ssub.add_parser("run", help="run one night and write the sleep bundle")
    run.add_argument("--profile", choices=PROFILES, default="copilot")
    run.add_argument("--out", type=Path, required=True, help="bundle directory (e.g. out/sleep-bundle)")
    run.add_argument("--max-tasks", type=int, default=40)
    run.add_argument("--max-minutes", type=float, default=60.0)
    run.add_argument("--max-rollouts", type=int, default=400)
    run.add_argument("--max-tokens", type=int, default=2_000_000)
    run.add_argument("--max-aiu", type=float, default=None)
    _common(run)
    run.add_argument("--run-dir", type=Path, default=None,
                     help="status.d markers + telemetry JSONL (default: $CI_RUN_DIR or <out>-work)")
    run.add_argument("--date", default=None, help="night date yyyymmdd (default: UTC today)")
    run.add_argument("--base-sha", default=None, help="base commit (default: git rev-parse HEAD)")
    run.add_argument("--lessons", action="store_true", default=None,
                     help="HOOK(M16): mine lesson candidates into the bundle as proposals "
                          "(default: $SLEEP_LESSONS, else off)")
    run.add_argument("--lessons-dir", type=Path, default=None,
                     help="local trajectory store, accumulated across nights; keep it gitignored "
                          "(default: <work-dir>/lessons; e.g. artifacts/lessons/sleep)")
    run.add_argument("--lessons-source", action="append", default=[],
                     help="extra local harvest input SOURCE:PATH (e.g. usage:DIR); repeatable")
    run.set_defaults(func=_run)

    dry = ssub.add_parser("dry-run", help="harvest + validate tasks only (no model calls)")
    _common(dry)
    dry.set_defaults(func=_dry_run)

    gate = ssub.add_parser("usage-gate", help="new reviewed tasks since last night >= threshold?")
    gate.add_argument("--repo", type=Path, default=Path("."))
    gate.add_argument("--targets", type=Path, default=None, help="skills registry YAML")
    gate.add_argument("--threshold", type=int, default=None,
                      help="default: $SLEEP_USAGE_THRESHOLD or 1")
    gate.add_argument("--force", action="store_true", help="run regardless of the threshold")
    gate.set_defaults(func=_usage_gate)

    hu = ssub.add_parser("harvest-usage", help="usage traces -> redacted PENDING tasks (reviewed:false)")
    hu.add_argument("--repo", type=Path, default=Path("."))
    hu.add_argument("--targets", type=Path, default=None, help="skills registry YAML")
    hu.add_argument("--source", action="append", default=[], required=True,
                    help="agl:PATH | spans:RUN_DIR | artifacts:DIR; repeatable")
    hu.add_argument("--date", default=None, help="yyyymmdd (default: UTC today)")
    hu.add_argument("--since-hours", type=float, default=None, help="ignore traces older than this")
    hu.add_argument("--max-new", type=int, default=25)
    hu.add_argument("--bundle", type=Path, default=None, help="write a kind=usage bundle for sleep_publish.py")
    hu.add_argument("--base-sha", default=None)
    hu.add_argument("--open-pr", action="store_true",
                    help="commit to exp/usage-<date>/tasks and open a DRAFT PR via gh (local use)")
    hu.set_defaults(func=_harvest_usage)

    rs = ssub.add_parser("redact-spans", help="redacted copy of <run-dir>/telemetry/spans-*.jsonl")
    rs.add_argument("--run-dir", type=Path, required=True)
    rs.add_argument("--out", type=Path, required=True)
    rs.set_defaults(func=_redact_spans)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--targets", type=Path, default=None, help="skills registry YAML (default: packaged)")
    p.add_argument("--tasks-file", type=Path, default=None, help="override (single target only)")
    p.add_argument("--agl-export", type=Path, action="append", default=[],
                   help="AGL journal export JSONL (evolve split only); repeatable")


def _gh_output(**values: Any) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in values.items():
            fh.write(f"{k}={str(v).lower() if isinstance(v, bool) else v}\n")


def _targets(path: Path | None) -> list[Any]:
    from ci_lab.sleep.registry import load_targets

    return load_targets(path.resolve() if path else None)


def _telemetry(profile: str, run_dir: Path) -> Any:
    """Lazy ``ci_lab.telemetry.setup`` (M12). Missing module or setup failure only warns."""
    try:
        from ci_lab import telemetry  # type: ignore[attr-defined]
    except ImportError:
        return None
    try:
        telemetry.setup("sleep", profile=profile, run_dir=run_dir)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the night
        print(f"sleep: telemetry disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return telemetry


def _attempt() -> int:
    raw = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    return int(raw) if raw.isdigit() and 1 <= int(raw) <= 9999 else 1


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _model_preflight(profile: Any) -> None:
    """Fail before the night spends budget if a Copilot model it uses is not available."""
    import asyncio

    from ci_lab.providers.models import check_copilot_models
    from ci_lab.sleep.wiring import model_uses

    if uses := model_uses(profile):
        asyncio.run(check_copilot_models(uses))


def _run(args: argparse.Namespace) -> int:
    from ci_lab.contracts import Profile
    from ci_lab.providers.models import ModelPreflightError
    from ci_lab.sleep.budget import BudgetLimits
    from ci_lab.sleep.lessons_hook import parse_source
    from ci_lab.sleep.night import SleepConfig, run_night
    from ci_lab.sleep.registry import RegistryError
    from ci_lab.sleep.wiring import WiringError, build_deps

    lessons = _env_flag("SLEEP_LESSONS") if args.lessons is None else bool(args.lessons)
    out = args.out.resolve()
    run_dir = (args.run_dir or (Path(os.environ["CI_RUN_DIR"]) if os.environ.get("CI_RUN_DIR")
                                else out.parent / f"{out.name}-work")).resolve()
    try:
        cfg = SleepConfig(
            repo_root=args.repo.resolve(), out_dir=out, profile=args.profile, targets=_targets(args.targets),
            tasks_file=args.tasks_file.resolve() if args.tasks_file else None, run_dir=run_dir,
            limits=BudgetLimits(max_tasks=args.max_tasks, max_rollouts=args.max_rollouts,
                                max_tokens=args.max_tokens, max_aiu=args.max_aiu, max_minutes=args.max_minutes),
            night_date=args.date, run_attempt=_attempt(), base_sha=args.base_sha, lessons_hook=lessons)
        sources = [parse_source(s) for s in args.lessons_source] if lessons else []
        deps = build_deps(Profile(args.profile), cfg, args.agl_export,
                          lessons_dir=args.lessons_dir.resolve() if args.lessons_dir else None,
                          lessons_sources=sources)
    except (WiringError, RegistryError, ValueError) as exc:
        print(f"sleep: setup failed: {exc}", file=sys.stderr)
        _gh_output(accepted=False, ledger_update=False, status="error")
        return 1
    try:
        _model_preflight(Profile(args.profile))
    except ModelPreflightError as exc:
        print(f"sleep: model preflight failed: {exc}", file=sys.stderr)
        _gh_output(accepted=False, ledger_update=False, status="error")
        return 1
    tel = _telemetry(args.profile, run_dir)
    try:
        result = run_night(cfg, deps)
    finally:
        if tel is not None:
            try:
                tel.shutdown()
            except Exception as exc:  # noqa: BLE001
                print(f"sleep: telemetry shutdown: {type(exc).__name__}", file=sys.stderr)
    _gh_output(accepted=result.accepted, ledger_update=result.ledger_update, status=result.status)
    print(json.dumps({"night_id": result.night_id, "status": result.status, "accepted": result.accepted,
                      "ledger_update": result.ledger_update, "targets": result.targets,
                      "bundle": str(result.bundle_dir), "error": result.error or None}, indent=2))
    return 1 if result.status == "error" else 0


def _dry_run(args: argparse.Namespace) -> int:
    from ci_lab.sleep.harvest import HarvestError, harvest, read_jsonl_rows
    from ci_lab.sleep.registry import RegistryError

    try:
        targets = _targets(args.targets)
        if args.tasks_file and len(targets) != 1:
            raise RegistryError("--tasks-file needs exactly one skill target")
        rows = read_jsonl_rows(args.agl_export)
        report = {t.name: harvest(args.tasks_file or (args.repo / t.tasks_file),
                                  rows if t is targets[0] else ()).summary() for t in targets}
    except (HarvestError, RegistryError) as exc:
        print(f"sleep: harvest rejected input: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


def _usage_gate(args: argparse.Namespace) -> int:
    from ci_lab.sleep.harvest import HarvestError
    from ci_lab.sleep.registry import RegistryError
    from ci_lab.sleep.usage import load_state, usage_gate

    threshold = args.threshold
    if threshold is None:
        env = os.environ.get("SLEEP_USAGE_THRESHOLD", "").strip()
        threshold = int(env) if env.isdigit() else 1
    force = args.force or os.environ.get("SLEEP_FORCE", "").lower() in ("1", "true", "yes")
    try:
        res = usage_gate(args.repo, _targets(args.targets), load_state(args.repo), threshold=threshold, force=force)
    except (HarvestError, RegistryError, ValueError) as exc:
        print(f"sleep: usage gate failed: {exc}", file=sys.stderr)
        _gh_output(run=False, new_reviewed=0)
        return 1
    _gh_output(run=res["run"], new_reviewed=res["new_reviewed"])
    print(json.dumps(res, indent=2))
    return 0


def _harvest_usage(args: argparse.Namespace) -> int:
    from ci_lab.sleep.harvest import HarvestError
    from ci_lab.sleep.night import git_head
    from ci_lab.sleep.registry import RegistryError
    from ci_lab.sleep.usage import harvest_usage, merge_sources, open_pending_pr, write_usage_bundle

    now = datetime.now(UTC)
    date = args.date or now.strftime("%Y%m%d")
    since = now.timestamp() - args.since_hours * 3600 if args.since_hours else None
    repo = args.repo.resolve()
    try:
        res = harvest_usage(repo, _targets(args.targets), merge_sources(args.source), date=date,
                            since=since, max_new=args.max_new)
    except (HarvestError, RegistryError, ValueError, OSError) as exc:
        print(f"sleep: usage harvest failed: {exc}", file=sys.stderr)
        _gh_output(ledger_update=False, new_pending=0)
        return 1
    out: dict[str, Any] = res.summary()
    if args.bundle:
        manifest = write_usage_bundle(args.bundle.resolve(), res, base_sha=args.base_sha or git_head(repo),
                                      run_attempt=_attempt())
        out["bundle"] = {"dir": str(args.bundle), "ledger_update": manifest["ledger_update"]}
    if args.open_pr:
        try:
            out["pr"] = open_pending_pr(repo, res)
        except subprocess.CalledProcessError as exc:
            print(f"sleep: opening the pending-review PR failed: {exc}", file=sys.stderr)
            return 1
    _gh_output(ledger_update=res.n_new > 0, new_pending=res.n_new)
    print(json.dumps(out, indent=2))
    return 0


def _redact_spans(args: argparse.Namespace) -> int:
    from ci_lab.sleep.traces import SpansJsonlSource, redact_spans_jsonl

    files = SpansJsonlSource(args.run_dir).files()
    stats = redact_spans_jsonl(files, args.out)
    print(json.dumps({"out": str(args.out), **stats}))
    return 0
