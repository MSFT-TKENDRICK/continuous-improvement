"""``ci-lab sleep run|dry-run`` (registered by ``ci_lab.cli``). Heavy imports stay lazy."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    run.add_argument("--date", default=None, help="night date yyyymmdd (default: UTC today)")
    run.add_argument("--base-sha", default=None, help="base commit (default: git rev-parse HEAD)")
    run.set_defaults(func=_run)

    dry = ssub.add_parser("dry-run", help="harvest + validate tasks only (no model calls)")
    _common(dry)
    dry.set_defaults(func=_dry_run)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--tasks-file", type=Path, default=None)
    p.add_argument("--agl-export", type=Path, action="append", default=[],
                   help="AGL journal export JSONL (evolve split only); repeatable")


def _gh_output(**values: Any) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in values.items():
            fh.write(f"{k}={str(v).lower() if isinstance(v, bool) else v}\n")


def _run(args: argparse.Namespace) -> int:
    from ci_lab.contracts import Profile
    from ci_lab.sleep.budget import BudgetLimits
    from ci_lab.sleep.night import SleepConfig, run_night
    from ci_lab.sleep.wiring import WiringError, build_deps

    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    cfg = SleepConfig(
        repo_root=args.repo.resolve(), out_dir=args.out.resolve(), profile=args.profile,
        tasks_file=args.tasks_file.resolve() if args.tasks_file else None,
        limits=BudgetLimits(max_tasks=args.max_tasks, max_rollouts=args.max_rollouts,
                            max_tokens=args.max_tokens, max_aiu=args.max_aiu, max_minutes=args.max_minutes),
        night_date=args.date, run_attempt=int(attempt) if attempt.isdigit() else 1, base_sha=args.base_sha)
    try:
        deps = build_deps(Profile(args.profile), cfg, args.agl_export)
    except WiringError as exc:
        print(f"sleep: wiring failed: {exc}", file=sys.stderr)
        _gh_output(accepted=False, ledger_update=False, status="error")
        return 1
    result = run_night(cfg, deps)
    _gh_output(accepted=result.accepted, ledger_update=result.ledger_update, status=result.status)
    print(json.dumps({"night_id": result.night_id, "status": result.status, "accepted": result.accepted,
                      "ledger_update": result.ledger_update, "bundle": str(result.bundle_dir),
                      "error": result.error or None}, indent=2))
    return 1 if result.status == "error" else 0


def _dry_run(args: argparse.Namespace) -> int:
    from ci_lab.sleep.harvest import HarvestError, harvest, read_jsonl_rows
    from ci_lab.sleep.night import TASKS_REL

    tasks_file = args.tasks_file or (args.repo / TASKS_REL)
    try:
        res = harvest(tasks_file, read_jsonl_rows(args.agl_export))
    except HarvestError as exc:
        print(f"sleep: harvest rejected input: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(res.summary(), indent=2))
    return 0
