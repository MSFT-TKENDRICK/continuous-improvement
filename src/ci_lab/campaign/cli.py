"""``ci-lab campaign new|calibrate|run|status|readjudicate|confirm|land``."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ci_lab.campaign.deps import CampaignDeps
from ci_lab.campaign.driver import Campaign, default_run_root
from ci_lab.contracts import Profile

COMMANDS = ("new", "calibrate", "run", "status", "readjudicate", "confirm", "land")


class IntegrationPending(RuntimeError):
    pass


def _parse_hyper(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise argparse.ArgumentTypeError(f"--hyper expects key=value, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def load_deps(profile: Profile, *, run_root: Path, ledger_dir: Path | None, repo: str,
              dry_run_publish: bool) -> CampaignDeps:
    """Build :class:`CampaignDeps` for a profile. Only ``fake`` is self-contained here;
    ``copilot``/``offline`` are wired by integration (M1/M4/M6/M7/M8a)."""
    if profile is Profile.FAKE:
        from ci_lab.campaign.fakes import fake_deps

        state = run_root / "_fake"
        return fake_deps(state, repo=repo, ledger_root=ledger_dir or state / "experiments")
    raise IntegrationPending(
        f"integration pending: profile {profile.value!r} needs the integrated modules "
        "(maf, gitops/ledger, rrsi, oes, meta); "
        "wire them in ci_lab.campaign.cli.load_deps. dry_run_publish="
        f"{dry_run_publish} would select GitHubPublisher(dry_run=...).")


def _setup_telemetry(profile: Profile, run_root: Path) -> Callable[[], None]:
    """``ci_lab.telemetry.setup("campaign", ...)`` (M12) if available; returns its shutdown."""
    try:
        from ci_lab import telemetry
    except ImportError:
        return lambda: None
    result = telemetry.setup("campaign", profile=profile, run_dir=run_root)
    shutdown = getattr(result, "shutdown", None) or getattr(telemetry, "shutdown", None)
    return shutdown if callable(shutdown) else (lambda: None)


def _main(args: argparse.Namespace) -> int:
    profile = Profile(args.profile)
    run_root = Path(args.run_dir) if args.run_dir else default_run_root()
    shutdown = _setup_telemetry(profile, run_root)
    try:
        return _dispatch(args, profile, run_root)
    finally:
        shutdown()


def _dispatch(args: argparse.Namespace, profile: Profile, run_root: Path) -> int:
    try:
        deps = load_deps(profile, run_root=run_root, ledger_dir=Path(args.ledger_dir) if args.ledger_dir else None,
                         repo=args.repo, dry_run_publish=args.dry_run_publish)
    except IntegrationPending as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    cmd = args.campaign_command
    if cmd == "new":
        camp = Campaign.new(args.cid, profile, _parse_hyper(args.hyper), deps=deps, run_root=run_root)
        out: Any = camp.status()
    else:
        camp = Campaign.load(args.cid, deps=deps, run_root=run_root)
        if cmd == "calibrate":
            out = {"delta": asyncio.run(camp.calibrate())}
        elif cmd == "run":
            out = asyncio.run(camp.run(args.rounds, Path(args.stop_file) if args.stop_file else None))
        elif cmd == "status":
            out = camp.status()
        elif cmd == "readjudicate":
            out = camp.readjudicate(args.eid)
        elif cmd == "confirm":
            out = asyncio.run(camp.confirm())
        else:
            out = camp.land()
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    if cmd == "readjudicate" and not out["consistent"]:
        return 1
    return 0


def register(sub: Any) -> None:
    parser = sub.add_parser("campaign", help="RRSI campaigns as OES experiments (MAF workflows)")
    csub = parser.add_subparsers(dest="campaign_command", required=True)
    for name in COMMANDS:
        p = csub.add_parser(name)
        p.add_argument("cid", help="campaign id")
        p.add_argument("--profile", choices=[p.value for p in Profile], default=Profile.COPILOT.value)
        p.add_argument("--dry-run-publish", action="store_true",
                       help="record intended gh/git calls instead of executing them")
        p.add_argument("--run-dir", help="run root (default: $CI_RUN_DIR or artifacts/ci-runs)")
        p.add_argument("--ledger-dir", help="ledger root (default: experiments/; fake: <run-dir>/_fake)")
        p.add_argument("--repo", default="example/harness", help="GitHub owner/repo for publishing")
        if name == "new":
            p.add_argument("--hyper", action="append", default=[], metavar="KEY=VALUE")
        if name == "run":
            p.add_argument("--rounds", type=int, default=None)
            p.add_argument("--stop-file", default=None)
        if name == "readjudicate":
            p.add_argument("eid")
        p.set_defaults(func=_main)
