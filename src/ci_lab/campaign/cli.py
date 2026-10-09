"""``ci-lab campaign new|calibrate|run|status|readjudicate|confirm|publish|land``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ci_lab.campaign.deps import CampaignDeps
from ci_lab.campaign.driver import Campaign, default_run_root
from ci_lab.contracts import Profile
from ci_lab.providers.models import ModelPreflightError

COMMANDS = ("new", "calibrate", "run", "status", "readjudicate", "confirm", "publish", "land")
GATED = ("calibrate", "run", "confirm", "publish", "land")  # governance launch gate (exit 3 when refused)


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
              dry_run_publish: bool, domain_name: str = "order_support") -> CampaignDeps:
    """Build :class:`CampaignDeps` for a profile: ``fake`` from :mod:`.fakes`, ``copilot`` /
    ``offline`` from :mod:`.wiring` (offline is network-free and never publishes)."""
    if profile is Profile.FAKE:
        from ci_lab.campaign.fakes import fake_deps

        state = run_root / "_fake"
        return fake_deps(state, repo=repo, ledger_root=ledger_dir or state / "experiments",
                         domain_name=domain_name)
    from ci_lab.campaign.wiring import NetworkPolicyError, wired_deps

    try:
        return wired_deps(profile, run_root=run_root, ledger_dir=ledger_dir, repo=repo,
                          dry_run_publish=dry_run_publish, domain_name=domain_name)
    except (NetworkPolicyError, ImportError) as exc:
        raise IntegrationPending(f"profile {profile.value!r}: {exc}") from exc


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
                         repo=args.repo, dry_run_publish=args.dry_run_publish,
                         domain_name=getattr(args, "domain", "order_support"))
    except IntegrationPending as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    cmd = args.campaign_command
    if getattr(args, "defer_publish", None):
        from ci_lab.publish.deferred import DeferredPublisher

        deps.publisher = DeferredPublisher(args.defer_publish)
    if cmd == "publish":
        from ci_lab.publish.deferred import replay_deferred

        camp = Campaign.load(args.cid, deps=deps, run_root=run_root)  # fails if the campaign is unknown
        if (refused := _launch_gate(camp, cmd, args, profile)) is not None:
            return refused
        out: Any = {"published": replay_deferred(args.requests, cid=args.cid, ledger=deps.ledger,
                                                 publisher=deps.publisher)}
    elif cmd == "new":
        camp = Campaign.new(args.cid, profile, _parse_hyper(args.hyper), deps=deps, run_root=run_root)
        out = camp.status()
    else:
        camp = Campaign.load(args.cid, deps=deps, run_root=run_root)
        if cmd in GATED and (refused := _launch_gate(camp, cmd, args, profile)) is not None:
            return refused
        try:
            out = _run_command(camp, cmd, args)
        except ModelPreflightError as exc:
            print(json.dumps({"error": "model preflight failed", "detail": str(exc)}, indent=2))
            print(f"ci-lab campaign {cmd}: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    if cmd == "readjudicate" and not out["consistent"]:
        return 1
    return 0


def _launch_gate(camp: Campaign, cmd: str, args: argparse.Namespace, profile: Profile) -> int | None:
    """ACS ``agent_startup`` on the ``campaign`` policy (:mod:`ci_lab.governance.campaign`);
    returns exit code 3 when refused or held for approval, else ``None``."""
    from ci_lab.governance.campaign import LaunchRefused, arm_scopes, check_launch

    domain = camp.deps.domain
    publish = cmd in ("run", "publish", "land") and not getattr(args, "defer_publish", None)
    dry_run = profile is not Profile.COPILOT or bool(getattr(args, "dry_run_publish", False))
    try:
        asyncio.run(check_launch(camp.cid, arms=arm_scopes(domain.component_globs, domain.surface_globs),
                                 publish=publish, dry_run=dry_run, budget_exhausted=camp.sre_exhausted()))
    except LaunchRefused as exc:
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
        print(f"ci-lab campaign {cmd}: {exc}", file=sys.stderr)
        return 3
    return None


def _run_command(camp: Campaign, cmd: str, args: argparse.Namespace) -> Any:
    if cmd == "calibrate":
        return {"delta": asyncio.run(camp.calibrate())}
    if cmd == "run":
        return asyncio.run(camp.run(args.rounds, Path(args.stop_file) if args.stop_file else None))
    if cmd == "status":
        return camp.status()
    if cmd == "readjudicate":
        return camp.readjudicate(args.eid)
    if cmd == "confirm":
        return asyncio.run(camp.confirm())
    return camp.land()


def register(sub: Any) -> None:
    from ci_lab.domain import DEFAULT_DOMAIN, DOMAIN_CHOICES

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
        p.add_argument("--domain", choices=DOMAIN_CHOICES, default=DEFAULT_DOMAIN)
        if name == "new":
            p.add_argument("--hyper", action="append", default=[], metavar="KEY=VALUE")
        if name == "run":
            p.add_argument("--rounds", type=int, default=None)
            p.add_argument("--stop-file", default=None)
            p.add_argument("--defer-publish", metavar="PATH", default=None,
                           help="queue publish requests to PATH (JSONL) for `campaign publish`")
        if name == "publish":
            p.add_argument("--requests", required=True, metavar="PATH",
                           help="deferred publish requests written by `campaign run --defer-publish`")
        if name == "readjudicate":
            p.add_argument("eid")
        p.set_defaults(func=_main)
