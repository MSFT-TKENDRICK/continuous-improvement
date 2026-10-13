"""``ci-lab chat serve``: the experiment-designer AG-UI server (docs/chat.md)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROFILES = ("copilot", "fake")


def _main(args: argparse.Namespace) -> int:
    import logging

    from ci_lab.campaign.driver import default_run_root
    from ci_lab.chat.server import ServeError, serve
    from ci_lab.chat.tools import ChatConfig

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    repo_root = Path.cwd()
    config = ChatConfig(
        run_root=Path(args.run_dir) if args.run_dir else default_run_root(),
        chat_dir=Path(args.chat_dir) if args.chat_dir else repo_root / "artifacts" / "chat",
        campaign_profile=args.campaign_profile or args.profile,
        repo=args.repo,
        ledger_dir=Path(args.ledger_dir) if args.ledger_dir else None,
        repo_root=repo_root,
        dry_run_launch=args.dry_run_launch,
        dry_run_publish=not args.live_publish,
    )
    try:
        return serve(config, profile=args.profile, host=args.host, port=args.port, model=args.model,
                     stdin_watch=not args.no_stdin_watch)
    except ServeError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr, flush=True)
        return 2


def register(sub: Any) -> None:
    parser = sub.add_parser("chat", help="experiment-designer chat (AG-UI server for the canvas)")
    csub = parser.add_subparsers(dest="chat_command", required=True)
    p = csub.add_parser("serve", help="serve the experiment_designer agent over AG-UI on loopback "
                                      "(needs $CI_CHAT_TOKEN)")
    p.add_argument("--host", default="127.0.0.1", help="loopback host: 127.0.0.1, localhost or ::1")
    p.add_argument("--port", type=int, default=0, help="port (0 = pick a free one; see the listening line)")
    p.add_argument("--profile", choices=PROFILES, default="copilot",
                   help="chat model: copilot (GitHub Copilot SDK) or fake (scripted, offline)")
    p.add_argument("--repo", default=None, help="GitHub OWNER/REPO for launched campaigns and the workflow")
    p.add_argument("--run-dir", default=None, help="campaign run root (default: $CI_RUN_DIR or artifacts/ci-runs)")
    p.add_argument("--dry-run-launch", action="store_true",
                   help="record launch argv instead of starting campaigns or dispatching workflows")
    p.add_argument("--no-stdin-watch", action="store_true", help="do not exit when stdin reaches EOF")
    p.add_argument("--campaign-profile", choices=("copilot", "fake", "offline"), default=None,
                   help="profile for locally launched campaigns (default: --profile)")
    p.add_argument("--chat-dir", default=None, help="drafts/launch state dir (default: artifacts/chat)")
    p.add_argument("--ledger-dir", default=None, help="campaign ledger root passed to launched campaigns")
    p.add_argument("--live-publish", action="store_true",
                   help="launched local campaigns publish for real (default: --dry-run-publish)")
    p.add_argument("--model", default=None, help="copilot model (default: $CI_CHAT_MODEL or gpt-5-mini)")
    p.set_defaults(func=_main)
