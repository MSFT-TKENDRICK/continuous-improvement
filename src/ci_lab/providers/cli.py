"""``ci-lab copilot-serve`` subcommand."""

from __future__ import annotations

import argparse
from typing import Any


def register(sub: Any) -> None:
    p = sub.add_parser("copilot-serve", help="serve a Copilot model as a loopback, tool-less OpenAI-compatible API",
                       description="Serve a GitHub Copilot model at http://<host>:<port>/v1 (chat completions "
                                   "only, no tools, no streaming). A random bearer key is written to --key-file.")
    p.add_argument("--host", default="127.0.0.1", help="loopback address to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True, help="Copilot model id, e.g. gpt-5-mini")
    p.add_argument("--key-file", required=True, help="where to write the bearer key")
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from ci_lab.providers.serve import check_loopback, serve

    try:
        check_loopback(args.host)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    # Long-lived service: trace context arrives per request (obs.use_carrier in serve.py), not via
    # TRACEPARENT at startup. Exporters come from ci_lab.telemetry.setup (M12) when installed.
    serve(host=args.host, port=args.port, model=args.model, key_file=args.key_file,
          reasoning_effort=args.reasoning_effort, timeout_s=args.timeout_s)
    return 0
