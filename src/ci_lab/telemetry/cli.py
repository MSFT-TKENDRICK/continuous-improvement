"""``ci-lab telemetry …`` and ``ci-lab dashboard …`` subcommands (never print secrets)."""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from typing import Any

from ci_lab.telemetry import aspire, importer, pull


def _out(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def _import(a: argparse.Namespace) -> int:
    try:
        _out(importer.import_files(a.paths, otlp_url=a.otlp_url, otlp_key=a.otlp_key,
                                   batch=a.batch, redact=not a.keep_sensitive))
    except (importer.TelemetryImportError, ValueError, OSError) as exc:
        return _err(str(exc))
    return 0


def _pull(a: argparse.Namespace) -> int:
    try:
        m = pull.pull(a.run, repo=a.repo, artifact=a.artifact,
                      allow_no_digest=a.allow_no_digest, force=a.force)
    except (pull.PullError, ValueError, OSError) as exc:
        return _err(str(exc))
    res: dict[str, Any] = {"pulled": m}
    if not a.no_import:
        if aspire.live_state() is None:
            res["imported"] = None
            res["note"] = "dashboard not running; cached only (ci-lab dashboard up, then re-run)"
        else:
            try:
                res["imported"] = importer.import_files(pull.files_of(m))
            except importer.TelemetryImportError as exc:
                return _err(str(exc))
    _out(res)
    return 0


def _imports(_a: argparse.Namespace) -> int:
    _out(pull.list_imports())
    return 0


def _up(a: argparse.Namespace) -> int:
    try:
        _out(aspire.up(version=a.version, rid=a.rid, trust_new=a.trust_new, grpc=a.grpc,
                       wait=a.wait))
    except aspire.DashboardError as exc:
        return _err(str(exc))
    return 0


def _down(_a: argparse.Namespace) -> int:
    res = aspire.down()
    _out(res)
    return 0 if res.get("stopped") or res.get("reason") in ("not running", "stale state removed") else 1


def _status(_a: argparse.Namespace) -> int:
    _out(aspire.status())
    return 0


def _url(a: argparse.Namespace) -> int:
    st = aspire.live_state()
    if not st:
        return _err("dashboard is not running (ci-lab dashboard up)")
    print(aspire.login_url(st) if a.with_token else st["ui_url"])
    return 0


def _open(_a: argparse.Namespace) -> int:
    st = aspire.live_state()
    if not st:
        return _err("dashboard is not running (ci-lab dashboard up)")
    webbrowser.open(aspire.login_url(st))
    print(f"opened {st['ui_url']}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    tel = subparsers.add_parser("telemetry", help="span JSONL import / CI artifact pull")
    tsub = tel.add_subparsers(dest="telemetry_command", required=True)
    imp = tsub.add_parser("import", help="replay span JSONL files/dirs into the dashboard via OTLP")
    imp.add_argument("paths", nargs="+", help="spans-*.jsonl files or directories")
    imp.add_argument("--otlp-url", help="OTLP/HTTP base URL (default: running dashboard)")
    imp.add_argument("--otlp-key", help="x-otlp-api-key for --otlp-url")
    imp.add_argument("--batch", type=int, default=importer.DEFAULT_BATCH)
    imp.add_argument("--keep-sensitive", action="store_true",
                     help="do not strip gen_ai content attributes/events")
    imp.set_defaults(func=_import)
    pl = tsub.add_parser("pull", help="download a CI run's spans artifact (gh), verify, cache, import")
    pl.add_argument("--run", required=True, help="GitHub Actions run id")
    pl.add_argument("--repo", help="owner/name (default: current repo)")
    pl.add_argument("--artifact", default=pull.DEFAULT_ARTIFACT)
    pl.add_argument("--allow-no-digest", action="store_true")
    pl.add_argument("--force", action="store_true", help="re-download even if cached")
    pl.add_argument("--no-import", action="store_true", help="cache only")
    pl.set_defaults(func=_pull)
    ls = tsub.add_parser("imports", help="list pulled CI runs (~/.ci-lab/imports)")
    ls.set_defaults(func=_imports)

    dash = subparsers.add_parser("dashboard", help="local Aspire dashboard (optional OTLP viewer)")
    dsub = dash.add_subparsers(dest="dashboard_command", required=True)
    u = dsub.add_parser("up", help="download/verify if needed and start on loopback")
    u.add_argument("--version", default=None)
    u.add_argument("--rid", default=None, choices=aspire.RIDS)
    u.add_argument("--trust-new", action="store_true",
                   help="TOFU: accept and record the sha256 of an unpinned RID/version")
    u.add_argument("--grpc", action="store_true", help="also open an OTLP/gRPC endpoint")
    u.add_argument("--wait", type=float, default=60.0, help="readiness timeout (s)")
    u.set_defaults(func=_up)
    dsub.add_parser("down", help="stop the recorded dashboard").set_defaults(func=_down)
    dsub.add_parser("status", help="state without secrets").set_defaults(func=_status)
    ur = dsub.add_parser("url", help="print the UI URL")
    ur.add_argument("--with-token", action="store_true", help="print the login URL incl. browser token")
    ur.set_defaults(func=_url)
    dsub.add_parser("open", help="open the UI (logged in) in the browser").set_defaults(func=_open)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ci_lab.telemetry")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
