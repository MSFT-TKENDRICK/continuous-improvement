"""``ci-lab oes validate <paths/globs...> [--json] [--look-ledger FILE]``."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from .validate import read_look_ledger, validate_file


def expand(patterns: list[str]) -> list[Path]:
    """Files, directories (recursive ``*.json``) and globs (``**`` supported); de-duplicated, ordered."""
    seen: dict[str, Path] = {}
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            hits = sorted(p.rglob("*.json"))
        elif p.is_file():
            hits = [p]
        else:
            hits = [Path(h) for h in sorted(glob.glob(pattern, recursive=True)) if Path(h).is_file()]
        for h in hits:
            seen.setdefault(str(h.resolve()), h)
    return list(seen.values())


def run_validate(args: argparse.Namespace) -> int:
    files = expand(args.paths)
    try:
        looks = read_look_ledger(args.look_ledger) if args.look_ledger else None
    except (OSError, ValueError, AttributeError) as exc:
        print(f"cannot read look ledger {args.look_ledger}: {exc}")
        return 1
    report: list[dict[str, Any]] = [{"path": str(f), "errors": validate_file(f, look_counts=looks)} for f in files]
    ok = bool(files) and all(not r["errors"] for r in report)
    if args.json:
        print(json.dumps({"ok": ok, "files": report}, indent=2))
    else:
        if not files:
            print(f"no files matched: {' '.join(args.paths)}")
        for r in report:
            print(f"{'OK  ' if not r['errors'] else 'FAIL'} {r['path']}")
            for e in r["errors"]:
                print(f"    {e}")
        bad = sum(1 for r in report if r["errors"])
        print(f"{len(files) - bad}/{len(files)} valid")
    return 0 if ok else 1


def register(sub: argparse._SubParsersAction) -> None:
    oes = sub.add_parser("oes", help="Open Experiment Standard (OES 0.1.0) tools")
    oes_sub = oes.add_subparsers(dest="oes_command", required=True)
    v = oes_sub.add_parser("validate", help="validate OES envelopes (schema + ci_lab extensions + semantic rules)")
    v.add_argument("paths", nargs="+", help="files, directories or globs (e.g. 'experiments/**/*.json')")
    v.add_argument("--json", action="store_true", help="machine-readable report")
    v.add_argument("--look-ledger", metavar="JSONL", help="global held-out look ledger (lines with datasetHash)")
    v.set_defaults(func=run_validate)
