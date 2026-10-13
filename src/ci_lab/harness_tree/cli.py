"""``ci-lab harness validate|metrics``: check and measure a self-hosted harness tree (``harness/``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("harness", help="validate or measure a harness tree (default: $CI_HARNESS_DIR, "
                                              "else <repo>/harness)")
    sub = p.add_subparsers(dest="harness_cmd", required=True)
    v = sub.add_parser("validate", help="check the tree against the frozen manifest (exit 1 on errors)")
    v.add_argument("--dir", type=Path, default=None, help="harness dir")
    v.add_argument("--json", action="store_true", help="print {ok, root, errors} as JSON")
    v.set_defaults(func=_validate)
    m = sub.add_parser("metrics", help="surface/simplicity metrics of the tree, per manifest component")
    m.add_argument("--dir", type=Path, default=None, help="harness dir")
    m.add_argument("--json", action="store_true", help="print the metrics as JSON")
    m.set_defaults(func=_metrics)


def _root(args: argparse.Namespace) -> Path:
    from ci_lab.harness_tree import default_root

    return Path(args.dir) if args.dir is not None else default_root()


def _validate(args: argparse.Namespace) -> int:
    from ci_lab.harness_tree import HarnessTree, HarnessTreeError

    root = _root(args)
    try:
        errors = HarnessTree(root).validate()
    except HarnessTreeError as exc:
        errors = [str(exc)]
    if args.json:
        print(json.dumps({"ok": not errors, "root": str(root), "errors": errors}, indent=2))
    elif errors:
        for e in errors:
            print(f"[HARNESS][ERROR] {e}", file=sys.stderr)
    else:
        print(f"[HARNESS] {root}: valid")
    return 1 if errors else 0


def _metrics(args: argparse.Namespace) -> int:
    from ci_lab.harness_tree import HarnessTree
    from ci_lab.metrics.simplicity import surface_metrics

    root = _root(args)
    if not root.is_dir():
        print(f"[HARNESS][ERROR] harness dir not found: {root}", file=sys.stderr)
        return 2
    metrics = surface_metrics(root, HarnessTree(root).manifest["components"])
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        for k, val in sorted(metrics.items()):
            print(f"{k}\t{val:g}")
    return 0
