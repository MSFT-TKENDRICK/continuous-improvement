"""``ci-lab template init|doctor``: bootstrap and check a repository created from the template."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("template", help="bootstrap / check a repository created from this template")
    sub = p.add_subparsers(dest="template_command", required=True)

    from ci_lab.template.init import add_arguments
    i = sub.add_parser("init", help="make a fresh copy your own: CODEOWNERS, template state reset, marker "
                                    "(dry run unless --apply)")
    add_arguments(i)
    i.set_defaults(func=_init)

    d = sub.add_parser("doctor", help="offline readiness checks for a derived repository (exit 1 on failure)")
    d.add_argument("--root", type=Path, default=None,
                   help="repository root (default: git toplevel of cwd)")
    d.add_argument("--skip-lint", action="store_true", help="skip the `ci-lab lint` check")
    d.add_argument("--format", choices=("text", "json"), default="text")
    d.set_defaults(func=_doctor)


def _init(args: argparse.Namespace) -> int:
    from ci_lab.template.init import run
    return run(args)


def _doctor(args: argparse.Namespace) -> int:
    from ci_lab.template.doctor import run
    return run(args)
