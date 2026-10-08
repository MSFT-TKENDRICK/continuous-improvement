"""``ci-lab lint`` (R5 dev lint) and ``ci-lab reflect`` (dev-transcript lesson mining, B8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("lint", help="lint the repo against lint/rules/*.yaml (R5; lint-arch output)")
    p.add_argument("--root", type=Path, default=None, help="repo root (default: git toplevel of cwd)")
    p.add_argument("--rules", type=Path, nargs="+", default=None, help="rule files (default: <root>/lint/rules/*.yaml)")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--staged", action="store_true", help="lint only staged files (index content)")
    scope.add_argument("--paths", nargs="+", default=(), help="lint only these files/dirs")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.set_defaults(func=_lint)

    r = subparsers.add_parser("reflect", help="mine local Copilot session transcripts for repeated corrections (opt-in)")
    r.add_argument("--source", choices=("copilot-sessions",), required=True)
    r.add_argument("--i-consent-local-mining", dest="consent", action="store_true",
                   help="required: confirms local-only mining of your own session transcripts (B8)")
    r.add_argument("--sessions-dir", type=Path, default=Path("~/.copilot/session-state"))
    r.add_argument("--root", type=Path, default=None, help="repo root; only sessions whose cwd is under it are read")
    r.add_argument("--out", type=Path, default=None, help="default: <root>/artifacts/reflect/proposals.yaml")
    r.add_argument("--min-sessions", type=int, default=2, help="convergence: distinct sessions per lesson")
    r.add_argument("--keep", action="store_true", help="also persist typed records under artifacts/reflect/")
    r.add_argument("--draft-pr", action="store_true",
                   help="commit proposed lint rules to a local lessons/reflect-<date> branch and print the gh command")
    r.set_defaults(func=_reflect)


def _lint(args: argparse.Namespace) -> int:
    from ci_lab.lint.engine import format_json, format_text, repo_root, run
    from ci_lab.lint.spec import RuleLoadError

    root = Path(args.root).resolve() if args.root else repo_root()
    try:
        res = run(root, staged=args.staged, paths=args.paths, rule_paths=args.rules)
    except RuleLoadError as e:
        for err in e.errors:
            print(f"[LINT][ERROR] rule load: {err}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"[LINT][ERROR] {e}", file=sys.stderr)
        return 2
    print(format_json(res) if args.format == "json" else format_text(res))
    return res.exit_code


def _reflect(args: argparse.Namespace) -> int:
    from ci_lab.lint.engine import repo_root
    from ci_lab.lint.reflect import ReflectError, reflect_main

    if not args.consent:
        print("reflect: refusing to read transcripts without --i-consent-local-mining (design §13.6 B8)",
              file=sys.stderr)
        return 2
    root = Path(args.root).resolve() if args.root else repo_root()
    try:
        return reflect_main(root=root, sessions_dir=args.sessions_dir.expanduser(), out=args.out,
                            min_sessions=args.min_sessions, keep=args.keep, draft_pr=args.draft_pr)
    except ReflectError as e:
        print(f"reflect: {e}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ci_lab.lint")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
