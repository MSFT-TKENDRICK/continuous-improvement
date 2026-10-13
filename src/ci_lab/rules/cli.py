"""``ci-lab rules check|eval`` (exposes ``register(subparsers)`` for the ci-lab CLI registry)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ci_lab.rulespec import Trajectory, TrajectoryStep


def _expand(paths: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted([*p.glob("*.yaml"), *p.glob("*.yml")], key=lambda x: x.name))
        else:
            out.append(p)
    return out


def _print_problems(details: Sequence[Any]) -> None:
    for d in details:
        print(f"[RULES][ERROR] {d.file}")
        print(f"  Violation: {d.violation}")
        if d.fix:
            print(f"  Fix: {d.fix}")


def _load(args: argparse.Namespace) -> Any:
    from ci_lab.rules import load_bundle, load_templates

    templates = load_templates(Path(args.templates)) if getattr(args, "templates", None) else None
    return load_bundle(_expand(args.rules), _expand(args.extractors or []), templates=templates)


def cmd_check(args: argparse.Namespace) -> int:
    from ci_lab.rules import RuleLoadError

    try:
        bundle = _load(args)
    except RuleLoadError as exc:
        _print_problems(exc.details)
        print(f"[RULES] {len(exc.details)} problem(s)")
        return 1
    print(f"[RULES][OK] {len(bundle.rules)} rule(s), {len(bundle.extractors)} extractor(s), "
          f"digest {bundle.digest}")
    return 0


def _read_steps(path: Path) -> list[TrajectoryStep]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "source" in data:
        return list(Trajectory.model_validate(data).steps)
    if isinstance(data, dict):
        data = data.get("steps", [])
    return TypeAdapter(list[TrajectoryStep]).validate_python(data)


def match_json(m: Any) -> dict[str, Any]:
    r = m.rule
    return {"rule": r.id, "version": r.version, "rung": r.rung, "on": r.on, "target": r.target,
            "action": r.action, "mode": r.mode, "severity": r.severity, "step_index": m.step_index,
            "message": m.message, "fix": m.fix, "see": m.see}


def cmd_eval(args: argparse.Namespace) -> int:
    from ci_lab.rules import RuleLoadError, evaluate_trajectory

    try:
        bundle = _load(args)
    except RuleLoadError as exc:
        _print_problems(exc.details)
        return 1
    try:
        steps = _read_steps(Path(args.trajectory))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"[RULES][ERROR] {args.trajectory}\n  Violation: unreadable trajectory: "
              f"{str(exc).splitlines()[0]}\n  Fix: Provide a JSON list of TrajectoryStep or a Trajectory.")
        return 1
    print(json.dumps([match_json(m) for m in evaluate_trajectory(bundle, steps)], indent=2, ensure_ascii=False))
    return 0


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("rules", help="validate and evaluate lessons-as-structure rule bundles")
    sub = p.add_subparsers(dest="rules_command", required=True)
    chk = sub.add_parser("check", help="validate rule files (schema, templates, RE2, conflicts)")
    chk.add_argument("rules", nargs="+", help="rule YAML files or directories")
    chk.add_argument("--extractors", nargs="*", default=[], help="extractor YAML files or directories")
    chk.add_argument("--templates", default=None, help="template catalog (default: trusted catalog)")
    chk.set_defaults(func=cmd_check)
    ev = sub.add_parser("eval", help="replay a trajectory against a bundle; print matches as JSON")
    ev.add_argument("--rules", nargs="+", required=True)
    ev.add_argument("--extractors", nargs="*", default=[])
    ev.add_argument("--templates", default=None)
    ev.add_argument("--trajectory", required=True, help="JSON: list of TrajectoryStep, {steps: [...]}, or Trajectory")
    ev.set_defaults(func=cmd_eval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci-lab")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
