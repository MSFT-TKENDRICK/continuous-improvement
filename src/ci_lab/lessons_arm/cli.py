"""``ci-lab lessons-arm``: promote (N3), prose (N5), retire (N6).

# HOOK(integration): add ``"lessons_arm"`` to ``ci_lab.cli.COMMAND_MODULES``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ci_lab.rulespec import GUARDS_DIR, LESSON_REGISTRY, PROMOTE_FP_UCB, PROMOTE_MIN_OPPORTUNITIES


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("lessons-arm", help="guard arm lifecycle: promote / prose / retire")
    sub = p.add_subparsers(dest="lessons_arm_cmd", required=True)

    pr = sub.add_parser("promote", help="N3 shadow->enforce gate; writes a PR-ready patch (never merges)")
    pr.add_argument("--rule", required=True)
    pr.add_argument("--decisions", required=True, type=Path, help="GuardDecision + opportunity JSONL")
    pr.add_argument("--labels", type=Path, default=None, help="adjudications JSONL {attempt_digest,label,intent?}")
    pr.add_argument("--repo", type=Path, default=Path("."), help="harness repo/worktree root")
    pr.add_argument("--guards-dir", type=Path, default=None,
                    help=f"default <repo>/{GUARDS_DIR}")
    pr.add_argument("--epsilon", type=float, default=PROMOTE_FP_UCB)
    pr.add_argument("--out", type=Path, default=None, help="write the patch here")
    pr.add_argument("--branch", default=None, help="also create this local branch (git plumbing; no checkout)")
    pr.set_defaults(func=_promote)

    ps = sub.add_parser("prose", help="N5 propose deletion of prose made redundant by an enforced lesson")
    ps.add_argument("--lesson", required=True)
    ps.add_argument("--repo", type=Path, default=Path("."))
    ps.add_argument("--registry", type=Path, default=None, help=f"default <repo>/{LESSON_REGISTRY}")
    ps.add_argument("--out", type=Path, default=None, help="write the patch here")
    ps.set_defaults(func=_prose)

    rt = sub.add_parser("retire", help="N6 exposure-based retirement candidates for ablation arms")
    rt.add_argument("--decisions", required=True, type=Path, nargs="+")
    rt.add_argument("--repo", type=Path, default=Path("."))
    rt.add_argument("--min-opportunities", type=int, default=PROMOTE_MIN_OPPORTUNITIES)
    rt.add_argument("--max-fire-ucb", type=float, default=None)
    rt.set_defaults(func=_retire)


def _guards(repo: Path) -> Path:
    from ci_lab.domain.layout import repo_guards_dir

    return repo_guards_dir(repo)


def _all_rules(guards: Path) -> list[Any]:
    from .bundle import read_rule_file, rule_files

    return [r for p in rule_files(guards) for r in read_rule_file(p).rules]


def _promote(a: Any) -> int:
    from .promote import PromotionError, promote

    try:
        res = promote(a.rule, a.decisions, a.labels, repo=a.repo, guards_dir=a.guards_dir, epsilon=a.epsilon,
                      branch=a.branch)
    except PromotionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if res.patch and a.out:
        a.out.write_text(res.patch, encoding="utf-8", newline="")
    print(json.dumps(res.to_json(), indent=2, sort_keys=True))
    return 0 if res.verdict == "promote" else 1


def _prose(a: Any) -> int:
    from ci_lab.lessons.registry import Registry

    from .prose import ProseError, propose_deletions

    reg = Registry.load(a.registry or a.repo / LESSON_REGISTRY)
    entry = reg.get(a.lesson)
    if entry is None:
        print(f"error: lesson {a.lesson!r} not in registry", file=sys.stderr)
        return 2
    try:
        prop = propose_deletions(entry, _all_rules(_guards(a.repo)), root=a.repo)
    except ProseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if prop.diff and a.out:
        a.out.write_text(prop.diff, encoding="utf-8", newline="")
    print(json.dumps({"lesson_id": prop.lesson_id, "files": list(prop.files), "kept": prop.kept,
                      "deletions": [d.__dict__ for d in prop.deletions], "hypothesis": prop.hypothesis()},
                     indent=2))
    return 0 if prop.deletions else 1


def _retire(a: Any) -> int:
    from .retire import DEFAULT_MAX_FIRE_UCB, retirement_candidates

    cands = retirement_candidates(_all_rules(_guards(a.repo)), a.decisions,
                                  min_opportunities=a.min_opportunities,
                                  max_fire_ucb=DEFAULT_MAX_FIRE_UCB if a.max_fire_ucb is None else a.max_fire_ucb)
    print(json.dumps([c.__dict__ | {"hypothesis": c.hypothesis()} for c in cands], indent=2))
    return 0
