"""Exposure-based retirement (design 13.6 N6).

A guard is a retirement candidate only after enough **exposure** (opportunities: its target was
attempted), never after a number of nights. With ``opportunities >= min_opportunities``:

* ``dormant``: zero fires, so the behaviour is learned or the rule is unreachable;
* ``rare``: the one-sided Clopper-Pearson upper bound of the fire rate is ``<= max_fire_ucb``.

Candidates feed an **adversarial ablation arm**. :func:`apply_ablation` removes the rule from its
guard file, and the arm's paired guard-off/on eval, run with adversarial cases, decides whether
delivered violations stay flat. Removal goes through the experiment, not around it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ci_lab.contracts import Edit
from ci_lab.rulespec import PROMOTE_MIN_OPPORTUNITIES, GuardDecision, RuleSpec

from .bundle import dump_rule_file, read_rule_file, rule_files
from .promote import _jsonl, opportunity_applies
from .strategy import COMMIT_TRAILER, GUARD_COMPONENT, Committer, GuardPathViolation, git_commit

DEFAULT_MAX_FIRE_UCB = 0.005


@dataclass(frozen=True)
class Exposure:
    rule_id: str
    version: int
    opportunities: int
    fires: int
    blocks: int


@dataclass(frozen=True)
class RetirementCandidate:
    rule_id: str
    version: int
    reason: Literal["dormant", "rare"]
    opportunities: int
    fires: int
    fire_ucb: float

    def hypothesis(self) -> str:
        return (f"Guard {self.rule_id}@v{self.version} is {self.reason} after {self.opportunities} opportunities "
                f"({self.fires} fires, fire-rate UCB {self.fire_ucb:.4f}); ablating it does not raise delivered "
                f"violations under adversarial cases (N6).")


def exposure(rules: Sequence[RuleSpec], decisions: Iterable[Path]) -> dict[str, Exposure]:
    """Per-rule exposure for the rules' *current* versions from GuardDecision + opportunity JSONL."""
    opp: dict[str, int] = defaultdict(int)
    fires: dict[str, int] = defaultdict(int)
    blocks: dict[str, int] = defaultdict(int)
    by_id = {r.id: r for r in rules}
    for path in decisions:
        for obj in _jsonl(Path(path)):
            if obj.get("kind") == "opportunity":
                for r in rules:
                    if opportunity_applies(obj, r):
                        opp[r.id] += int(obj.get("n", 1))
                continue
            rule = by_id.get(obj.get("rule_id", ""))
            if rule is None:
                continue
            d = GuardDecision.model_validate({k: v for k, v in obj.items() if k in GuardDecision.model_fields})
            if d.rule_version != rule.version:
                continue
            fires[rule.id] += 1
            blocks[rule.id] += d.action == "block" and d.enforced
    return {r.id: Exposure(r.id, r.version, opp[r.id], fires[r.id], blocks[r.id]) for r in rules}


def retirement_candidates(rules: Sequence[RuleSpec], decisions: Iterable[Path], *,
                          min_opportunities: int = PROMOTE_MIN_OPPORTUNITIES,
                          max_fire_ucb: float = DEFAULT_MAX_FIRE_UCB,
                          confidence: float = 0.95) -> list[RetirementCandidate]:
    from ci_lab.lessons.stats import cp_upper

    out: list[RetirementCandidate] = []
    for rid, ex in sorted(exposure(rules, decisions).items()):
        if ex.opportunities < min_opportunities:
            continue
        fires = min(ex.fires, ex.opportunities)
        ucb = cp_upper(fires, ex.opportunities, confidence, one_sided=True)
        if fires == 0:
            out.append(RetirementCandidate(rid, ex.version, "dormant", ex.opportunities, 0, ucb))
        elif ucb <= max_fire_ucb:
            out.append(RetirementCandidate(rid, ex.version, "rare", ex.opportunities, fires, ucb))
    return out


def apply_ablation(candidate: RetirementCandidate, worktree: Path, *, committer: Committer = git_commit,
                   guards_dir: Path | None = None) -> Edit:
    """Remove ``candidate``'s rule from its guard file (only under the guard dir the agent loads,
    default :func:`~ci_lab.domain.layout.repo_guards_dir`) and commit one Edit."""
    from ci_lab.domain.layout import repo_guards_dir

    wt = Path(worktree).resolve()
    guards = (Path(guards_dir) if guards_dir is not None else repo_guards_dir(wt)).resolve()
    for path in rule_files(guards):
        rf = read_rule_file(path)
        if not any(r.id == candidate.rule_id for r in rf.rules):
            continue
        if guards not in path.resolve().parents:
            raise GuardPathViolation(f"{path} is outside {guards}")
        rel = path.resolve().relative_to(wt).as_posix()
        keep = [r for r in rf.rules if r.id != candidate.rule_id]
        if keep:
            path.write_text(dump_rule_file(keep), encoding="utf-8", newline="")
        else:
            path.unlink()
        hypothesis = candidate.hypothesis()
        sha = committer(wt, [rel], f"guards: ablate {candidate.rule_id} (N6)\n\n{hypothesis}\n\n{COMMIT_TRAILER}")
        return Edit(component=GUARD_COMPONENT, hypothesis=hypothesis, files=(rel,), commit=sha)
    raise GuardPathViolation(f"rule {candidate.rule_id!r} not found under {guards}")
