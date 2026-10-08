"""Shadow -> enforce promotion (design 13.6 N3).

Evidence:

* ``--decisions`` JSONL: ``GuardDecision`` lines recorded while the rule ran in shadow (extra keys
  ``night``/``slice`` and ``intent`` are optional) plus opportunity lines
  ``{"kind": "opportunity", "rule_id"?: str, "n": int, "intent"?: str, "night"?: str}``
  (an opportunity = the rule's target was attempted; lines without ``rule_id`` apply to every rule).
* ``--labels`` JSONL adjudications: ``{"attempt_digest": str, "label": "tp"|"fp", "intent"?: str,
  "rule_id"?: str}`` (``true_positive``/``violation`` and ``false_positive``/``benign`` accepted).

Only decisions for the rule's *current* ``version`` count. With any ``intent`` present the gate is
``ci_lab.lessons.stats.promotion_ok_stratified`` (per-intent FP UCB), else ``promotion_ok``. Shadow
lasts at most ``SHADOW_MAX_NIGHTS`` distinct nights; past that an unpromotable rule is ``expired``
(retirement candidate). On success the output is a git-apply-able patch (and optionally a local
branch built with git plumbing, so the working tree is never touched). Nothing is merged or pushed.
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ci_lab.rulespec import PROMOTE_FP_UCB, SHADOW_MAX_NIGHTS, GuardDecision, RuleSpec

from .bundle import dump_rule_file, load_rules, read_rule_file, rule_files
from .strategy import COMMIT_TRAILER, FALLBACK_IDENTITY

POSITIVE = frozenset({"tp", "true_positive", "violation", "positive"})
NEGATIVE = frozenset({"fp", "false_positive", "benign", "negative"})
DEFAULT_INTENT = "_all"


class PromotionError(RuntimeError):
    pass


@dataclass
class Evidence:
    opportunities: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fires: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    positives: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    false_positives: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    nights: set[str] = field(default_factory=set)
    stratified: bool = False
    stale_version: int = 0

    def strata(self) -> dict[str, tuple[int, int, int, int]]:
        keys = set(self.opportunities) | set(self.fires)
        return {k: (self.opportunities.get(k, 0), self.fires.get(k, 0), self.positives.get(k, 0),
                    self.false_positives.get(k, 0)) for k in sorted(keys)}

    def totals(self) -> tuple[int, int, int, int]:
        s = self.strata().values()
        return (sum(x[0] for x in s), sum(x[1] for x in s), sum(x[2] for x in s), sum(x[3] for x in s))


@dataclass
class PromotionResult:
    rule_id: str
    verdict: Literal["promote", "hold", "expired"]
    reasons: list[str]
    evidence: dict[str, Any]
    file: str | None = None
    patch: str | None = None
    branch: str | None = None
    commit: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "patch"} | {"has_patch": bool(self.patch)}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromotionError(f"{Path(path).name}:{n}: bad JSON: {exc}") from exc


def opportunity_applies(obj: dict[str, Any], rule: RuleSpec) -> bool:
    """An opportunity line counts for ``rule`` if it names the rule, else its target tool, else all rules."""
    if obj.get("rule_id") is not None:
        return obj["rule_id"] == rule.id
    if obj.get("target") is not None:
        return rule.target in ("*", obj["target"])
    return True


def gather_evidence(rule: RuleSpec, decisions: Path, labels: Path | None) -> Evidence:
    ev = Evidence()
    fired: dict[str, str] = {}  # attempt_digest -> intent
    for obj in _jsonl(decisions):
        intent = obj.get("intent")
        ev.stratified |= intent is not None
        intent = str(intent or DEFAULT_INTENT)
        night = obj.get("night") or obj.get("slice")
        if obj.get("kind") == "opportunity":
            if opportunity_applies(obj, rule):
                ev.opportunities[intent] += int(obj.get("n", 1))
                if night:
                    ev.nights.add(str(night))
            continue
        if obj.get("rule_id") != rule.id:
            continue
        d = GuardDecision.model_validate({k: v for k, v in obj.items() if k in GuardDecision.model_fields})
        if d.rule_version != rule.version:
            ev.stale_version += 1
            continue
        if night:
            ev.nights.add(str(night))
        ev.fires[intent] += 1
        fired[d.attempt_digest] = intent
    if labels is not None:
        seen: set[str] = set()
        for obj in _jsonl(labels):
            digest = obj.get("attempt_digest")
            if obj.get("rule_id") not in (None, rule.id) or digest not in fired or digest in seen:
                continue
            seen.add(digest)
            label = str(obj.get("label", "")).lower()
            intent = str(obj.get("intent") or fired[digest])
            ev.stratified |= obj.get("intent") is not None
            if label in POSITIVE:
                ev.positives[intent] += 1
            elif label in NEGATIVE:
                ev.false_positives[intent] += 1
    return ev


def _gate(ev: Evidence, epsilon: float) -> tuple[bool, list[str]]:
    from ci_lab.lessons.stats import promotion_ok, promotion_ok_stratified

    if ev.stratified:
        return promotion_ok_stratified(ev.strata(), epsilon=epsilon)
    return promotion_ok(*ev.totals(), epsilon=epsilon)


def find_rule(guards_dir: Path, rule_id: str) -> tuple[Path, RuleSpec]:
    for p in rule_files(guards_dir):
        for r in read_rule_file(p).rules:
            if r.id == rule_id:
                return p, r
    raise PromotionError(f"rule {rule_id!r} not found under {guards_dir}")


def _git(repo: Path, *args: str, env: dict[str, str] | None = None, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, input=stdin,
                          env={**os.environ, **(env or {})}).stdout.strip()


def _branch(repo: Path, rel: str, content: str, name: str, message: str) -> str:
    """Commit ``content`` at ``rel`` on top of HEAD into branch ``name`` without touching the worktree."""
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir"))
    index = git_dir / f"ci-promote-{os.getpid()}.index"
    env = {"GIT_INDEX_FILE": str(index)}
    try:
        _git(repo, "config", "user.email")
    except subprocess.CalledProcessError:
        env |= {"GIT_AUTHOR_NAME": FALLBACK_IDENTITY[0], "GIT_AUTHOR_EMAIL": FALLBACK_IDENTITY[1],
                "GIT_COMMITTER_NAME": FALLBACK_IDENTITY[0], "GIT_COMMITTER_EMAIL": FALLBACK_IDENTITY[1]}
    try:
        _git(repo, "read-tree", "HEAD", env=env)
        blob = _git(repo, "hash-object", "-w", "--stdin", stdin=content)
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}", env=env)
        tree = _git(repo, "write-tree", env=env)
        commit = _git(repo, "commit-tree", tree, "-p", "HEAD", "-m", message, env=env)
        _git(repo, "branch", name, commit)
        return commit
    finally:
        index.unlink(missing_ok=True)


def promote(rule_id: str, decisions: Path, labels: Path | None, *, repo: Path, guards_dir: Path | None = None,
            epsilon: float = PROMOTE_FP_UCB, branch: str | None = None,
            extractors: Iterable[Path] | None = None) -> PromotionResult:
    repo = Path(repo)
    if guards_dir is None:
        from ci_lab.domain.layout import repo_guards_dir

        gdir = repo_guards_dir(repo)
    else:
        gdir = Path(guards_dir)
    path, rule = find_rule(gdir, rule_id)
    ev = gather_evidence(rule, decisions, labels)
    opp, fires, pos, fp = ev.totals()
    summary = {"opportunities": opp, "fires": fires, "adjudicated_positives": pos, "false_positives": fp,
               "nights": len(ev.nights), "stratified": ev.stratified, "strata": ev.strata(),
               "stale_version_decisions": ev.stale_version, "epsilon": epsilon}
    if rule.mode == "enforce":
        return PromotionResult(rule_id, "hold", ["already enforced"], summary)
    ok, reasons = _gate(ev, epsilon)
    if not ok:
        verdict: Literal["hold", "expired"] = "expired" if len(ev.nights) >= SHADOW_MAX_NIGHTS else "hold"
        if verdict == "expired":
            reasons = [f"shadow exceeded {SHADOW_MAX_NIGHTS} nights without meeting N3", *reasons]
        return PromotionResult(rule_id, verdict, reasons, summary)

    rf = read_rule_file(path)
    new_rules = [r.model_copy(update={"mode": "enforce"}) if r.id == rule_id else r for r in rf.rules]
    old_text = path.read_text(encoding="utf-8")
    new_text = dump_rule_file(new_rules)
    # the flipped bundle must still load (conflicts, extractors, templates)
    tmp = path.with_name(f".promote-check-{path.name}")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        others = [p for p in rule_files(gdir) if p not in (path, tmp)]
        ext = list(extractors) if extractors is not None else sorted(gdir.glob("*extractor*.yaml"))
        load_rules([*others, tmp], ext)
    finally:
        tmp.unlink(missing_ok=True)
    try:
        rel = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        rel = path.name
    patch = "".join(difflib.unified_diff(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
                                         fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    msg = (f"guards: promote {rule_id} shadow -> enforce (N3)\n\n"
           f"opportunities={opp} fires={fires} adjudicated_positives={pos} false_positives={fp} "
           f"nights={len(ev.nights)} epsilon={epsilon}\n\n{COMMIT_TRAILER}\n")
    commit = _branch(repo, rel, new_text, branch, msg) if branch else None
    return PromotionResult(rule_id, "promote", [], summary, rel, patch, branch, commit)
