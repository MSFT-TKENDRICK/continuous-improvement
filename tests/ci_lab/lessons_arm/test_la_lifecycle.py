from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from ci_lab.contracts import Edit
from ci_lab.lessons_arm import cli
from ci_lab.lessons_arm.bundle import dump_rule_file, read_rule_file
from ci_lab.lessons_arm.features import LessonFeatures
from ci_lab.lessons_arm.promote import PromotionError, gather_evidence, promote
from ci_lab.lessons_arm.prose import ProseError, apply_deletion, is_redundant, propose_deletions, redundant_vocabulary
from ci_lab.lessons_arm.retire import apply_ablation, exposure, retirement_candidates
from ci_lab.lessons_arm.strategy import COMMIT_TRAILER
from ci_lab.lessons_arm.synth import synth_prior_call
from ci_lab.rulespec import Fingerprint, LessonCluster, LessonEntry, RuleSpec

RULE_FILE = "harness/guards/refund.yaml"
SKILL = "harness/skills/refunds.md"
SKILL_TEXT = """# Refunds

## Order lookup
Issue_refund requires a prior successful lookup_order for the same order_id. Refund requests need a reason.
- Call lookup_order for this order_id first, then retry issue_refund.
- If the lookup fails, escalate to a human agent.

```
issue_refund requires lookup_order order_id
```

## Other
Issue_refund requires a prior successful lookup_order for the same order_id.
"""


def git(wt: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=wt, check=True, capture_output=True, text=True).stdout.strip()


def make_rule(mode: str = "shadow", cid: str = "refund") -> RuleSpec:
    c = LessonCluster(id=cid, fingerprint=Fingerprint(pin="p", oracle_rules=("refund.ineligible_order",)),
                      members=("a", "b"), families=("f1", "f2"), slices=("s1", "s2"), route="R2")
    f = LessonFeatures(kind="prior_call", target_tool="issue_refund", prior_tool="lookup_order",
                       subject_arg="order_id")
    return synth_prior_call(c, f).model_copy(update={"mode": mode})


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, RuleSpec]:
    wt = tmp_path / "wt"
    (wt / "harness" / "guards").mkdir(parents=True)
    (wt / "harness" / "skills").mkdir(parents=True)
    rule = make_rule()
    (wt / RULE_FILE).write_text(dump_rule_file([rule]), encoding="utf-8", newline="")
    (wt / SKILL).write_text(SKILL_TEXT, encoding="utf-8", newline="")
    git(wt, "init", "-q")
    git(wt, "config", "user.name", "t")
    git(wt, "config", "user.email", "t@x")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "base")
    return wt, rule


def decision(rule: RuleSpec, i: int, **kw: object) -> dict:
    return {"rule_id": rule.id, "rule_version": rule.version, "mode": "shadow", "action": "block",
            "enforced": False, "step_index": 1, "target": "issue_refund", "attempt_digest": f"d{i}", **kw}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def evidence(tmp: Path, rule: RuleSpec, *, opp: int = 400, fires: int = 20, tp: int = 12, fp: int = 0,
             nights: int = 5, intents: tuple[str, ...] = ()) -> tuple[Path, Path]:
    rows: list[dict] = []
    for n in range(nights):
        extra = {"intent": intents[n % len(intents)]} if intents else {}
        rows.append({"kind": "opportunity", "n": opp // nights, "night": f"n{n}", **extra})
    rows += [decision(rule, i, night=f"n{i % nights}") for i in range(fires)]
    rows.append(decision(rule.model_copy(update={"version": 9}), 999))  # stale version ignored
    rows.append(decision(rule.model_copy(update={"id": "other.rule"}), 998))
    labels = [{"attempt_digest": f"d{i}", "label": "tp"} for i in range(tp)]
    labels += [{"attempt_digest": f"d{tp + i}", "label": "false_positive"} for i in range(fp)]
    labels.append({"attempt_digest": "unknown", "label": "fp"})  # not a fire of this rule
    return write_jsonl(tmp / "dec.jsonl", rows), write_jsonl(tmp / "labels.jsonl", labels)


def test_promote_writes_patch_and_branch_without_touching_worktree(repo, tmp_path):
    wt, rule = repo
    dec, lab = evidence(tmp_path, rule)
    head = git(wt, "rev-parse", "HEAD")
    res = promote(rule.id, dec, lab, repo=wt, branch="promote/refund")
    assert res.verdict == "promote", res.reasons
    assert res.evidence["opportunities"] == 400 and res.evidence["fires"] == 20
    assert res.evidence["adjudicated_positives"] == 12 and res.evidence["stale_version_decisions"] == 1
    assert "-  mode: shadow" in res.patch and "+  mode: enforce" in res.patch and res.file == RULE_FILE
    assert git(wt, "rev-parse", "HEAD") == head and git(wt, "status", "--porcelain") == ""
    assert read_rule_file(wt / RULE_FILE).rules[0].mode == "shadow"
    assert git(wt, "rev-parse", "promote/refund") == res.commit
    assert git(wt, "diff", "--name-only", head, res.commit) == RULE_FILE
    assert COMMIT_TRAILER in git(wt, "log", "-1", "--format=%B", res.commit)
    assert "mode: enforce" in git(wt, "show", f"{res.commit}:{RULE_FILE}")
    (tmp_path / "p.diff").write_text(res.patch, encoding="utf-8", newline="")
    git(wt, "apply", "--check", str(tmp_path / "p.diff"))


def test_promote_holds_on_thin_evidence_and_expires_after_max_nights(repo, tmp_path):
    wt, rule = repo
    dec, lab = evidence(tmp_path, rule, opp=100, fires=5, tp=2)
    res = promote(rule.id, dec, lab, repo=wt)
    assert res.verdict == "hold" and res.patch is None
    assert any("opportunities 100 < 200" in r for r in res.reasons)
    dec, lab = evidence(tmp_path, rule, opp=300, fires=5, tp=2, nights=30)
    res = promote(rule.id, dec, lab, repo=wt)
    assert res.verdict == "expired" and "30 nights" in res.reasons[0]


def test_promote_fp_bound_and_intent_stratification(repo, tmp_path):
    wt, rule = repo
    dec, lab = evidence(tmp_path, rule, fp=6)
    res = promote(rule.id, dec, lab, repo=wt)
    assert res.verdict == "hold" and any("fp_ucb" in r for r in res.reasons)
    # stratified: aggregate passes but the sparse "exchange" stratum can't bound its FP rate
    dec, lab = evidence(tmp_path, rule, opp=400, nights=4, intents=("refund", "refund", "refund", "exchange"))
    ev = gather_evidence(rule, dec, lab)
    assert ev.stratified and set(ev.strata()) >= {"refund", "exchange"}
    res = promote(rule.id, dec, lab, repo=wt)
    assert res.verdict == "hold" and any("stratum exchange" in r for r in res.reasons)


def test_promote_errors(repo, tmp_path):
    wt, rule = repo
    dec, lab = evidence(tmp_path, rule)
    with pytest.raises(PromotionError):
        promote("lsn.nope.prior", dec, lab, repo=wt)
    (tmp_path / "bad.jsonl").write_text("{nope\n", encoding="utf-8")
    with pytest.raises(PromotionError):
        promote(rule.id, tmp_path / "bad.jsonl", None, repo=wt)


def test_cli_promote(repo, tmp_path, capsys):
    wt, rule = repo
    dec, lab = evidence(tmp_path, rule)
    parser = argparse.ArgumentParser()
    cli.register(parser.add_subparsers(dest="command"))
    out = tmp_path / "promote.patch"
    args = parser.parse_args(["lessons-arm", "promote", "--rule", rule.id, "--decisions", str(dec), "--labels",
                              str(lab), "--repo", str(wt), "--out", str(out)])
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "promote" and "+  mode: enforce" in out.read_text()
    args = parser.parse_args(["lessons-arm", "promote", "--rule", rule.id, "--decisions", str(dec), "--repo", str(wt)])
    assert args.func(args) == 1  # no labels -> 0 adjudicated positives -> hold


def test_prose_deletes_only_mechanically_redundant_sentences(repo):
    wt, _ = repo
    rule = make_rule("enforce")
    vocab = redundant_vocabulary([rule])
    assert is_redundant("Issue_refund requires a prior successful lookup_order for the same order_id.", vocab)
    assert not is_redundant("Refund requests need a reason.", vocab)
    assert not is_redundant("If the lookup fails, escalate to a human agent.", vocab)
    assert not is_redundant("You must always call lookup_order for this order_id before issue_refund.", vocab)
    entry = LessonEntry(lesson_id="refund", rule_ids=[rule.id], status="enforced",
                        prose_anchors=[f"{SKILL}#Order lookup", f"{SKILL}#Other"])
    prop = propose_deletions(entry, [rule], root=wt)
    assert [d.anchor for d in prop.deletions] == [f"{SKILL}#Order lookup"] * 2
    new = prop.new_text[SKILL]
    assert "Refund requests need a reason." in new and "escalate to a human agent" in new
    assert "retry issue_refund" not in new and "issue_refund requires lookup_order order_id" in new  # code kept
    other = new.split("## Other", 1)[1]
    assert "Issue_refund requires a prior successful" in other  # never empty a section
    assert prop.diff.startswith(f"--- a/{SKILL}")
    edit = apply_deletion(prop, wt)
    assert isinstance(edit, Edit) and edit.component == "skill" and edit.files == (SKILL,)
    assert "model pins" in edit.hypothesis and "OOD" in edit.hypothesis
    assert git(wt, "diff", "--name-only", f"{edit.commit}~1", edit.commit) == SKILL


def test_prose_refuses_unenforced_lessons(repo):
    wt, rule = repo
    with pytest.raises(ProseError, match="only enforced"):
        propose_deletions(LessonEntry(lesson_id="refund", rule_ids=[rule.id], status="shadow"), [rule], root=wt)
    with pytest.raises(ProseError, match="shadow"):
        propose_deletions(LessonEntry(lesson_id="refund", rule_ids=[rule.id], status="enforced"), [rule], root=wt)
    entry = LessonEntry(lesson_id="refund", rule_ids=[rule.id], status="enforced", prose_anchors=["../x.md#h"])
    with pytest.raises(ProseError, match="escapes"):
        propose_deletions(entry, [make_rule("enforce")], root=wt)


def test_retirement_is_exposure_based_and_ablation_commits(repo, tmp_path):
    wt, rule = repo
    busy = make_rule("enforce", cid="busy")
    rows = [{"kind": "opportunity", "target": "issue_refund", "n": 900},
            {"kind": "opportunity", "rule_id": "lsn.thin.prior", "n": 10}]
    rows += [decision(busy, i, action="block", enforced=True) for i in range(50)]
    thin = make_rule(cid="thin").model_copy(update={"target": "cancel_order"})
    dec = write_jsonl(tmp_path / "d.jsonl", rows)
    ex = exposure([rule, busy, thin], [dec])
    assert ex[rule.id].opportunities == 900 and ex[busy.id].fires == 50 and ex[busy.id].blocks == 50
    assert ex[thin.id].opportunities == 10
    cands = retirement_candidates([rule, busy, thin], [dec])
    assert [(c.rule_id, c.reason) for c in cands] == [(rule.id, "dormant")]  # thin: too little exposure
    rare = retirement_candidates([busy], [write_jsonl(tmp_path / "r.jsonl", [
        {"kind": "opportunity", "n": 100000}, decision(busy, 1)])])
    assert [c.reason for c in rare] == ["rare"]
    edit = apply_ablation(cands[0], wt)
    assert edit.files == (RULE_FILE,) and edit.component == "config" and not (wt / RULE_FILE).exists()
    assert git(wt, "diff", "--name-only", f"{edit.commit}~1", edit.commit) == RULE_FILE
