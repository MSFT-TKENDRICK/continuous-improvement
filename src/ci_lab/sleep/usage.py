"""Usage-driven pending-review flow (design §11.3, C22) and the nightly usage gate.

* :func:`harvest_usage` — trace sources -> :func:`ci_lab.sleep.traces.pending_tasks` per skill
  target -> merged ``<tasks>.pending.jsonl`` content (``reviewed: false``; the header also says
  ``pending: true`` so :func:`ci_lab.sleep.harvest.load_reviewed_tasks` refuses the file).
* :func:`open_pending_pr` — branch ``exp/usage-<date>/tasks`` + DRAFT PR via an injectable
  ``run`` (``git``/``gh`` subprocess; tests pass a fake). It never merges: a human authors
  reference/judge, flips ``reviewed: true`` and moves rows into the reviewed tasks file.
* :func:`write_usage_bundle` — the same change as a ``kind: usage`` bundle for the privileged
  CI publish job (``scripts/sleep_publish.py``, C10).
* :func:`usage_gate` — new reviewed tasks since the last recorded night (state watermark) vs a
  threshold; gates the ``sleep-nightly`` evaluate job.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ci_lab.sleep.bundle import FileChange, make_patch, write_bundle
from ci_lab.sleep.harvest import HarvestError, reviewed_ids
from ci_lab.sleep.registry import SkillTarget
from ci_lab.sleep.traces import PENDING_FORMAT, Redactor, TraceSource, pending_tasks

STATE_REL = "experiments/sleep/state.json"
PENDING_NOTE = ("PENDING usage-derived task proposals (C22). Not used by SkillOpt-Sleep. To adopt a row: "
                "author reference + judge, set reviewed:true and move it to the reviewed tasks file.")
BOT = ("ci-lab-sleep[bot]", "ci-lab-sleep@users.noreply.github.com")

Run = Callable[..., subprocess.CompletedProcess]


def pending_rel(target: SkillTarget) -> str:
    return re.sub(r"\.jsonl$", ".pending.jsonl", target.tasks_file)


def read_pending(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for n, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip() or ln.lstrip().startswith("//"):
            continue
        row = json.loads(ln)
        if n == 1 and isinstance(row, dict) and "format" in row:
            if row.get("reviewed") is not False:
                raise HarvestError(f"{path}: pending header must say reviewed:false")
            continue
        if not isinstance(row, dict) or row.get("reviewed") is not False:
            raise HarvestError(f"{path}:{n}: pending rows must have reviewed:false")
        rows.append(row)
    return rows


def render_pending(rows: Sequence[Mapping[str, Any]], project: str) -> str:
    header = {"format": PENDING_FORMAT, "project": project, "reviewed": False, "pending": True, "note": PENDING_NOTE}
    lines = [json.dumps(header, ensure_ascii=False)]
    lines += [json.dumps(dict(r), ensure_ascii=False, sort_keys=True) for r in rows]
    return "\n".join(lines) + "\n"


@dataclass
class TargetUsage:
    target: SkillTarget
    new: list[dict[str, Any]]
    old_text: str | None
    new_text: str
    stats: dict[str, int]

    @property
    def path(self) -> str:
        return pending_rel(self.target)


@dataclass
class UsageResult:
    date: str
    targets: list[TargetUsage] = field(default_factory=list)

    @property
    def n_new(self) -> int:
        return sum(len(t.new) for t in self.targets)

    def changes(self) -> list[FileChange]:
        return [FileChange(t.path, t.old_text, t.new_text) for t in self.targets if t.new]

    def summary(self) -> dict[str, Any]:
        return {"date": self.date, "new_pending": self.n_new,
                "targets": {t.target.name: {"pending_file": t.path, "new": [r["id"] for r in t.new],
                                            "stats": t.stats} for t in self.targets}}


def harvest_usage(repo: Path, targets: Sequence[SkillTarget], sources: Sequence[TraceSource], *, date: str,
                  since: float | None = None, max_new: int = 25, redactor: Redactor | None = None) -> UsageResult:
    """Build new pending rows per target, skipping ids already pending or reviewed."""
    red = redactor or Redactor()
    res = UsageResult(date=date)
    for index, target in enumerate(targets):
        harvested = pending_tasks(sources, project=target.name, aliases=(target.owner_agent,),
                                  redactor=red, since=since, max_tasks=max_new,
                                  accept_unlabeled=index == 0)
        pend_path = repo / pending_rel(target)
        old_text = pend_path.read_text(encoding="utf-8") if pend_path.exists() else None
        existing = read_pending(pend_path)
        known = {str(r.get("id")) for r in existing} | set(reviewed_ids(repo / target.tasks_file))
        new = [t for t in harvested.tasks if t["id"] not in known]
        res.targets.append(TargetUsage(target=target, new=new, old_text=old_text,
                                       new_text=render_pending([*existing, *new], target.name),
                                       stats={**harvested.stats, "new": len(new),
                                              "already_known": len(harvested.tasks) - len(new)}))
    return res


def _body(res: UsageResult) -> str:
    lines = [f"Usage-derived **pending** SkillOpt-Sleep tasks for {res.date} (design §11.3, C22).", "",
             "Every row is `reviewed: false` and is ignored by SkillOpt-Sleep until a human:",
             "1. checks the (redacted) intent is a legitimate, non-adversarial request;",
             "2. authors `reference` + `judge` (known ops only);",
             "3. sets `reviewed: true` and moves the row into the reviewed tasks file.", "",
             "| target | new pending | traces | injection dropped |", "|---|---|---|---|"]
    for t in res.targets:
        lines.append(f"| {t.target.name} | {len(t.new)} | {t.stats.get('traces', 0)} | "
                     f"{t.stats.get('injection_dropped', 0)} |")
    lines += ["", "_Opened by `ci-lab sleep harvest-usage`; never auto-merged._"]
    return "\n".join(lines) + "\n"


def write_usage_bundle(out_dir: Path, res: UsageResult, *, base_sha: str, run_attempt: int = 1) -> dict[str, Any]:
    patch = make_patch(res.changes())
    return write_bundle(out_dir, patch=patch, experiment={"kind": "usage", **res.summary()},
                        results={"kind": "usage", "status": "pending_review", "pr_body": _body(res)},
                        base_sha=base_sha, night_id=f"usage-{res.date}-{int(run_attempt)}", date=res.date,
                        accepted=False, ledger_update=bool(patch), status="pending_review", kind="usage")


def _default_run(args: Sequence[str], **kw: Any) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), check=True, text=True, capture_output=True, **kw)


def open_pending_pr(repo: Path, res: UsageResult, *, run: Run = _default_run, base: str = "main",
                    push: bool = True) -> dict[str, Any]:
    """Write the pending files on ``exp/usage-<date>/tasks`` and open a DRAFT PR (never merge)."""
    if not re.fullmatch(r"\d{8}", res.date):
        raise ValueError("date must be yyyymmdd")
    if not res.n_new:
        return {"opened": False, "reason": "no new pending tasks"}
    branch = f"exp/usage-{res.date}/tasks"
    git = ["git", "-C", str(repo)]
    run([*git, "switch", "-c", branch])
    paths = []
    for t in res.targets:
        if t.new:
            p = repo / t.path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t.new_text, encoding="utf-8", newline="\n")
            paths.append(t.path)
    run([*git, "add", "--", *paths])
    run([*git, "-c", f"user.name={BOT[0]}", "-c", f"user.email={BOT[1]}", "commit", "--no-verify", "-m",
         (f"Propose {res.n_new} usage-derived pending sleep task(s) for {res.date}\n\n"
          f"Usage-Date: {res.date}\nReviewed: false\n")])
    if not push:
        return {"opened": False, "branch": branch, "reason": "push disabled"}
    run(["gh", "auth", "setup-git"], cwd=str(repo))
    run([*git, "push", "origin", f"HEAD:refs/heads/{branch}"])
    existing = run(["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "url", "--jq", ".[0].url"],
                   cwd=str(repo))
    url = (existing.stdout or "").strip()
    if url:
        return {"opened": False, "branch": branch, "url": url, "reason": "draft PR already open"}
    created = run(["gh", "pr", "create", "--draft", "--base", base, "--head", branch,
                   "--title", f"[usage] pending sleep tasks {res.date}", "--body-file", "-"],
                  cwd=str(repo), input=_body(res))
    return {"opened": True, "branch": branch, "url": (created.stdout or "").strip()}


# ------------------------------------------------------------------ nightly usage gate

def usage_gate(repo: Path, targets: Sequence[SkillTarget], state: Mapping[str, Any] | None, *,
               threshold: int = 1, force: bool = False) -> dict[str, Any]:
    """Count reviewed tasks not in the last night's watermark. No watermark = everything is new."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    marks = ((state or {}).get("watermark") or {}).get("task_ids") or {}
    per: dict[str, int] = {}
    for t in targets:
        seen = set(marks.get(t.name) or [])
        per[t.name] = len([i for i in reviewed_ids(repo / t.tasks_file) if i not in seen])
    total = sum(per.values())
    run = force or total >= threshold
    reason = ("forced (manual dispatch)" if force and total < threshold
              else f"{total} new reviewed task(s) since last night (threshold {threshold})")
    return {"run": run, "new_reviewed": total, "threshold": threshold, "targets": per, "reason": reason}


def load_state(repo: Path) -> dict[str, Any] | None:
    p = repo / STATE_REL
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def merge_sources(specs: Iterable[str]) -> list[TraceSource]:
    from ci_lab.sleep.traces import parse_source

    return [parse_source(s) for s in specs]
