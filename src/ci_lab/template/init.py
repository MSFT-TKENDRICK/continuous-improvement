"""``ci-lab template init``: turn a fresh copy of the template into your own repository (STDLIB ONLY).

Dry run by default (prints the plan); ``--apply`` writes. Idempotent: a second run with the same inputs
plans nothing. Refuses to run in the template repository itself unless ``--force``. Pure Python, no
network: the repository comes from ``--repo`` or ``git remote get-url origin``.

What it changes (docs/template.md):

* ``.github/CODEOWNERS``: owners of every rule still owned by the template owner (or by the previous
  ``--owners``) become ``--owners``; rules a human customized are kept and reported.
* template history: campaign ledgers, sleep night envelopes/lesson proposals, pending harvested tasks,
  the sleep state counters and the lessons registry are reset. They describe the template's own runs
  and reference branches/PRs that do not exist in the copy.
* ``--reset-state`` additionally empties the example agent's reviewed sleep tasks and the held-out look
  ledger (only do that when you also replace the frozen ASSERT test sets: the look budget is keyed by
  dataset hash). Suites, frozen test sets, schemas and rules are never touched.
* ``.github/template.yml``: the marker (``role: derived``, ``initialized: true``, source template,
  commit, version and date).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ci_lab.template.codeowners import (
    CODEOWNERS_REL,
    OwnersError,
    parse_owners,
    rewrite_owners,
)
from ci_lab.template.marker import (
    FORMAT,
    MARKER_REL,
    MarkerError,
    read_marker,
    render_marker,
    template_identity,
)

REPO_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$")
_REMOTE_RE = re.compile(r"[:/]([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/*$")

SLEEP_STATE_REL = "experiments/sleep/state.json"
SLEEP_TASKS_REL = "experiments/sleep/tasks.jsonl"
HARNESS_SLEEP_TASKS_RELS = (
    "experiments/sleep/harness-editing.jsonl",
    "experiments/sleep/trace-triage.jsonl",
)
LOOKS_REL = "experiments/holdout-looks.jsonl"
REGISTRY_REL = "lessons/registry.yaml"
# Template history: (path, reason). Directories are removed with everything below them.
HISTORY_DIRS = (
    ("experiments/campaigns", "template campaign ledgers (stack/published refer to the template's PRs)"),
    ("experiments/sleep/nights", "template sleep-night envelopes"),
    ("experiments/sleep/envelopes", "template sleep-night envelopes"),
    ("experiments/sleep/lessons", "lesson proposals mined from the template's traces"),
)
PENDING_GLOB = "experiments/sleep/*.pending.jsonl"
DEFAULT_SLEEP_STATE = {"accepted_total": 0, "format": "ci_lab.sleep.state.v1", "history": [],
                       "last_night_id": None, "last_status": None, "night": 0}
EMPTY_REGISTRY = "schema_version: 1\nlessons: []\n"
TASKS_FORMAT = "skillopt_sleep.tasks.v1"


class InitError(Exception):
    """Bad input or an unreadable tree (exit 2)."""


class Refused(InitError):
    """Running here would damage the template (exit 2; ``--force`` overrides)."""


@dataclass
class Action:
    op: str            # "write" | "delete"
    path: str          # repo-relative POSIX path (directories end with "/")
    reason: str
    content: str | None = None
    files: int = 0     # files removed by a directory delete


@dataclass
class Plan:
    repo: str
    owners: tuple[str, ...]
    template_repository: str
    actions: list[Action] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"repo": self.repo, "owners": list(self.owners), "template_repository": self.template_repository,
                "actions": [{k: v for k, v in asdict(a).items() if k != "content"} for a in self.actions],
                "notes": self.notes}


@dataclass(frozen=True)
class Options:
    repo: str | None = None
    owners: Sequence[str] = ()
    reset_state: bool = False
    force: bool = False
    template_commit: str | None = None
    today: _dt.date | None = None


# ---------------------------------------------------------------- helpers


def _git(root: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30,
                           check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None


def repo_from_remote(url: str) -> str | None:
    m = _REMOTE_RE.search(url.strip())
    if not m:
        return None
    repo = f"{m.group(1)}/{m.group(2)}"
    return repo if REPO_RE.match(repo) else None


def origin_repo(root: Path) -> str | None:
    url = _git(root, "remote", "get-url", "origin")
    return repo_from_remote(url) if url else None


def _version(root: Path) -> str:
    try:
        return str(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def _inside(root: Path, rel: str) -> Path:
    path = root / rel
    base = root.resolve()
    if not path.resolve().is_relative_to(base):
        raise InitError(f"{rel} resolves outside the repository (symlink?); refusing")
    return path


def _read(root: Path, rel: str) -> str | None:
    path = _inside(root, rel)
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


def _count_files(path: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(path, followlinks=False))


def _tasks_header(existing: str) -> str:
    project = "harness"
    first = next((ln for ln in existing.splitlines() if ln.strip() and not ln.lstrip().startswith("//")), "")
    try:
        head = json.loads(first)
        if isinstance(head, dict) and head.get("format") == TASKS_FORMAT and head.get("project"):
            project = str(head["project"])
    except json.JSONDecodeError:
        pass
    header = {"format": TASKS_FORMAT, "project": project, "reviewed": True,
              "note": "Human-reviewed SkillOpt-Sleep tasks: one task per line, each \"reviewed\": true "
                      "(docs/sleep.md)."}
    return json.dumps(header, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- plan


def build_plan(root: Path, opts: Options) -> Plan:
    root = Path(root)
    try:
        marker = read_marker(root)
    except MarkerError as e:
        raise InitError(str(e)) from None
    template_repo, template_owners = template_identity(marker)
    derived = bool(marker and marker.get("role") == "derived")
    prev = dict(marker) if derived else {}

    repo = opts.repo or origin_repo(root)
    if not repo:
        raise InitError("cannot determine the repository: pass --repo owner/name (no parsable `origin` remote)")
    if not REPO_RE.match(repo):
        raise InitError(f"--repo must be owner/name, got {repo!r}")
    if repo.lower() == template_repo.lower() and not opts.force:
        raise Refused(f"{repo} is the template repository ({MARKER_REL}); `template init` only runs in a "
                      "repository created from it (--force overrides)")

    try:
        owners = parse_owners(opts.owners) if opts.owners else tuple(prev.get("owners") or ())
    except OwnersError as e:
        raise InitError(str(e)) from None
    if not owners:
        raise InitError('pass --owners "@org/team ..." (CODEOWNERS owners of the protected harness paths)')
    if set(owners) & set(template_owners) and not opts.force and list(owners) != prev.get("owners"):
        raise Refused(f"--owners includes the template owner {', '.join(template_owners)}; name your own "
                      "owners (--force if you really own both)")

    plan = Plan(repo=repo, owners=owners, template_repository=template_repo)
    if marker is None:
        plan.notes.append(f"{MARKER_REL} not found: assuming template {template_repo} owned by "
                          f"{' '.join(template_owners)}")

    # CODEOWNERS
    text = _read(root, CODEOWNERS_REL)
    if text is None:
        plan.notes.append(f"{CODEOWNERS_REL} not found: protected harness paths have no required reviewers")
    else:
        rw = rewrite_owners(text, {*template_owners, *prev.get("owners", ())}, owners)
        if rw.changed:
            plan.actions.append(Action("write", CODEOWNERS_REL, f"owners -> {' '.join(owners)} "
                                       f"({len(rw.changed)} rules)", content=rw.text))
        if rw.kept_custom:
            plan.notes.append(f"{CODEOWNERS_REL}: kept customized owners on {', '.join(rw.kept_custom)}")

    # template history
    for rel, reason in HISTORY_DIRS:
        path = _inside(root, rel)
        if path.is_dir() and not path.is_symlink():
            n = _count_files(path)
            plan.actions.append(Action("delete", rel + "/", reason, files=n))
        elif path.exists() or path.is_symlink():
            plan.actions.append(Action("delete", rel, reason, files=1))
    for path in sorted(root.glob(PENDING_GLOB)):
        rel = path.relative_to(root).as_posix()
        _inside(root, rel)
        plan.actions.append(Action("delete", rel, "unreviewed tasks harvested from the template's usage traces",
                                   files=1))
    state = _read(root, SLEEP_STATE_REL)
    if state is not None:
        try:
            current = json.loads(state)
        except json.JSONDecodeError:
            current = None
        if current != DEFAULT_SLEEP_STATE:
            plan.actions.append(Action("write", SLEEP_STATE_REL, "sleep night counters/history -> night 0",
                                       content=json.dumps(DEFAULT_SLEEP_STATE, indent=2, sort_keys=True) + "\n"))
    registry = _read(root, REGISTRY_REL)
    if registry is not None and _norm(registry).strip() != EMPTY_REGISTRY.strip():
        plan.actions.append(Action("write", REGISTRY_REL, "lessons learned from the template's traces -> empty",
                                   content=EMPTY_REGISTRY))

    if opts.reset_state:
        for rel in HARNESS_SLEEP_TASKS_RELS:
            tasks = _read(root, rel)
            if tasks is not None:
                header = _tasks_header(tasks)
                if _norm(tasks) != header:
                    plan.actions.append(Action("write", rel, "harness target's reviewed sleep tasks -> "
                                               "header only", content=header))
        if _inside(root, LOOKS_REL).is_file():
            plan.actions.append(Action("delete", LOOKS_REL, "held-out look ledger (budget is per dataset hash)",
                                       files=1))
            plan.notes.append(f"--reset-state removes {LOOKS_REL}: only do this if you also replace the frozen "
                              "ASSERT test sets, or the template's held-out looks are forgotten")

    # marker
    today = (opts.today or _dt.datetime.now(_dt.UTC).date()).isoformat()
    data: dict[str, Any] = {k: v for k, v in (marker or {}).items()}
    data.update({
        "format": FORMAT, "role": "derived", "initialized": True, "repository": repo, "owners": list(owners),
        "template_repository": template_repo, "template_owners": list(template_owners),
        "template_commit": opts.template_commit or prev.get("template_commit") or "unknown",
        "template_version": prev.get("template_version") or _version(root),
        "source_commit": prev.get("source_commit") or _git(root, "rev-parse", "HEAD") or "unknown",
        "initialized_on": prev.get("initialized_on") or today,
        "reset_state": bool(prev.get("reset_state")) or opts.reset_state,
    })
    if derived and any(data.get(k) != prev.get(k) for k in ("repository", "owners", "reset_state",
                                                            "template_commit")):
        data["updated_on"] = today
    rendered = render_marker(data)
    current_marker = _read(root, MARKER_REL)
    if current_marker is None or _norm(current_marker) != rendered:
        plan.actions.append(Action("write", MARKER_REL, "marker: role derived, initialized", content=rendered))
    return plan


# ---------------------------------------------------------------- apply


def _write(path: Path, content: str) -> None:
    newline = "\n"
    if path.is_file() and b"\r\n" in path.read_bytes():
        newline = "\r\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content.replace("\n", newline).encode("utf-8"))
    os.replace(tmp, path)


def _delete(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
        for name in filenames:
            Path(dirpath, name).unlink()
        for name in dirnames:
            d = Path(dirpath, name)
            d.unlink() if d.is_symlink() else d.rmdir()
    path.rmdir()


def apply_plan(root: Path, plan: Plan) -> None:
    for a in plan.actions:
        path = _inside(Path(root), a.path.rstrip("/"))
        if a.op == "write":
            assert a.content is not None
            _write(path, a.content)
        elif a.op == "delete":
            _delete(path)
        else:  # pragma: no cover
            raise InitError(f"unknown action {a.op}")


# ---------------------------------------------------------------- CLI


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", type=Path, default=None, help="repository root (default: git toplevel of cwd)")
    p.add_argument("--repo", default=None, help="owner/name (default: parsed from `git remote get-url origin`)")
    p.add_argument("--owners", nargs="+", default=(),
                   help='CODEOWNERS owners, e.g. "@my-org/agent-owners" (space/comma separated)')
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--keep-example", action="store_true",
                       help="keep the example agent's reviewed sleep tasks and held-out look ledger (default)")
    scope.add_argument("--reset-state", action="store_true",
                       help="also empty the example's reviewed sleep tasks and the held-out look ledger")
    p.add_argument("--template-commit", default=None,
                   help="template commit the copy was made from (recorded in the marker; default unknown)")
    p.add_argument("--force", action="store_true",
                   help="run even in the template repository / with the template owner as owner")
    p.add_argument("--apply", action="store_true", help="write the changes (default: dry run, print the plan)")
    p.add_argument("--format", choices=("text", "json"), default="text")


def format_plan(plan: Plan, *, applied: bool) -> str:
    lines = [f"ci-lab template init: {plan.repo} (from template {plan.template_repository})",
             f"owners: {' '.join(plan.owners)}"]
    if not plan.actions:
        lines.append("nothing to do: already initialized")
    for a in plan.actions:
        extra = f" ({a.files} files)" if a.op == "delete" and a.path.endswith("/") else ""
        lines.append(f"  {a.op:<6}  {a.path:<36}  {a.reason}{extra}")
    lines += [f"note: {n}" for n in plan.notes]
    if plan.actions:
        lines.append("applied." if applied else "dry run: nothing written; re-run with --apply to write.")
    if applied or not plan.actions:
        lines.append("next: `ci-lab template doctor`, then commit and open a PR (docs/template.md).")
    return "\n".join(lines)


def _toplevel() -> Path:
    top = _git(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(top) if top else Path.cwd()


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else _toplevel()
    commit = args.template_commit
    if commit is not None and not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        print("template init: --template-commit must be a hex commit SHA", file=sys.stderr)
        return 2
    opts = Options(repo=args.repo, owners=tuple(args.owners or ()), reset_state=args.reset_state,
                   force=args.force, template_commit=commit)
    try:
        plan = build_plan(root, opts)
        if args.apply:
            apply_plan(root, plan)
    except InitError as e:
        print(f"template init: {'REFUSED: ' if isinstance(e, Refused) else ''}{e}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps({**plan.to_json(), "applied": bool(args.apply)}, indent=2))
    else:
        print(format_plan(plan, applied=bool(args.apply)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="template_init", description=__doc__.split("\n\n")[0])
    add_arguments(p)
    return run(p.parse_args(argv))
