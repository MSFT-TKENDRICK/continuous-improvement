#!/usr/bin/env python3
"""Publish deferred campaign rounds (``campaign-scheduled.yml`` publish job). STDLIB ONLY.

Runs in the privileged job with system Python (``python3 -I -B``): no ``uv sync``, no third-party
packages, no caches. It imports only the stdlib-only modules ``ci_lab.contracts``,
``ci_lab.publish.github`` and ``ci_lab.publish.deferred`` from the trusted checkout
(``github.sha``), the same code the ``ci-lab campaign publish`` CLI uses. It trusts nothing
produced by the unprivileged, model-driven ``round`` job:

1. requests: at most ``MAX_REQUESTS``; eids belong to the campaign; winner is ``None`` or one of
   the heads; heads are full SHAs; title/body pass the publisher's text validation;
2. every head is a commit whose diff from ``merge-base(<base-ref>, head)`` touches only the
   allowlisted harness prefixes (arms edit the harness surface, nothing else);
3. the ledger artifact is regular files only (no symlinks, safe names, size/count limits) and its
   ``campaign.json`` names this campaign. It replaces ``experiments/campaigns/<cid>``, but
   ``stack.json`` / ``published.jsonl`` are restored from the trusted ledger branch: only this job
   writes them, so a forged stack cannot redirect PR bases or stack edits;
4. :func:`ci_lab.publish.deferred.replay_deferred` pushes winners to ``exp/<eid>/<arm>``, archives
   losers as tags and opens DRAFT stacked PRs. It NEVER merges; the workflow then commits the ledger
   to ``exp-ledger/<cid>``.

Exit 0 on success, 1 (``REFUSED: ...``) on any validation failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Stdlib-only modules from the trusted checkout (pinned by the `-I -S` test).
from ci_lab.contracts import CAMPAIGN_RE
from ci_lab.publish.deferred import PUBLISHED, read_requests, replay_deferred
from ci_lab.publish.github import (
    MAX_BODY,
    MAX_TITLE,
    GitHubPublisher,
    PublishError,
    Runner,
    subprocess_runner,
    validate_repo,
    validate_sha,
    validate_text,
)

DEFAULT_ALLOW = ("src/order_support/harness/",)
TRUSTED = ("stack.json", PUBLISHED)
MAX_REQUESTS = 20
MAX_ARM_COMMITS = 200
MAX_LEDGER_FILES = 5000
MAX_LEDGER_BYTES = 64 * 1024 * 1024
SAFE_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")


class MemOutbox:
    """Per-process outbox: each publish job starts fresh and the publisher's guarded ops
    reconcile against GitHub (existing branch/PR/stack), so reruns are idempotent."""

    def __init__(self) -> None:
        self.done: dict[str, Any] = {}

    def run_once(self, op: str, fn: Callable[[], Any], *, reconcile: Callable[[], Any | None] | None = None) -> Any:
        if op not in self.done:
            self.done[op] = fn()
        return self.done[op]


class DirLedger:
    """Minimal ledger over ``<checkout>/experiments``; the workflow commits the result."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.commits: list[tuple[str, list[str]]] = []

    def _path(self, rel: str) -> Path:
        parts = rel.split("/")
        if not rel or any(not SAFE_PART.match(p) for p in parts):
            raise PublishError(f"bad ledger path {rel!r}")
        return self.root.joinpath(*parts)

    def read_json(self, rel: str) -> Any | None:
        path = self._path(rel)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_json(self, rel: str, obj: Any) -> None:
        path = self._path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def read_jsonl(self, rel: str) -> list[dict[str, Any]]:
        path = self._path(rel)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append_jsonl(self, rel: str, obj: Mapping[str, Any], *, key: str) -> bool:
        if any(r.get(key) == obj[key] for r in self.read_jsonl(rel)):
            return False
        path = self._path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(obj), sort_keys=True) + "\n")
        return True

    def commit(self, message: str, paths: Sequence[str]) -> None:
        self.commits.append((message, list(paths)))


def _git(checkout: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, check=False,
                          timeout=300)
    if check and proc.returncode != 0:
        raise PublishError(f"git {' '.join(args[:2])} failed: {proc.stderr.strip()[-300:]}")
    return proc


def validate_requests(path: Path, cid: str) -> list[dict[str, Any]]:
    reqs = read_requests(path)
    if len(reqs) > MAX_REQUESTS:
        raise PublishError(f"{len(reqs)} requests > {MAX_REQUESTS}")
    for req in reqs:
        if not req["eid"].startswith(f"{cid}-r"):
            raise PublishError(f"request {req['eid']!r} does not belong to campaign {cid!r}")
        for sha in req["heads"].values():
            validate_sha(sha)
        if req["winner"] is not None and req["winner"] not in req["heads"]:
            raise PublishError(f"{req['eid']}: winner {req['winner']!r} has no head")
        validate_text(req["title"], MAX_TITLE, "title")
        validate_text(req["body"], MAX_BODY, "body")
    return reqs


def _allowed(path: str, allow: Sequence[str]) -> bool:
    return ".." not in path.split("/") and any(path.startswith(a) for a in allow)


def check_heads(checkout: Path, reqs: Sequence[Mapping[str, Any]], base_ref: str, allow: Sequence[str]) -> None:
    """Every commit an arm branch would add on top of ``base_ref`` must be a single-parent commit
    touching only allowlisted paths. Checking each commit (not just the net diff) means the pushed
    ancestry carries no transient out-of-surface edits; requiring a unique merge base and no
    merges closes the criss-cross trick where a diff against one merge base hides a revert that a
    recursive merge (virtual base) would apply."""
    for req in reqs:
        for arm, sha in sorted(req["heads"].items()):
            where = f"{req['eid']}/{arm}"
            if _git(checkout, "cat-file", "-t", sha, check=False).stdout.strip() != "commit":
                raise PublishError(f"{where}: {sha} is not a commit in the checkout")
            bases = _git(checkout, "merge-base", "--all", base_ref, sha, check=False).stdout.split()
            if len(bases) != 1:
                raise PublishError(f"{where}: needs exactly one merge base with {base_ref}, found {len(bases)}")
            span = f"{bases[0]}..{sha}"
            if _git(checkout, "rev-list", "--min-parents=2", span).stdout.strip():
                raise PublishError(f"{where}: merge commits are not allowed in arm history")
            commits = _git(checkout, "rev-list", span).stdout.split()
            if len(commits) > MAX_ARM_COMMITS:
                raise PublishError(f"{where}: {len(commits)} commits > {MAX_ARM_COMMITS}")
            for c in commits:
                names = [p for p in _git(checkout, "diff-tree", "--no-commit-id", "--name-only", "-r", "-z",
                                         "--no-renames", c).stdout.split("\x00") if p]
                bad = [p for p in names if not _allowed(p, allow)]
                if bad:
                    raise PublishError(f"{where}: commit {c[:12]} edits outside {list(allow)}: {bad[:5]}")


def import_ledger(src: Path, dest: Path, cid: str) -> None:
    if src.is_symlink() or not src.is_dir():
        raise PublishError(f"ledger artifact {src} is not a directory")
    files: list[tuple[Path, Path]] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        here = Path(dirpath)
        for name in [*dirnames, *filenames]:
            p = here / name
            st = os.lstat(p)
            if not SAFE_PART.match(name) or stat.S_ISLNK(st.st_mode):
                raise PublishError(f"unsafe ledger entry {p.relative_to(src).as_posix()!r}")
            if name in filenames:
                if not stat.S_ISREG(st.st_mode):
                    raise PublishError(f"not a regular file: {p.relative_to(src).as_posix()!r}")
                total += st.st_size
                files.append((p, p.relative_to(src)))
    if len(files) > MAX_LEDGER_FILES or total > MAX_LEDGER_BYTES:
        raise PublishError(f"ledger artifact too large ({len(files)} files, {total} bytes)")
    meta = json.loads((src / "campaign.json").read_text(encoding="utf-8")) if (src / "campaign.json").is_file() \
        else None
    if not isinstance(meta, dict) or meta.get("campaignId") != cid:
        raise PublishError(f"ledger artifact is not campaign {cid!r}")
    if dest.exists():
        shutil.rmtree(dest)
    for p, rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)


def restore_trusted(checkout: Path, dest: Path, trusted_ref: str | None) -> str:
    """Restore ``stack.json`` / ``published.jsonl`` from the ledger branch or, when it does not exist
    (never published, or merged and deleted), from the trusted checkout commit (``HEAD``). Never
    from the artifact. Returns the ref used."""
    rel_dir = dest.relative_to(checkout).as_posix()
    have_ref = bool(trusted_ref) and _git(checkout, "rev-parse", "-q", "--verify", f"{trusted_ref}^{{commit}}",
                                          check=False).returncode == 0
    source = trusted_ref if have_ref and trusted_ref else "HEAD"
    for name in TRUSTED:
        target = dest / name
        if target.exists():
            target.unlink()
        proc = _git(checkout, "show", f"{source}:{rel_dir}/{name}", check=False)
        if proc.returncode == 0:
            target.write_text(proc.stdout, encoding="utf-8")
    return source


def publish(*, cid: str, repo: str, requests: Path, ledger_src: Path, checkout: Path, base: str, base_ref: str,
            trusted_ref: str | None, allow: Sequence[str], runner: Runner | None = None,
            dry_run: bool = False) -> list[dict[str, Any]]:
    if not CAMPAIGN_RE.match(cid):
        raise PublishError(f"bad campaign id {cid!r}")
    validate_repo(repo)
    checkout = checkout.resolve()
    reqs = validate_requests(requests, cid)
    check_heads(checkout, reqs, base_ref, allow)
    dest = checkout / "experiments" / "campaigns" / cid
    import_ledger(ledger_src, dest, cid)
    restore_trusted(checkout, dest, trusted_ref)
    if runner is None and not dry_run:
        subprocess.run(["gh", "auth", "setup-git"], cwd=checkout, check=True, timeout=60)
    publisher = GitHubPublisher(repo, outbox=MemOutbox(), runner=runner or subprocess_runner(checkout),
                                dry_run=dry_run, base=base)
    return replay_deferred(requests, cid=cid, ledger=DirLedger(checkout / "experiments"), publisher=publisher)


def main(argv: Sequence[str] | None = None, *, runner: Runner | None = None) -> int:
    ap = argparse.ArgumentParser(description="Publish deferred campaign rounds (stdlib only).")
    ap.add_argument("--cid", required=True)
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True, help="ledger artifact dir (campaign.json, ...)")
    ap.add_argument("--checkout", type=Path, default=Path("."))
    ap.add_argument("--base", default="main", help="base branch for the bottom PR")
    ap.add_argument("--base-ref", default="origin/main", help="ref arm diffs are measured against")
    ap.add_argument("--trusted-ref", default=None, help="ledger branch ref (default origin/exp-ledger/<cid>)")
    ap.add_argument("--allow-prefix", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        out = publish(cid=args.cid, repo=args.repo, requests=args.requests, ledger_src=args.ledger,
                      checkout=args.checkout, base=args.base, base_ref=args.base_ref,
                      trusted_ref=args.trusted_ref or f"origin/exp-ledger/{args.cid}",
                      allow=tuple(args.allow_prefix or DEFAULT_ALLOW), runner=runner, dry_run=args.dry_run)
    except (PublishError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"published": out}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
