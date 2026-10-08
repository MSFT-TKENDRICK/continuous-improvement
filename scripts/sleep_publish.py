#!/usr/bin/env python3
"""Publish a sleep/usage bundle as a DRAFT pull request (design C10). STDLIB ONLY.

Runs in the privileged ``publish`` job with system Python: no ``uv sync``, no project imports,
no caches. It trusts nothing in the bundle (built by the unprivileged ``evaluate`` job):

1. bundle = exactly ``candidate.patch``, ``experiment.json``, ``results.json``, ``manifest.json``
   (regular files, size limits); sha256 digests and sizes match the manifest;
2. ``base_sha`` equals the checked-out commit;
3. the patch is plain text edits/new files only (no mode/rename/copy/delete/binary) and every
   path is allowlisted: ``kind: sleep`` -> ``src/order_support/harness/skills/**`` (accepted
   candidates only) + ``experiments/sleep/**``; ``kind: usage`` -> only
   ``experiments/sleep/*.pending.jsonl``; ``git apply --numstat`` must agree with the parse;
4. ``git apply --check`` then ``git apply --index`` on branch
   ``exp/sleep-<yyyymmdd>-<run_attempt>/cand`` (or ``exp/usage-<yyyymmdd>/tasks``), commit with
   trailers, push, ``gh pr create --draft``. It NEVER merges.

Exit 0 on success / nothing to publish, 1 (``REFUSED: ...``) on any validation failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

BUNDLE_FORMAT = "ci_lab.sleep.bundle.v1"
FILES = ("candidate.patch", "experiment.json", "results.json")
MANIFEST = "manifest.json"
MAX_PATCH = 200 * 1024
MAX_JSON = 1024 * 1024
MAX_FILES_IN_PATCH = 32
SKILLS_PREFIX = "src/order_support/harness/skills/"
LEDGER_PREFIX = "experiments/sleep/"
PENDING_RE = re.compile(r"experiments/sleep/[a-z0-9][a-z0-9-]{0,63}\.pending\.jsonl")
PATH_RE = re.compile(r"[A-Za-z0-9._/-]{1,200}")
SHA_RE = re.compile(r"[0-9a-f]{40}")
STATUSES = {"sleep": {"accepted", "rejected", "no_tasks", "budget_exceeded", "error"},
            "usage": {"pending_review"}}
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898283+github-actions[bot]@users.noreply.github.com"

Run = Callable[..., subprocess.CompletedProcess]


class PublishError(Exception):
    pass


def _run(args: Sequence[str], **kw: Any) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), check=True, text=True, capture_output=True, **kw)


# ------------------------------------------------------------------ bundle

def _read_regular(path: Path, limit: int) -> bytes:
    try:
        st = path.lstat()
    except OSError as exc:
        raise PublishError(f"missing bundle file {path.name}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise PublishError(f"{path.name} is not a regular file")
    if st.st_size > limit:
        raise PublishError(f"{path.name} exceeds {limit} bytes")
    return path.read_bytes()


def _json_obj(data: bytes, name: str) -> dict[str, Any]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PublishError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(obj, dict):
        raise PublishError(f"{name} must be a JSON object")
    return obj


def load_bundle(bundle: Path) -> dict[str, Any]:
    if not bundle.is_dir() or bundle.is_symlink():
        raise PublishError(f"bundle directory not found: {bundle}")
    names = sorted(p.name for p in bundle.iterdir())
    if names != sorted((*FILES, MANIFEST)):
        raise PublishError(f"bundle must contain exactly {sorted((*FILES, MANIFEST))}, found {names}")
    manifest_bytes = _read_regular(bundle / MANIFEST, MAX_JSON)
    manifest = _json_obj(manifest_bytes, MANIFEST)
    if manifest.get("format") != BUNDLE_FORMAT:
        raise PublishError("unknown bundle format")
    kind = manifest.get("kind", "sleep")
    if kind not in STATUSES:
        raise PublishError(f"unknown bundle kind {kind!r}")
    base_sha, date, night_id = manifest.get("base_sha"), manifest.get("date"), manifest.get("night_id")
    if not isinstance(base_sha, str) or not SHA_RE.fullmatch(base_sha):
        raise PublishError("manifest base_sha must be 40 lowercase hex chars")
    if not isinstance(date, str) or not re.fullmatch(r"20\d{6}", date):
        raise PublishError("manifest date must be yyyymmdd")
    if not isinstance(night_id, str) or not re.fullmatch(rf"{kind}-{date}-\d{{1,4}}", night_id):
        raise PublishError("manifest night_id does not match kind/date")
    for flag in ("accepted", "ledger_update"):
        if not isinstance(manifest.get(flag), bool):
            raise PublishError(f"manifest {flag} must be a bool")
    if manifest.get("status") not in STATUSES[kind]:
        raise PublishError(f"bad status {manifest.get('status')!r} for kind {kind}")
    if manifest["accepted"] != (manifest["status"] == "accepted"):
        raise PublishError("accepted flag inconsistent with status")
    files = manifest.get("files")
    if not isinstance(files, dict) or sorted(files) != sorted(FILES):
        raise PublishError("manifest files must list exactly the bundle files")
    blobs: dict[str, bytes] = {}
    for name in FILES:
        data = _read_regular(bundle / name, MAX_PATCH if name.endswith(".patch") else MAX_JSON)
        meta = files[name]
        if (not isinstance(meta, dict) or meta.get("sha256") != hashlib.sha256(data).hexdigest()
                or meta.get("bytes") != len(data)):
            raise PublishError(f"digest mismatch for {name}")
        blobs[name] = data
    results = _json_obj(blobs["results.json"], "results.json")
    experiment = _json_obj(blobs["experiment.json"], "experiment.json")
    if results.get("status", manifest["status"]) != manifest["status"]:
        raise PublishError("results status disagrees with manifest")
    try:
        patch = blobs["candidate.patch"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublishError("patch is not UTF-8") from exc
    return {"manifest": manifest, "kind": kind, "patch": patch, "results": results, "experiment": experiment,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


# ------------------------------------------------------------------ patch

def _check_path(path: str, kind: str, accepted: bool) -> None:
    parts = path.split("/")
    if (not PATH_RE.fullmatch(path) or path.startswith("/")
            or any(p in ("", ".", "..") or p.lower() == ".git" for p in parts)):
        raise PublishError(f"illegal path in patch: {path!r}")
    if kind == "usage":
        if not PENDING_RE.fullmatch(path):
            raise PublishError(f"usage bundle may only touch experiments/sleep/*.pending.jsonl, not {path!r}")
        return
    if PENDING_RE.fullmatch(path):
        raise PublishError(f"sleep bundle must not touch pending tasks: {path!r}")
    if path.startswith(LEDGER_PREFIX):
        return
    if path.startswith(SKILLS_PREFIX):
        if not accepted:
            raise PublishError(f"skill change {path!r} in a bundle that was not accepted")
        return
    raise PublishError(f"path outside the publish allowlist: {path!r}")


_FORBIDDEN_HEADERS = ("old mode", "new mode", "deleted file mode", "rename ", "copy ", "similarity index",
                      "dissimilarity index", "Binary files", "GIT binary patch")


def parse_patch(patch: str, kind: str, accepted: bool) -> list[str]:
    """Strict parse of a ``git diff``-style patch; returns the touched paths."""
    if "\x00" in patch:
        raise PublishError("NUL byte in patch")
    paths: list[str] = []
    lines = patch.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
        if not m or m.group(1) != m.group(2):
            raise PublishError(f"unexpected patch line {i + 1}: {line[:80]!r}")
        path = m.group(1)
        _check_path(path, kind, accepted)
        if path in paths:
            raise PublishError(f"duplicate path in patch: {path}")
        paths.append(path)
        i += 1
        new_file = False
        while i < n and not lines[i].startswith("--- "):
            hdr = lines[i]
            if hdr == "new file mode 100644":
                new_file = True
            elif re.fullmatch(r"index [0-9a-f]{7,64}\.\.[0-9a-f]{7,64}( 100644)?", hdr):
                pass
            elif hdr.startswith(_FORBIDDEN_HEADERS):
                raise PublishError(f"forbidden patch header for {path}: {hdr[:40]!r}")
            else:
                raise PublishError(f"unexpected patch header for {path}: {hdr[:40]!r}")
            i += 1
        if i + 1 >= n:
            raise PublishError(f"truncated patch for {path}")
        old_hdr, new_hdr = lines[i], lines[i + 1]
        if old_hdr != ("--- /dev/null" if new_file else f"--- a/{path}") or new_hdr != f"+++ b/{path}":
            raise PublishError(f"---/+++ headers do not match {path}")
        i += 2
        hunks = 0
        while i < n and not lines[i].startswith("diff --git "):
            body = lines[i]
            if body.startswith("@@ "):
                if not re.match(r"@@ -\d+(,\d+)? \+\d+(,\d+)? @@", body):
                    raise PublishError(f"bad hunk header for {path}")
                hunks += 1
            elif hunks and (body[:1] in (" ", "+", "-") or body == "\\ No newline at end of file"):
                pass
            else:
                raise PublishError(f"unexpected line in hunk for {path}: {body[:40]!r}")
            if "\r" in body.rstrip("\r"):
                raise PublishError(f"carriage return inside a line for {path}")
            i += 1
        if not hunks:
            raise PublishError(f"no hunks for {path}")
    if len(paths) > MAX_FILES_IN_PATCH:
        raise PublishError("patch touches too many files")
    return paths


# ------------------------------------------------------------------ PR text (built here, never copied verbatim)

def _clean(value: Any, limit: int = 300) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))
    s = s.replace("@", "@\u200b").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("`", "'")
    return s[:limit]


def pr_title(b: dict[str, Any]) -> str:
    m = b["manifest"]
    if b["kind"] == "usage":
        return f"[usage] pending sleep tasks {m['date']} (human review required)"
    what = "candidate skill accepted" if m["accepted"] else f"ledger update ({m['status']})"
    return f"[sleep] {m['night_id']}: {what}"


LESSONS_PREFIX = "experiments/sleep/lessons/"


def pr_body(b: dict[str, Any], paths: Sequence[str]) -> str:
    m, res, exp = b["manifest"], b["results"], b["experiment"]
    out: list[str] = []
    if b["kind"] == "usage":
        out += [f"Usage-derived **pending** SkillOpt-Sleep tasks for {m['date']} (design §11.3, C22).", "",
                "Rows are `reviewed: false` and are ignored by SkillOpt-Sleep until a human checks the redacted",
                "intent, authors `reference` + `judge`, sets `reviewed: true` and moves the row into the",
                "reviewed tasks file.", "", "| target | new pending |", "|---|---|"]
        targets = exp.get("targets") if isinstance(exp.get("targets"), dict) else {}
        for name, t in sorted(targets.items())[:20]:
            new = t.get("new") if isinstance(t, dict) and isinstance(t.get("new"), list) else []
            out.append(f"| {_clean(name, 48)} | {len(new)} |")
    else:
        out += [f"SkillOpt-Sleep night **{m['night_id']}**: status **{m['status']}**.", "",
                "| target | status | Δ lower bound | δ | critical (incumbent→candidate) |", "|---|---|---|---|---|"]
        targets = res.get("targets") if isinstance(res.get("targets"), dict) else {}
        for name, t in sorted(targets.items())[:20]:
            t = t if isinstance(t, dict) else {}
            g = t.get("gate") if isinstance(t.get("gate"), dict) else {}
            safety = f"{_clean(g.get('critical_incumbent', '-'), 8)}→{_clean(g.get('critical_candidate', '-'), 8)}"
            out.append(f"| {_clean(name, 48)} | {_clean(t.get('status', '?'), 20)} | "
                       f"{_clean(g.get('delta_lcb', '-'), 12)} | {_clean(g.get('delta', '-'), 12)} | {safety} |")
        reasons = [r for r in (res.get("reasons") or []) if isinstance(r, str)][:10]
        if reasons:
            out += ["", "**Reasons**", *[f"- {_clean(r)}" for r in reasons]]
    if any(p.startswith(LESSONS_PREFIX) for p in paths):
        out += ["", "**Lesson candidates** (HOOK(M16)): proposals only. They are never adopted or enforced",
                "automatically; confirm with `ci-lab lessons confirm` and land rules in a separate reviewed PR."]
    out += ["", "**Files**", *[f"- `{p}`" for p in paths], "",
            f"Base: `{m['base_sha']}` · bundle manifest sha256: `{b['manifest_sha256']}`", "",
            "_Draft opened by `scripts/sleep_publish.py`; a human must review and merge (C10)._"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ publish

def branch_name(b: dict[str, Any], attempt: int) -> str:
    m = b["manifest"]
    return f"exp/usage-{m['date']}/tasks" if b["kind"] == "usage" else f"exp/sleep-{m['date']}-{attempt}/cand"


def publish(repo: Path, bundle_dir: Path, *, attempt: int, base_ref: str, dry_run: bool = False,
            run: Run = _run) -> dict[str, Any]:
    b = load_bundle(bundle_dir)
    m = b["manifest"]
    git = ["git", "-C", str(repo)]
    head = run([*git, "rev-parse", "HEAD"]).stdout.strip()
    if head != m["base_sha"]:
        raise PublishError(f"base sha mismatch: bundle {m['base_sha']} != checkout {head}")
    if not b["patch"].strip():
        if m["ledger_update"] or m["accepted"]:
            raise PublishError("manifest claims changes but the patch is empty")
        return {"published": False, "reason": "nothing to publish"}
    if not m["ledger_update"]:
        raise PublishError("non-empty patch but ledger_update is false")
    paths = parse_patch(b["patch"], b["kind"], m["accepted"])
    if run([*git, "status", "--porcelain", "--untracked-files=no"]).stdout.strip():
        raise PublishError("checkout has local modifications")
    patch_file = str(bundle_dir / "candidate.patch")
    numstat = run([*git, "apply", "--numstat", "-z", patch_file]).stdout
    applied = sorted(rec.split("\t", 2)[2] for rec in numstat.split("\x00") if rec.count("\t") >= 2)
    if applied != sorted(paths):
        raise PublishError(f"git apply sees different paths: {applied} vs {sorted(paths)}")
    run([*git, "apply", "--check", patch_file])
    branch = branch_name(b, attempt)
    if dry_run:
        return {"published": False, "dry_run": True, "branch": branch, "paths": paths}
    run([*git, "checkout", "-q", "-b", branch])
    run([*git, "apply", "--index", patch_file])
    staged = sorted(p for p in run([*git, "diff", "--cached", "--name-only", "-z"]).stdout.split("\x00") if p)
    if staged != sorted(paths):
        raise PublishError(f"staged files differ from the patch: {staged}")
    trailer = "Usage" if b["kind"] == "usage" else "Sleep"
    msg = (f"{pr_title(b)}\n\n"
           f"{trailer}-Night-Id: {m['night_id']}\n{trailer}-Base: {m['base_sha']}\n"
           f"{trailer}-Bundle-Manifest: sha256:{b['manifest_sha256']}\n{trailer}-Decision: {m['status']}\n")
    run([*git, "-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}", "commit", "-q", "--no-verify",
         "-m", msg])
    run(["gh", "auth", "setup-git"], cwd=str(repo))
    run([*git, "push", "origin", f"HEAD:refs/heads/{branch}"])
    existing = run(["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "url", "--jq", ".[0].url"],
                   cwd=str(repo)).stdout.strip()
    if existing:
        return {"published": True, "branch": branch, "url": existing, "pr": "existing"}
    created = run(["gh", "pr", "create", "--draft", "--base", base_ref, "--head", branch, "--title", pr_title(b),
                   "--body-file", "-"], cwd=str(repo), input=pr_body(b, paths)).stdout.strip()
    return {"published": True, "branch": branch, "url": created, "pr": "created"}


def main(argv: Sequence[str] | None = None, *, run: Run = _run) -> int:
    ap = argparse.ArgumentParser(description="Publish a sleep/usage bundle as a draft PR (stdlib only).")
    ap.add_argument("--bundle", type=Path, default=Path("out/sleep-bundle"))
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true", help="validate + git apply --check only")
    args = ap.parse_args(argv)
    raw_attempt = os.environ.get("SLEEP_RUN_ATTEMPT") or os.environ.get("GITHUB_RUN_ATTEMPT") or "1"
    base_ref = os.environ.get("SLEEP_BASE_REF") or os.environ.get("GITHUB_REF_NAME") or "main"
    try:
        if not re.fullmatch(r"[1-9]\d{0,3}", raw_attempt):
            raise PublishError(f"bad run attempt {raw_attempt!r}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}", base_ref) or ".." in base_ref:
            raise PublishError(f"bad base ref {base_ref!r}")
        out = publish(args.repo.resolve(), args.bundle.resolve(), attempt=int(raw_attempt), base_ref=base_ref,
                      dry_run=args.dry_run, run=run)
    except PublishError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"REFUSED: {' '.join(map(str, exc.cmd[:4]))} failed: {(exc.stderr or '').strip()[:500]}",
              file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
