"""Sleep bundle: the ONLY thing that crosses from the unprivileged ``evaluate`` job to the
``publish`` job (C10). ``out/sleep-bundle/{candidate.patch, experiment.json, results.json,
manifest.json}``; the manifest pins sha256 digests and the base commit SHA so the publisher
can refuse anything tampered with or built against another commit.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLE_FORMAT = "ci_lab.sleep.bundle.v1"
BUNDLE_FILES = ("candidate.patch", "experiment.json", "results.json")
MANIFEST = "manifest.json"
ALLOWED_PREFIXES = ("src/order_support/harness/skills/", "experiments/sleep/")


@dataclass(frozen=True)
class FileChange:
    path: str  # repo-relative POSIX path
    old: str | None  # None = new file
    new: str


def _check_path(path: str) -> None:
    parts = path.split("/")
    if (not path or path.startswith("/") or "\\" in path or ":" in path
            or any(p in ("", ".", "..", ".git") for p in parts)
            or not path.startswith(ALLOWED_PREFIXES)):
        raise ValueError(f"refusing to patch path {path!r}")


def _hunks(old: str, new: str, path: str, *, new_file: bool) -> list[str]:
    out = []
    for line in difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                     fromfile="/dev/null" if new_file else f"a/{path}",
                                     tofile=f"b/{path}", n=3, lineterm="\n"):
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return out


def make_patch(changes: Sequence[FileChange]) -> str:
    """git-style unified diff (``git apply`` compatible); unchanged files are skipped."""
    chunks: list[str] = []
    for ch in sorted(changes, key=lambda c: c.path):
        _check_path(ch.path)
        if ch.old is not None and ch.old == ch.new:
            continue
        head = [f"diff --git a/{ch.path} b/{ch.path}\n"]
        if ch.old is None:
            head.append("new file mode 100644\n")
        body = _hunks(ch.old or "", ch.new, ch.path, new_file=ch.old is None)
        if body:
            chunks.extend(head + body)
    return "".join(chunks)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def write_bundle(out_dir: Path, *, patch: str, experiment: Mapping[str, Any], results: Mapping[str, Any],
                 base_sha: str, night_id: str, date: str, accepted: bool, ledger_update: bool,
                 status: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (*BUNDLE_FILES, MANIFEST):
        (out_dir / stale).unlink(missing_ok=True)
    blobs = {"candidate.patch": patch.encode("utf-8"), "experiment.json": _dump(experiment),
             "results.json": _dump(results)}
    files = {}
    for name, data in blobs.items():
        (out_dir / name).write_bytes(data)
        files[name] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    manifest = {"format": BUNDLE_FORMAT, "base_sha": base_sha, "night_id": night_id, "date": date,
                "status": status, "accepted": bool(accepted), "ledger_update": bool(ledger_update),
                "allowed_prefixes": list(ALLOWED_PREFIXES), "files": files}
    (out_dir / MANIFEST).write_bytes(_dump(manifest))
    return manifest


def verify_bundle(out_dir: Path) -> dict[str, Any]:
    """Recompute digests (tests and ``sleep dry-run``); publish has its own stdlib copy."""
    manifest = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("bad bundle format")
    for name in BUNDLE_FILES:
        meta = manifest["files"][name]
        data = (out_dir / name).read_bytes()
        if sha256_bytes(data) != meta["sha256"] or len(data) != meta["bytes"]:
            raise ValueError(f"digest mismatch for {name}")
    return manifest
