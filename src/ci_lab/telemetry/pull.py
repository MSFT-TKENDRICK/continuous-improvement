"""``ci-lab telemetry pull --run <id>`` (design §12.5 D5): fetch the ``spans`` artifact of a
GitHub Actions run via ``gh``, verify its digest and span schema version, and cache it under
``~/.ci-lab/imports/<run_id>/`` (``manifest.json`` + ``*.jsonl``) for import / the canvas."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ci_lab.contracts import SPAN_SCHEMA_VERSION
from ci_lab.telemetry.aspire import safe_extract
from ci_lab.telemetry.jsonl import read_jsonl

IMPORTS_ENV = "CI_IMPORTS_DIR"
DEFAULT_ARTIFACT = "spans"
MANIFEST = "manifest.json"


class PullError(RuntimeError):
    pass


def imports_dir() -> Path:
    return Path(os.environ.get(IMPORTS_ENV) or Path.home() / ".ci-lab" / "imports")


def _gh(args: list[str]) -> bytes:  # patched in tests
    try:
        r = subprocess.run(["gh", *args], check=False, capture_output=True, timeout=600)
    except FileNotFoundError as exc:
        raise PullError("GitHub CLI 'gh' not found on PATH") from exc
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise PullError(f"gh {args[0]} failed: {msg[-1] if msg else r.returncode}")
    return r.stdout


def _find_artifact(run_id: str, repo: str, name: str) -> dict[str, Any]:
    meta = json.loads(_gh(["api", f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"]))
    for art in meta.get("artifacts", []):
        if art.get("name") == name and not art.get("expired"):
            return art
    raise PullError(f"run {run_id} has no unexpired artifact named {name!r}")


def pull(run_id: str | int, *, repo: str | None = None, artifact: str = DEFAULT_ARTIFACT,
         allow_no_digest: bool = False, force: bool = False,
         cache: Path | None = None) -> dict[str, Any]:
    """Download, verify and cache a run's span artifact; returns its manifest."""
    run_id = str(int(run_id))  # numeric only: it becomes a path component
    repo_ref = repo or "{owner}/{repo}"  # gh resolves placeholders from the cwd's git remote
    dest = (cache or imports_dir()) / run_id
    art = _find_artifact(run_id, repo_ref, artifact)
    digest = str(art.get("digest") or "")
    want = digest.split(":", 1)[1].lower() if digest.startswith("sha256:") else None
    old = dest / MANIFEST
    if old.is_file() and not force:
        m = json.loads(old.read_text(encoding="utf-8"))
        if m.get("artifact_id") == art.get("id") and m.get("schemaVersion") == SPAN_SCHEMA_VERSION:
            return {**m, "cached": True}
    if want is None and not allow_no_digest:
        raise PullError(f"artifact {art.get('id')} has no sha256 digest; pass --allow-no-digest to trust it")
    data = _gh(["api", f"repos/{repo_ref}/actions/artifacts/{art['id']}/zip"])
    sha = hashlib.sha256(data).hexdigest()
    if want is not None and sha != want:
        raise PullError(f"artifact digest mismatch: got sha256:{sha}, expected {digest}")
    root = dest.parent
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{run_id}.{secrets.token_hex(4)}"
    try:
        tmp.mkdir()
        (tmp / "artifact.zip").write_bytes(data)
        safe_extract(tmp / "artifact.zip", tmp / "files")
        (tmp / "artifact.zip").unlink()
        files = sorted((tmp / "files").rglob("*.jsonl"))
        records, bad = read_jsonl(files)  # raises SchemaError on unknown schemaVersion
        manifest = {
            "run_id": run_id, "repo": repo, "artifact": artifact, "artifact_id": art.get("id"),
            "digest": digest or None, "sha256": sha, "schemaVersion": SPAN_SCHEMA_VERSION,
            "files": [f.relative_to(tmp).as_posix() for f in files], "spans": len(records),
            "skipped_lines": bad,
            "traces": len({r["traceId"] for r in records}),
            "services": sorted({str(r["resource"].get("service.name", "")) for r in records} - {""}),
            "artifact_created": art.get("created_at"),
            "pulled": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return {**manifest, "cached": False}


def files_of(manifest: dict[str, Any], cache: Path | None = None) -> list[Path]:
    base = (cache or imports_dir()) / manifest["run_id"]
    return [base / f for f in manifest.get("files", [])]


def list_imports(cache: Path | None = None) -> list[dict[str, Any]]:
    root = cache or imports_dir()
    out = []
    for m in sorted(root.glob(f"*/{MANIFEST}")) if root.is_dir() else []:
        try:
            out.append(json.loads(m.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out