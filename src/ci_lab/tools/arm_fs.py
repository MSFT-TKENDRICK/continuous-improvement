"""``arm_fs``: list/read/write tools scoped to the evolvable surface of ONE worktree (C13).

Tools return strings (``ERROR: ...`` on rejection) so the model sees why a call failed;
:class:`ArmFS` exposes the same operations raising :class:`PathRejected` for Python callers.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ci_lab.tools.paths import (
    PathRejected,
    contained,
    glob_match,
    glob_prefix,
    is_link_or_reparse,
    matches_any,
    normalize_rel,
)

__all__ = ["DATA_SUFFIXES", "DEFAULT_MAX_BYTES", "ArmFS", "make_arm_fs"]

DEFAULT_MAX_BYTES = 64 * 1024
DATA_SUFFIXES = (".md", ".yaml", ".yml", ".json", ".txt")  # I7: data-only surface
MAX_LIST = 500


@dataclass
class ArmFS:
    worktree: Path
    surface_globs: Sequence[str]
    frozen_globs: Sequence[str] = ()
    max_bytes: int = DEFAULT_MAX_BYTES
    allowed_suffixes: Sequence[str] = DATA_SUFFIXES
    writable_globs: Sequence[str] | None = None  # narrower write scope (e.g. one component)
    written: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.worktree = Path(self.worktree).resolve(strict=True)

    def allowed(self, rel: str) -> bool:
        return matches_any(rel, self.surface_globs) and not matches_any(rel, self.frozen_globs)

    def _check(self, rel: str) -> tuple[str, Path]:
        norm = normalize_rel(rel)
        if not self.allowed(norm):
            raise PathRejected(f"outside the editable surface: {norm!r}")
        return norm, contained(self.worktree, norm)

    def list(self, pattern: str = "**") -> list[str]:
        out: set[str] = set()
        for glob in self.surface_globs:
            prefix = glob_prefix(glob)
            try:
                start = contained(self.worktree, prefix) if prefix else self.worktree
            except PathRejected:
                continue
            if not start.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
                base = Path(dirpath)
                dirnames[:] = sorted(d for d in dirnames if d.lower() != ".git"
                                     and not (base / d).is_symlink() and not is_link_or_reparse(base / d))
                for name in filenames:
                    p = base / name
                    if p.is_symlink() or is_link_or_reparse(p):
                        continue
                    rel = p.relative_to(self.worktree).as_posix()
                    if self.allowed(rel) and glob_match(rel, pattern):
                        out.add(rel)
        return sorted(out)[:MAX_LIST]

    def read(self, rel: str) -> str:
        norm, path = self._check(rel)
        if not path.is_file():
            raise PathRejected(f"no such file: {norm!r}")
        data = path.read_bytes()
        text = data[: self.max_bytes].decode("utf-8", errors="replace")
        if len(data) > self.max_bytes:
            text += f"\n[truncated at {self.max_bytes} bytes of {len(data)}]"
        return text

    def write(self, rel: str, content: str) -> int:
        norm, path = self._check(rel)
        if self.writable_globs is not None and not matches_any(norm, self.writable_globs):
            raise PathRejected(f"not writable in this arm (outside its component): {norm!r}")
        if not norm.lower().endswith(tuple(s.lower() for s in self.allowed_suffixes)):
            raise PathRejected(f"only data files {tuple(self.allowed_suffixes)} may be written: {norm!r}")
        data = content.encode("utf-8")
        if len(data) > self.max_bytes:
            raise PathRejected(f"content is {len(data)} bytes; limit is {self.max_bytes}")
        if path.exists() and not path.is_file():
            raise PathRejected(f"not a regular file: {norm!r}")
        _mkdirs(self.worktree, norm)
        fd, tmp = tempfile.mkstemp(prefix=".arm_fs-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            contained(self.worktree, norm)  # re-check after mkdirs (TOCTOU narrowing)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        if norm not in self.written:
            self.written.append(norm)
        return len(data)


def _mkdirs(root: Path, norm: str) -> None:
    cur = root
    parts = norm.split("/")[:-1]
    for i, part in enumerate(parts):
        cur = cur / part
        if not cur.exists():
            contained(root, "/".join(parts[: i + 1]))
            cur.mkdir()
        elif not cur.is_dir():
            raise PathRejected(f"not a directory: {'/'.join(parts[: i + 1])!r}")


def make_arm_fs(worktree: Path | str, surface_globs: Sequence[str], frozen_globs: Sequence[str] = (),
                max_bytes: int = DEFAULT_MAX_BYTES, *,
                writable_globs: Sequence[str] | None = None) -> dict[str, Callable[..., str]]:
    """Bind ``list_files`` / ``read_file`` / ``write_file`` to one arm worktree.

    Returns a name -> function map usable as declarative tool ``bindings``. The backing
    :class:`ArmFS` is available as ``make_arm_fs(...)["write_file"].arm_fs``. Reads cover the
    whole surface; ``writable_globs`` (if given) narrows writes further, e.g. to one component.
    """
    fs = ArmFS(Path(worktree), tuple(surface_globs), tuple(frozen_globs), max_bytes,
               writable_globs=tuple(writable_globs) if writable_globs is not None else None)

    def list_files(pattern: str = "**") -> str:
        """List editable harness files (repo-relative posix paths), optionally filtered by a glob."""
        files = fs.list(pattern)
        return "\n".join(files) if files else "(no matching files)"

    def read_file(path: str) -> str:
        """Read one editable harness file by its repo-relative path (as shown by list_files)."""
        try:
            return fs.read(path)
        except (PathRejected, OSError) as exc:
            return f"ERROR: {exc}"

    def write_file(path: str, content: str) -> str:
        """Create or overwrite one editable harness data file (Markdown/YAML) with the full new content."""
        try:
            n = fs.write(path, content)
        except (PathRejected, OSError) as exc:
            return f"ERROR: {exc}"
        return f"wrote {n} bytes to {normalize_rel(path)}"

    for fn in (list_files, read_file, write_file):
        fn.arm_fs = fs  # type: ignore[attr-defined]
    return {"list_files": list_files, "read_file": read_file, "write_file": write_file}
