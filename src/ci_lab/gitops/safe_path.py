"""Filesystem containment for agent-driven edits (C13).

``safe_join(root, rel)`` returns ``root/rel`` only if ``rel`` is a plain relative path that
stays inside ``root`` on every platform:

* rejects absolute / UNC / drive-relative paths, ``..``, NUL and control characters,
  ``:`` (drive letters and NTFS alternate data streams), Windows-invalid characters,
  reserved device names (``CON``, ``NUL``, ``COM1``...), trailing dots/spaces and 8.3
  short-name aliases (``GIT~1``);
* rejects any ``.git`` component (case-insensitively);
* walks every existing component with ``lstat`` and rejects symlinks, junctions and any
  other reparse point (``FILE_ATTRIBUTE_REPARSE_POINT``), plus case aliases of existing
  entries (``SRC/x`` when the directory is ``src``) so case-sensitive glob policies cannot
  be bypassed on case-insensitive filesystems;
* finally requires the resolved path to stay inside the resolved root.

Backslashes are treated as separators everywhere. ``match_globs`` matches POSIX-style
relative paths (``*``/``?`` stay within a segment, ``**`` spans segments).
"""

from __future__ import annotations

import functools
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path

from ci_lab.ledger.atomic import atomic_write_bytes

FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
             *(f"COM{i}" for i in "0123456789¹²³"), *(f"LPT{i}" for i in "0123456789¹²³")}
_INVALID_CHARS = set('<>:"|?*') | {chr(i) for i in range(32)}
_SHORT_NAME = re.compile(r"^[^.~]{1,6}~\d+(\.[^.]{0,3})?$")


class UnsafePathError(ValueError):
    pass


def _components(rel: str | os.PathLike[str]) -> list[str]:
    s = os.fspath(rel)
    if not isinstance(s, str):
        raise UnsafePathError("path must be text")
    if not s or not s.strip():
        raise UnsafePathError("empty path")
    if "\x00" in s:
        raise UnsafePathError("NUL in path")
    s = s.replace("\\", "/")
    if s.startswith("/"):
        raise UnsafePathError(f"absolute path {rel!r}")
    if ":" in s:
        raise UnsafePathError(f"drive or alternate data stream in {rel!r}")
    parts = [p for p in s.split("/") if p not in ("", ".")]
    if not parts:
        raise UnsafePathError(f"path {rel!r} names the root")
    for part in parts:
        _check_component(part, rel)
    return parts


def _check_component(part: str, rel: object) -> None:
    if part == "..":
        raise UnsafePathError(f"'..' in {rel!r}")
    if bad := (_INVALID_CHARS & set(part)):
        raise UnsafePathError(f"invalid character(s) {sorted(bad)!r} in {rel!r}")
    if part != part.rstrip(". "):
        raise UnsafePathError(f"trailing dot/space in component {part!r}")
    stem = part.split(".")[0].upper().rstrip()
    if stem in _RESERVED:
        raise UnsafePathError(f"reserved device name {part!r}")
    if part.casefold() == ".git" or part.casefold().startswith(".git."):
        raise UnsafePathError(f"'.git' component in {rel!r}")
    if "~" in part and _SHORT_NAME.match(part):
        raise UnsafePathError(f"8.3 short-name alias {part!r}")


def is_link_like(path: str | os.PathLike[str]) -> bool:
    """True for symlinks, junctions and other reparse points (does not follow links)."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    if getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _inside(child: Path, root: Path) -> bool:
    c, r = os.path.normcase(str(child)), os.path.normcase(str(root))
    return c == r or c.startswith(r.rstrip("\\/") + os.sep)


def safe_join(root: str | os.PathLike[str], rel: str | os.PathLike[str]) -> Path:
    """Contained absolute path for ``rel`` under ``root`` (see module doc). The target need
    not exist; existing components are verified. Raises :class:`UnsafePathError`."""
    parts = _components(rel)
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise UnsafePathError(f"root {root!r} is not a directory")
    cur = base
    exists = True
    for part in parts:
        if exists:
            try:
                names = os.listdir(cur)
            except (FileNotFoundError, NotADirectoryError):
                names, exists = [], False
            if exists and part not in names:
                alias = next((n for n in names if n.casefold() == part.casefold()), None)
                if alias is not None:
                    raise UnsafePathError(f"case alias {part!r} of existing {alias!r}")
                exists = False
        cur = cur / part
        if exists and is_link_like(cur):
            raise UnsafePathError(f"symlink/junction/reparse point at {cur.relative_to(base).as_posix()!r}")
    real = Path(os.path.realpath(cur))
    if not _inside(real, base):
        raise UnsafePathError(f"{rel!r} escapes the root")
    return cur


def safe_write_bytes(root: str | os.PathLike[str], rel: str | os.PathLike[str], data: bytes) -> Path:
    """Contained atomic write (temp + rename); parents are created then re-verified."""
    target = safe_join(root, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = safe_join(root, rel)
    if target.exists() and not target.is_file():
        raise UnsafePathError(f"{rel!r} is not a regular file")
    return atomic_write_bytes(target, data)


def safe_write_text(root: str | os.PathLike[str], rel: str | os.PathLike[str], text: str) -> Path:
    return safe_write_bytes(root, rel, text.encode("utf-8"))


# ---------------------------------------------------------------- globs


@functools.lru_cache(maxsize=512)
def _glob_regex(glob: str) -> re.Pattern[str]:
    g = glob.replace("\\", "/").lstrip("/")
    while g.startswith("./"):
        g = g[2:]
    out: list[str] = []
    i, n = 0, len(g)
    while i < n:
        c = g[i]
        if c == "*":
            if g.startswith("**", i):
                at_seg_start = i == 0 or g[i - 1] == "/"
                j = i + 2
                if at_seg_start and j < n and g[j] == "/":
                    out.append("(?:[^/]+/)*")  # '**/' = zero or more directories
                    i = j + 1
                    continue
                out.append(".*")
                i = j
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = g.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = g[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"(?:(?!/)[{body}])")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def normalize_rel(rel: str | os.PathLike[str]) -> str:
    return "/".join(p for p in os.fspath(rel).replace("\\", "/").split("/") if p not in ("", "."))


def match_globs(rel: str | os.PathLike[str], globs: Iterable[str]) -> bool:
    """True if POSIX-style relative ``rel`` matches any glob (case-sensitive)."""
    path = normalize_rel(rel)
    return any(_glob_regex(g).match(path) for g in globs)
