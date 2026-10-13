"""Path containment (C13) and glob matching shared by the meta-agent tools.

``safe_join`` is a local equivalent of ``ci_lab.gitops.safe_path.safe_join`` (built in
parallel by M6). When that module is importable :func:`contained` runs both checks
(defence in depth); otherwise only the local one. :data:`DENIED_GLOBS` (the sealed rubric
vault) is rejected by :func:`normalize_rel`, so no model-supplied path can reach it.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

__all__ = [
    "DENIED_GLOBS",
    "PROTECTED_WRITE_GLOBS",
    "PathRejected",
    "check_writable",
    "components_for",
    "contained",
    "glob_match",
    "glob_prefix",
    "is_link_or_reparse",
    "matches_any",
    "normalize_rel",
    "safe_join",
]


class PathRejected(ValueError):
    """A relative path failed containment / allowlist checks."""


_WIN_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}
_SHORT_NAME = re.compile(r"~\d")
# Sealed hidden rubrics (ci_lab.taskgraph.vault.SEALED_DIRNAME): never readable by arm/student tools.
DENIED_GLOBS: tuple[str, ...] = ("**/sealed/**",)
# Governance-protected write scope: never writable/committable by meta-agent tools, even inside an
# editable surface. Must equal ``protected_globs`` in ci_lab/governance/policies/{meta_agents,campaign}.acs.yaml.
PROTECTED_WRITE_GLOBS: tuple[str, ...] = (
    "src/ci_lab/governance/**", "src/ci_lab/rules/**", "src/ci_lab/guards/**", ".github/workflows/**",
    "evals/**", "**/sealed/**", "**/.lkg/**", "**/lessons/**",
)


def check_writable(norm: str) -> str:
    """Raise :class:`PathRejected` if normalized ``norm`` is in :data:`PROTECTED_WRITE_GLOBS`."""
    if matches_any(norm.lower(), PROTECTED_WRITE_GLOBS):
        raise PathRejected(f"protected path (governance): {norm!r}")
    return norm


def normalize_rel(rel: str) -> str:
    """Validate a model-supplied relative path and return it in posix form.

    Rejects empty, absolute, drive/UNC, ``..``/``.``/empty segments, ``.git``, NUL, ADS
    (``:``), Windows trailing dot/space aliases, 8.3 short names, reserved device names and
    :data:`DENIED_GLOBS`.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise PathRejected("empty path")
    if "\x00" in rel:
        raise PathRejected("NUL in path")
    s = rel.replace("\\", "/")
    if PurePosixPath(s).is_absolute() or PureWindowsPath(rel).drive or PureWindowsPath(rel).root:
        raise PathRejected(f"absolute path not allowed: {rel!r}")
    parts = s.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise PathRejected(f"bad segment {part!r} in {rel!r}")
        low = part.lower()
        if low == ".git" or low.startswith(".git/"):
            raise PathRejected(f".git not allowed: {rel!r}")
        if ":" in part:
            raise PathRejected(f"':' not allowed: {rel!r}")
        if part != part.rstrip(". "):
            raise PathRejected(f"trailing dot/space alias: {rel!r}")
        if _SHORT_NAME.search(part):
            raise PathRejected(f"8.3 short-name alias: {rel!r}")
        if low.split(".")[0] in _WIN_RESERVED:
            raise PathRejected(f"reserved device name: {rel!r}")
    if matches_any(s.lower(), DENIED_GLOBS):
        raise PathRejected(f"denied path (sealed rubric vault): {rel!r}")
    return "/".join(parts)


def is_link_or_reparse(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(p))


def safe_join(root: Path | str, rel: str) -> Path:
    """Join ``rel`` under ``root``; every existing component must be a real (non-link,
    non-reparse) entry with exactly the requested case; result must stay under root."""
    root_p = Path(root).resolve(strict=True)
    norm = normalize_rel(rel)
    cur = root_p
    for part in norm.split("/"):
        nxt = cur / part
        if cur.is_dir():
            try:
                names = os.listdir(cur)
            except OSError as exc:
                raise PathRejected(f"cannot list {cur}: {exc}") from exc
            if part not in names:
                if any(n.lower() == part.lower() for n in names):
                    raise PathRejected(f"case alias for existing entry: {rel!r}")
            elif is_link_or_reparse(nxt):
                raise PathRejected(f"symlink/junction/reparse point in path: {rel!r}")
        cur = nxt
    resolved = cur.resolve(strict=False)
    try:
        resolved.relative_to(root_p)
    except ValueError:
        raise PathRejected(f"escapes root: {rel!r}") from None
    return cur


def _external_safe_join() -> object | None:
    try:
        from ci_lab.gitops.safe_path import (
            safe_join as ext,  # type: ignore[import-not-found]
        )
    except Exception:  # noqa: BLE001 - optional module built in parallel
        return None
    return ext


def contained(root: Path | str, rel: str) -> Path:
    """Local :func:`safe_join` plus ``ci_lab.gitops.safe_path.safe_join`` when available."""
    path = safe_join(root, rel)
    ext = _external_safe_join()
    if ext is not None:
        try:
            ext(Path(root), normalize_rel(rel))  # type: ignore[operator]
        except Exception as exc:
            raise PathRejected(f"gitops.safe_join rejected {rel!r}: {exc}") from exc
    return path


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    i, out, n = 0, [], len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(path: str, pattern: str) -> bool:
    """Posix glob match: ``*``/``?`` within a segment, ``**`` across segments
    (``a/**`` matches everything under ``a``; ``**/x`` matches ``x`` at any depth)."""
    return bool(_glob_regex(pattern).match(path.replace("\\", "/")))


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(glob_match(path, p) for p in patterns)


def glob_prefix(pattern: str) -> str:
    """Literal directory prefix of a glob (``src/a/**/*.md`` -> ``src/a``)."""
    parts: list[str] = []
    for part in pattern.split("/")[:-1]:
        if any(ch in part for ch in "*?["):
            break
        parts.append(part)
    return "/".join(parts)


def components_for(path: str, component_globs: Mapping[str, Sequence[str]]) -> set[str]:
    """Components whose globs match ``path``. ``memory`` files are not also ``skill``."""
    found = {c for c, globs in component_globs.items() if matches_any(path, globs)}
    if "memory" in found:
        found.discard("skill")
    return found
