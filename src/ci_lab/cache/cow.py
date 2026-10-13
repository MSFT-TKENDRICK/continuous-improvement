"""Copy-on-write capability detection (design §4; research-cow §1).

``detect(path)`` returns how files should be materialized next to ``path``:

* ``clone``    — CoW block cloning/reflinks: ReFS / Dev Drive on Windows (volume file
  system read with ``GetVolumeInformationW`` via ctypes, no subprocess), APFS
  ``clonefile`` on macOS, ``FICLONE`` reflinks on Linux (btrfs, xfs reflink=1, ...);
* ``hardlink`` — same-volume hard links (NTFS, ext4, ...);
* ``copy``     — byte copies (FAT/exFAT, network shares, cross-volume caches).

POSIX detection probes with a tiny file pair in a scratch dir next to ``path``.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CowMode = Literal["clone", "hardlink", "copy"]
MODES: tuple[CowMode, ...] = ("clone", "hardlink", "copy")

FILE_SUPPORTS_HARD_LINKS = 0x00400000
FILE_SUPPORTS_BLOCK_REFCOUNTING = 0x08000000
FICLONE = 0x40049409
_CLONE_FS = {"refs", "apfs", "btrfs", "xfs", "bcachefs", "zfs"}


@dataclass(frozen=True)
class CowInfo:
    mode: CowMode
    filesystem: str | None
    reason: str


def _existing_dir(path: Path) -> Path:
    p = Path(path).absolute()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p if p.is_dir() else p.parent


# ---------------------------------------------------------------- Windows


def windows_volume_info(path: str | os.PathLike[str]) -> tuple[str, str, int] | None:
    """``(volume_root, filesystem_name, fs_flags)`` via kernel32, or ``None``."""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetVolumePathNameW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    k32.GetVolumeInformationW.argtypes = (
        wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD)
    target = str(_existing_dir(Path(path)))
    root = ctypes.create_unicode_buffer(1024)
    if not k32.GetVolumePathNameW(target, root, len(root)):
        return None
    fs_name = ctypes.create_unicode_buffer(64)
    serial, max_comp, flags = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
    if not k32.GetVolumeInformationW(root.value, None, 0, ctypes.byref(serial), ctypes.byref(max_comp),
                                     ctypes.byref(flags), fs_name, len(fs_name)):
        return None
    return root.value, fs_name.value, flags.value


def _detect_windows(path: Path) -> CowInfo:
    info = windows_volume_info(path)
    if info is None:
        return CowInfo("copy", None, "volume information unavailable")
    _root, fs, flags = info
    if fs.upper() == "REFS" or flags & FILE_SUPPORTS_BLOCK_REFCOUNTING:
        return CowInfo("clone", fs, "ReFS / Dev Drive block cloning")
    if flags & FILE_SUPPORTS_HARD_LINKS or fs.upper() == "NTFS":
        return CowInfo("hardlink", fs, f"{fs}: no block cloning; hard links supported")
    return CowInfo("copy", fs, f"{fs}: no block cloning or hard links")


# ---------------------------------------------------------------- POSIX


def linux_filesystem(path: str | os.PathLike[str]) -> str | None:
    """Filesystem type of the mount containing ``path`` (from ``/proc/self/mountinfo``)."""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    target = os.path.realpath(_existing_dir(Path(path)))
    best, fstype = -1, None
    for line in lines:
        left, _, right = line.partition(" - ")
        fields = left.split()
        if len(fields) < 5 or not right:
            continue
        mnt = fields[4].replace("\\040", " ")
        if (target == mnt or target.startswith(mnt.rstrip("/") + "/")) and len(mnt) > best:
            best, fstype = len(mnt), right.split()[0]
    return fstype


def reflink_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> bool:
    """CoW-clone one file (Linux ``FICLONE`` / macOS ``clonefile``); ``False`` if unsupported."""
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            clonefile = libc.clonefile
        except (OSError, AttributeError):
            return False
        clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        return clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0
    if sys.platform.startswith("linux"):
        import fcntl

        with open(src, "rb") as s:
            try:
                with open(dst, "wb") as d:
                    fcntl.ioctl(d.fileno(), FICLONE, s.fileno())
                return True
            except OSError:
                try:
                    os.unlink(dst)
                except OSError:
                    pass
                return False
    return False


def _detect_posix(path: Path) -> CowInfo:
    fs = linux_filesystem(path) if sys.platform.startswith("linux") else ("apfs?" if sys.platform == "darwin"
                                                                            else None)
    base = _existing_dir(path)
    try:
        scratch = Path(tempfile.mkdtemp(prefix=".cow-probe-", dir=base))
    except OSError as exc:
        return CowInfo("copy", fs, f"cannot probe {base}: {exc.strerror}")
    try:
        src = scratch / "a"
        src.write_bytes(b"cow-probe")
        if reflink_file(src, scratch / "b"):
            return CowInfo("clone", fs, "reflink/clonefile probe succeeded")
        try:
            os.link(src, scratch / "c")
            return CowInfo("hardlink", fs, "no reflink; hard link probe succeeded")
        except OSError:
            return CowInfo("copy", fs, "no reflink or hard links")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------- API


def same_volume(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    return os.stat(_existing_dir(Path(a))).st_dev == os.stat(_existing_dir(Path(b))).st_dev


def detect(path: str | os.PathLike[str], *, source: str | os.PathLike[str] | None = None) -> CowInfo:
    """CoW mode for files materialized at ``path``. When ``source`` (e.g. the uv cache or a
    golden venv) is on another volume, neither clones nor hard links work: ``copy``."""
    p = Path(path)
    info = _detect_windows(p) if os.name == "nt" else _detect_posix(p)
    if source is not None and info.mode != "copy" and not same_volume(p, source):
        return CowInfo("copy", info.filesystem, f"{info.reason}; but source is on another volume")
    return info


def cow_mode(path: str | os.PathLike[str], *, source: str | os.PathLike[str] | None = None) -> CowMode:
    return detect(path, source=source).mode
