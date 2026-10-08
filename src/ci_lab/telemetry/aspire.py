"""Optional local Aspire dashboard (design §12.2, C30–C34): download, verify, launch, stop.

The dashboard is an external, on-demand dev viewer (like a browser), not a harness
dependency (C33). Everything binds to loopback, every endpoint is keyed with fresh random
secrets, and the secrets live only in the owner-only state file (C30).
"""
from __future__ import annotations

import contextlib
import ctypes
import datetime as _dt
import getpass
import hashlib
import json
import logging
import os
import platform
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ci_lab.contracts import DASHBOARD_STATE_ENV

log = logging.getLogger(__name__)

PACKAGE = "Aspire.Dashboard.Sdk"
LOCK_FILE = Path(__file__).with_name("aspire.lock.json")
FEED_ENV = "CI_NUGET_FLAT"
DEFAULT_FEED = ("https://ms-feed-25.pkgs.visualstudio.com/6f084628-a36d-42cb-934d-057357e379dc/"
                "_packaging/52ee4efa-5537-4eae-acb1-a914a4cca686/nuget/v3/flat2/")
FALLBACK_FEED = "https://api.nuget.org/v3-flatcontainer/"
HOME_ENV = "CI_ASPIRE_HOME"   # override install root
RID_ENV = "CI_ASPIRE_RID"     # override RID detection
RIDS = ("win-arm64", "win-x64", "linux-x64", "linux-arm64", "osx-arm64", "osx-x64")
SECRET_FIELDS = ("browser_token", "otlp_key", "api_key")
TELEMETRY_LIMITS = {"MaxTraceCount": "20000", "MaxLogCount": "20000", "MaxMetricsCount": "5000",
                    "MaxAttributeCount": "128", "MaxAttributeLength": "16384",
                    "MaxSpanEventCount": "256"}


class DashboardError(RuntimeError):
    pass


class IntegrityError(DashboardError):
    pass


class UntrustedPackageError(DashboardError):
    pass


# ---------------------------------------------------------------- platform

def _windows_native_arch() -> str | None:
    """Host arch even under x64 emulation on ARM64 (platform.machine() reports AMD64)."""
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        fn = k32.IsWow64Process2
    except (AttributeError, OSError):
        return None
    from ctypes import wintypes
    proc_m, native_m = wintypes.USHORT(), wintypes.USHORT()
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT)]
    fn.restype = wintypes.BOOL
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    if not fn(k32.GetCurrentProcess(), ctypes.byref(proc_m), ctypes.byref(native_m)):
        return None
    return {0xAA64: "arm64", 0x8664: "x64", 0x014C: "x86"}.get(native_m.value)


def _macos_is_arm() -> bool:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return out == "1"
    except (OSError, subprocess.SubprocessError):
        return platform.machine().lower() == "arm64"


def detect_rid() -> str:
    if os.environ.get(RID_ENV):
        return os.environ[RID_ENV]
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64", "armv8l", "armv8b")
    if sys.platform == "win32":
        native = _windows_native_arch()
        arm = native == "arm64" if native else arm
        return "win-arm64" if arm else "win-x64"
    if sys.platform == "darwin":
        return "osx-arm64" if _macos_is_arm() else "osx-x64"
    return "linux-arm64" if arm else "linux-x64"


def install_root() -> Path:
    if os.environ.get(HOME_ENV):
        return Path(os.environ[HOME_ENV])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / "Library" / "Application Support")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "ci-lab" / "aspire-dashboard"


def install_dir(version: str) -> Path:
    return install_root() / version


def exe_path(version: str) -> Path:
    name = "Aspire.Dashboard.exe" if sys.platform == "win32" else "Aspire.Dashboard"
    return install_dir(version) / "pkg" / "tools" / name


# ---------------------------------------------------------------- lockfile

def user_lock_path() -> Path:
    return install_root() / "aspire.lock.json"


def load_lock(path: Path | None = None) -> dict[str, Any]:
    try:
        return json.loads((path or LOCK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def default_version() -> str:
    return load_lock().get("default_version", "13.6.1")


def expected_sha(rid: str, version: str) -> str | None:
    for p in (LOCK_FILE, user_lock_path()):
        sha = load_lock(p).get("packages", {}).get(version, {}).get(rid)
        if sha:
            return sha.upper()
    return None


def record_sha(rid: str, version: str, sha: str) -> Path:
    """TOFU-record a pin. Prefer the packaged lock (a reviewable repo diff in editable
    installs); fall back to a per-user lock when the package dir is read-only."""
    for p in (LOCK_FILE, user_lock_path()):
        data = load_lock(p) or {"package": PACKAGE, "default_version": version, "packages": {}}
        data.setdefault("packages", {}).setdefault(version, {})[rid] = sha.upper()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            return p
        except OSError:
            continue
    raise DashboardError("cannot record sha256 pin in any lock file")


# ---------------------------------------------------------------- download / extract

def feeds() -> list[str]:
    primary = os.environ.get(FEED_ENV) or DEFAULT_FEED
    out = [primary if primary.endswith("/") else primary + "/"]
    if FALLBACK_FEED not in out:
        out.append(FALLBACK_FEED)
    return out


def package_url(base: str, rid: str, version: str) -> str:
    pid = f"{PACKAGE}.{rid}".lower()
    ver = version.lower()
    return f"{base}{pid}/{ver}/{pid}.{ver}.nupkg"


def _urlopen(url: str, timeout: float = 120):  # patched in tests
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - https feed URLs only


def download(url: str, dest: Path) -> str:
    """Stream ``url`` to ``dest``; return upper-case sha256."""
    h = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _urlopen(url) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest().upper()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest().upper()


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a nupkg (zip) refusing absolute paths, drive letters, ``..`` and symlinks."""
    root = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = urllib.parse.unquote(info.filename).replace("\\", "/")
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if (name.startswith("/") or ".." in parts or (parts and ":" in parts[0])
                    or ((info.external_attr >> 16) & 0o170000) == 0o120000):
                raise IntegrityError(f"unsafe archive entry: {info.filename!r}")
            if not parts:
                continue
            target = root.joinpath(*parts).resolve()
            if target != root and root not in target.parents:
                raise IntegrityError(f"unsafe archive entry: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if os.name != "nt" and parts[0] == "tools":
                target.chmod(0o755)


def ensure_installed(version: str | None = None, rid: str | None = None, *,
                     trust_new: bool = False) -> Path:
    """Return the dashboard executable, downloading + verifying it if needed (C34)."""
    version = version or default_version()
    rid = rid or detect_rid()
    if rid not in RIDS:
        raise DashboardError(f"unsupported RID {rid!r}; expected one of {RIDS}")
    d = install_dir(version)
    exe = exe_path(version)
    marker = d / "install.json"
    want = expected_sha(rid, version)
    if exe.is_file():  # reuse a verified install; otherwise fall through and reinstall
        info = load_lock(marker)
        if info:
            if info.get("rid") == rid and (want is None or info.get("sha256", "").upper() == want):
                return exe
        elif want and (d / "pkg.nupkg").is_file() and sha256_file(d / "pkg.nupkg") == want:
            marker.write_text(json.dumps({"rid": rid, "version": version, "sha256": want}), encoding="utf-8")
            return exe
    if want is None and not trust_new:
        raise UntrustedPackageError(
            f"no sha256 pin for {PACKAGE}.{rid} {version} in {LOCK_FILE.name}; "
            "re-run with --trust-new to download and record it (TOFU)")
    nupkg = d / "pkg.nupkg"
    part = d / "pkg.nupkg.part"
    errors: list[str] = []
    sha = None
    for base in feeds():
        url = package_url(base, rid, version)
        try:
            sha = download(url, part)
            break
        except (urllib.error.URLError, OSError, ValueError) as exc:
            errors.append(f"{urllib.parse.urlsplit(url).netloc}: {exc}")
            part.unlink(missing_ok=True)
    if sha is None:
        raise DashboardError("download failed: " + "; ".join(errors))
    if want is not None and sha != want:
        part.unlink(missing_ok=True)
        raise IntegrityError(f"sha256 mismatch for {PACKAGE}.{rid} {version}: got {sha}, pinned {want}")
    if want is None:
        where = record_sha(rid, version, sha)
        log.warning("TOFU: recorded sha256 %s for %s.%s %s in %s", sha, PACKAGE, rid, version, where)
    os.replace(part, nupkg)
    tmp = d / f"pkg.tmp-{secrets.token_hex(4)}"
    try:
        safe_extract(nupkg, tmp)
        if (d / "pkg").exists():
            shutil.rmtree(d / "pkg")
        os.replace(tmp, d / "pkg")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    marker.write_text(json.dumps({"rid": rid, "version": version, "sha256": sha}), encoding="utf-8")
    if not exe.is_file():
        raise DashboardError(f"package did not contain {exe.relative_to(d)}")
    return exe


# ---------------------------------------------------------------- state file

def state_path() -> Path:
    return Path(os.environ.get(DASHBOARD_STATE_ENV) or Path.home() / ".ci-lab" / "dashboard.json")


def restrict_permissions(p: Path) -> None:
    """Owner-only access: chmod 600 / icacls inheritance removed + current user full."""
    if os.name == "nt":
        user = os.environ.get("USERNAME") or getpass.getuser()
        r = subprocess.run(["icacls", str(p), "/inheritance:r", "/grant:r", f"{user}:F"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("icacls failed on %s (rc=%s)", p, r.returncode)
    else:
        os.chmod(p, 0o600)


def write_state(state: dict[str, Any], path: Path | None = None) -> Path:
    p = path or state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        restrict_permissions(tmp)  # lock down before any secret is written; ACL survives rename
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def read_state(path: Path | None = None) -> dict[str, Any] | None:
    try:
        data = json.loads((path or state_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def public(state: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (state or {}).items() if k not in SECRET_FIELDS}


def login_url(state: dict[str, Any]) -> str:
    return f"{state['ui_url'].rstrip('/')}/login?t={urllib.parse.quote(state['browser_token'])}"


# ---------------------------------------------------------------- processes

if os.name == "nt":
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _QUERY, _TERMINATE, _SYNC = 0x1000, 0x0001, 0x00100000

    @contextlib.contextmanager
    def _handle(pid: int, access: int):
        h = _k32.OpenProcess(access, False, pid)
        try:
            yield h
        finally:
            if h:
                _k32.CloseHandle(h)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        with _handle(pid, _QUERY) as h:
            if not h:
                return False
            code = wintypes.DWORD()
            return bool(_k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_image(pid: int) -> str | None:
    """Absolute executable path of ``pid`` (None if unknown)."""
    if os.name == "nt":
        with _handle(pid, _QUERY) as h:
            if not h:
                return None
            buf = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buf))
            if not _k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return None
            return buf.value
    with contextlib.suppress(OSError):
        return os.readlink(f"/proc/{pid}/exe")
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _same_file(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _terminate(pid: int, timeout: float = 10.0) -> None:
    if os.name == "nt":
        with _handle(pid, _TERMINATE | _SYNC) as h:
            if not h:
                return
            _k32.TerminateProcess(h, 1)
            _k32.WaitForSingleObject(h, int(timeout * 1000))
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and pid_alive(pid):
        time.sleep(0.1)
    if pid_alive(pid):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- HTTP

_LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy loopback


def http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with _LOOPBACK.open(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def probe(state: dict[str, Any], timeout: float = 1.0) -> bool:
    """True when the dashboard API answers with the recorded key."""
    try:
        code, _ = http_get(f"{state['api_url'].rstrip('/')}/api/telemetry/resources",
                           {"x-api-key": state["api_key"]}, timeout)
    except (OSError, KeyError, ValueError):
        return False
    return code == 200


def live_state(path: Path | None = None, timeout: float = 1.0) -> dict[str, Any] | None:
    """State of a running, answering dashboard; None if absent or dead."""
    st = read_state(path)
    if not st or not pid_alive(int(st.get("pid") or 0)):
        return None
    return st if probe(st, timeout) else None


def query(path: str, state: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
    """GET ``/api/telemetry/<path>`` (traces, spans, logs, resources) as parsed JSON."""
    st = state or live_state()
    if not st:
        raise DashboardError("dashboard is not running (ci-lab dashboard up)")
    code, body = http_get(f"{st['api_url'].rstrip('/')}/api/telemetry/{path.lstrip('/')}",
                          {"x-api-key": st["api_key"]}, timeout)
    if code != 200:
        raise DashboardError(f"dashboard API returned HTTP {code}")
    return json.loads(body)


# ---------------------------------------------------------------- lifecycle

def build_env(*, ui_port: int, otlp_port: int, browser_token: str, otlp_key: str,
              api_key: str, grpc_port: int | None = None,
              base: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not k.startswith(("OTEL_", "Dashboard__", "ASPIRE_", "ASPNETCORE_", "DOTNET_DASHBOARD"))}
    env.update({
        "ASPNETCORE_URLS": f"http://127.0.0.1:{ui_port}",
        "ASPIRE_DASHBOARD_OTLP_HTTP_ENDPOINT_URL": f"http://127.0.0.1:{otlp_port}",
        "ASPIRE_ALLOW_UNSECURED_TRANSPORT": "true",
        "ASPIRE_DASHBOARD_SUPPRESS_BROWSER_TOKEN_IN_OUTPUT": "true",
        "Dashboard__Frontend__AuthMode": "BrowserToken",
        "Dashboard__Frontend__BrowserToken": browser_token,
        "Dashboard__Otlp__AuthMode": "ApiKey",
        "Dashboard__Otlp__PrimaryApiKey": otlp_key,
        "Dashboard__Api__Enabled": "true",
        "Dashboard__Api__AuthMode": "ApiKey",
        "Dashboard__Api__PrimaryApiKey": api_key,
        "AllowedHosts": "127.0.0.1;localhost",
    })
    if grpc_port:
        env["ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL"] = f"http://127.0.0.1:{grpc_port}"
    for k, v in TELEMETRY_LIMITS.items():
        env[f"Dashboard__TelemetryLimits__{k}"] = v
    return env


_CHILDREN: list[subprocess.Popen] = []  # keep Popen objects alive (no ResourceWarning on GC)


def _spawn(exe: Path, env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.touch()
        restrict_permissions(log_path)
    with open(log_path, "ab") as logf:
        kw: dict[str, Any] = dict(stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                                  cwd=str(exe.parent), env=env, close_fds=True)
        if os.name == "nt":
            flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                     | subprocess.CREATE_NO_WINDOW)
            try:
                proc = subprocess.Popen([str(exe)], creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
            except OSError:
                proc = subprocess.Popen([str(exe)], creationflags=flags, **kw)
        else:
            proc = subprocess.Popen([str(exe)], start_new_session=True, **kw)
    _CHILDREN.append(proc)
    return proc


def _log_tail(p: Path, n: int = 20) -> str:
    try:
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def up(*, version: str | None = None, rid: str | None = None, trust_new: bool = False,
       grpc: bool = False, wait: float = 60.0, path: Path | None = None,
       spawn: Callable[[Path, dict[str, str], Path], Any] = _spawn) -> dict[str, Any]:
    """Start (or reuse) the dashboard; returns the public (secret-free) state."""
    sp = path or state_path()
    cur = live_state(sp)
    if cur:
        return {**public(cur), "running": True, "already_running": True}
    version = version or default_version()
    rid = rid or detect_rid()
    exe = ensure_installed(version, rid, trust_new=trust_new)
    ui_port, otlp_port = free_port(), free_port()
    grpc_port = free_port() if grpc else None
    tokens = {f: secrets.token_urlsafe(32) for f in SECRET_FIELDS}
    env = build_env(ui_port=ui_port, otlp_port=otlp_port, grpc_port=grpc_port, **tokens)
    log_path = sp.with_name("dashboard.log")
    proc = spawn(exe, env, log_path)
    ui = f"http://127.0.0.1:{ui_port}"
    state: dict[str, Any] = {
        "pid": proc.pid, "version": version, "rid": rid, "exe": str(exe), "ui_url": ui,
        "otlp_url": f"http://127.0.0.1:{otlp_port}", "api_url": ui,
        "otlp_grpc_url": f"http://127.0.0.1:{grpc_port}" if grpc_port else None,
        "log": str(log_path),
        "started": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), **tokens,
    }
    write_state(state, sp)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sp.unlink(missing_ok=True)
            raise DashboardError(f"dashboard exited (rc={proc.returncode}); log tail:\n{_log_tail(log_path)}")
        if probe(state, timeout=1.0):
            return {**public(state), "running": True, "already_running": False}
        time.sleep(0.25)
    _terminate(proc.pid)
    sp.unlink(missing_ok=True)
    raise DashboardError(f"dashboard not ready after {wait:.0f}s; log tail:\n{_log_tail(log_path)}")


def down(path: Path | None = None) -> dict[str, Any]:
    """Stop the recorded dashboard — only if the pid's image is the recorded executable."""
    sp = path or state_path()
    st = read_state(sp)
    if not st:
        return {"stopped": False, "reason": "not running"}
    pid = int(st.get("pid") or 0)
    if not pid_alive(pid):
        sp.unlink(missing_ok=True)
        return {"stopped": False, "reason": "stale state removed"}
    image = process_image(pid)
    if not image or not st.get("exe") or not _same_file(image, st["exe"]):
        sp.unlink(missing_ok=True)
        return {"stopped": False, "reason": f"pid {pid} is not the recorded dashboard; state removed, process left alone"}
    _terminate(pid)
    if pid_alive(pid):
        return {"stopped": False, "reason": f"pid {pid} did not exit"}
    sp.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid}


def status(path: Path | None = None) -> dict[str, Any]:
    sp = path or state_path()
    st = read_state(sp)
    if not st:
        return {"running": False, "state_file": str(sp)}
    alive = pid_alive(int(st.get("pid") or 0))
    return {**public(st), "running": alive, "ready": alive and probe(st), "state_file": str(sp)}
