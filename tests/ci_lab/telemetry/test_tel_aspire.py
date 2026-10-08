"""Aspire dashboard management, offline: RID, URLs, download/verify/TOFU, zip-slip,
state-file perms, up/down/status with fake processes."""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import zipfile

import pytest

from ci_lab.telemetry import aspire

EXE = "Aspire.Dashboard.exe" if sys.platform == "win32" else "Aspire.Dashboard"


def _nupkg(extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"tools/{EXE}", b"fake-dashboard")
        z.writestr("tools/Aspire.Dashboard.dll", b"dll")
        z.writestr("Aspire.Dashboard.Sdk.nuspec", b"<package/>")
        for k, v in (extra or {}).items():
            z.writestr(k, v)
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated install root, lock file and feeds; records requested URLs."""
    monkeypatch.setenv(aspire.HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setenv(aspire.FEED_ENV, "https://primary.example/flat")
    lock = tmp_path / "aspire.lock.json"
    lock.write_text(aspire.LOCK_FILE.read_text("utf-8"), encoding="utf-8")
    monkeypatch.setattr(aspire, "LOCK_FILE", lock)
    pkg = _nupkg()
    calls: list[str] = []
    fail: set[str] = set()

    def fake_urlopen(url, timeout=120):
        calls.append(url)
        if any(url.startswith(f) for f in fail):
            raise urllib.error.URLError("unreachable")
        return io.BytesIO(pkg)

    monkeypatch.setattr(aspire, "_urlopen", fake_urlopen)

    class E:
        pass

    e = E()
    e.lock, e.pkg, e.calls, e.fail, e.root = lock, pkg, calls, fail, tmp_path / "home"
    e.sha = hashlib.sha256(pkg).hexdigest().upper()
    e.pin = lambda rid="linux-x64", ver="1.0.0": aspire.record_sha(rid, ver, e.sha)
    return e


def test_lock_seeded_with_win_arm64_pin():
    assert aspire.default_version() == "13.6.1"
    assert aspire.load_lock()["packages"]["13.6.1"]["win-arm64"] == (
        "B53B7F4A20D884A0CB673F847CC2D4D4FF753D9B3612D9ECA605EFB5C8C45166")


def test_lock_pins_common_rids_for_default_version():
    pins = aspire.load_lock()["packages"][aspire.default_version()]
    assert {"win-arm64", "win-x64", "linux-x64", "osx-arm64"} <= set(pins)
    assert set(pins) <= set(aspire.RIDS)
    for sha in pins.values():
        assert len(sha) == 64 and sha == sha.upper() and all(c in "0123456789ABCDEF" for c in sha)


def test_rid_detection(monkeypatch):
    assert aspire.detect_rid() in aspire.RIDS
    monkeypatch.setenv(aspire.RID_ENV, "linux-arm64")
    assert aspire.detect_rid() == "linux-arm64"


def test_package_url_and_feeds(monkeypatch):
    assert aspire.package_url("https://f/", "win-arm64", "13.6.1") == (
        "https://f/aspire.dashboard.sdk.win-arm64/13.6.1/aspire.dashboard.sdk.win-arm64.13.6.1.nupkg")
    monkeypatch.delenv(aspire.FEED_ENV, raising=False)
    assert aspire.feeds() == [aspire.DEFAULT_FEED, aspire.FALLBACK_FEED]
    monkeypatch.setenv(aspire.FEED_ENV, "https://mirror/x")
    assert aspire.feeds() == ["https://mirror/x/", aspire.FALLBACK_FEED]


def test_install_paths(monkeypatch, tmp_path):
    monkeypatch.setenv(aspire.HOME_ENV, str(tmp_path))
    assert aspire.exe_path("13.6.1") == tmp_path / "13.6.1" / "pkg" / "tools" / EXE


def test_install_verified_then_reused(env):
    env.pin()
    exe = aspire.ensure_installed("1.0.0", "linux-x64")
    assert exe.read_bytes() == b"fake-dashboard"
    assert env.calls == [aspire.package_url("https://primary.example/flat/", "linux-x64", "1.0.0")]
    marker = json.loads((exe.parents[2] / "install.json").read_text("utf-8"))
    assert marker == {"rid": "linux-x64", "version": "1.0.0", "sha256": env.sha}
    assert aspire.ensure_installed("1.0.0", "linux-x64") == exe
    assert len(env.calls) == 1
    assert not list(exe.parents[2].glob("pkg.tmp-*")) and not (exe.parents[2] / "pkg.nupkg.part").exists()


def test_reuse_preextracted_install_without_marker(env):
    env.pin()
    d = aspire.install_dir("1.0.0")
    (d / "pkg" / "tools").mkdir(parents=True)
    (d / "pkg" / "tools" / EXE).write_bytes(b"x")
    (d / "pkg.nupkg").write_bytes(env.pkg)
    assert aspire.ensure_installed("1.0.0", "linux-x64").is_file()
    assert env.calls == [] and (d / "install.json").is_file()


def test_sha_mismatch_rejected(env):
    aspire.record_sha("linux-x64", "1.0.0", "00" * 32)
    with pytest.raises(aspire.IntegrityError, match="mismatch"):
        aspire.ensure_installed("1.0.0", "linux-x64")
    d = aspire.install_dir("1.0.0")
    assert not (d / "pkg").exists() and not (d / "pkg.nupkg.part").exists()


def test_unpinned_requires_trust_new_then_records(env):
    with pytest.raises(aspire.UntrustedPackageError, match="--trust-new"):
        aspire.ensure_installed("2.0.0", "osx-arm64")
    assert env.calls == []
    exe = aspire.ensure_installed("2.0.0", "osx-arm64", trust_new=True)
    assert exe.is_file()
    assert json.loads(env.lock.read_text("utf-8"))["packages"]["2.0.0"]["osx-arm64"] == env.sha
    assert aspire.expected_sha("osx-arm64", "2.0.0") == env.sha


def test_fallback_feed_used_when_primary_fails(env):
    env.pin()
    env.fail.add("https://primary.example/")
    aspire.ensure_installed("1.0.0", "linux-x64")
    assert len(env.calls) == 2 and env.calls[1].startswith(aspire.FALLBACK_FEED)


def test_all_feeds_fail(env):
    env.pin()
    env.fail.update({"https://primary.example/", aspire.FALLBACK_FEED})
    with pytest.raises(aspire.DashboardError, match="download failed"):
        aspire.ensure_installed("1.0.0", "linux-x64")


def test_unsupported_rid(env):
    with pytest.raises(aspire.DashboardError, match="unsupported RID"):
        aspire.ensure_installed("1.0.0", "freebsd-x64")


def _zip_with(tmp_path, name, *, symlink=False):
    p = tmp_path / "evil.zip"
    with zipfile.ZipFile(p, "w") as z:
        info = zipfile.ZipInfo(name)
        if symlink:
            info.external_attr = (0o120777 << 16)
        z.writestr(info, b"pwned")
    return p


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/abs/evil.txt",
                                  "C:/evil.txt", "..\\evil.txt", "%2e%2e/evil.txt"])
def test_safe_extract_rejects_traversal(tmp_path, name):
    with pytest.raises(aspire.IntegrityError):
        aspire.safe_extract(_zip_with(tmp_path, name), tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_rejects_symlink(tmp_path):
    with pytest.raises(aspire.IntegrityError):
        aspire.safe_extract(_zip_with(tmp_path, "tools/link", symlink=True), tmp_path / "out")


def test_safe_extract_ok(tmp_path):
    p = tmp_path / "ok.zip"
    p.write_bytes(_nupkg({"a%20b/c.txt": b"c"}))
    aspire.safe_extract(p, tmp_path / "out")
    assert (tmp_path / "out" / "tools" / EXE).read_bytes() == b"fake-dashboard"
    assert (tmp_path / "out" / "a b" / "c.txt").is_file()


def test_state_file_owner_only(tmp_path):
    p = aspire.write_state({"pid": 1, "api_key": "s3cret"}, tmp_path / "s" / "dashboard.json")
    assert aspire.read_state(p) == {"pid": 1, "api_key": "s3cret"}
    assert not list(p.parent.glob(".*.tmp"))
    if os.name == "nt":
        out = subprocess.run(["icacls", str(p)], check=False, capture_output=True, text=True).stdout
        aces = [ln for ln in out.splitlines()[:-2] if ":" in ln.split(str(p))[-1]]
        assert "(I)" not in out, out  # inheritance removed
        assert len(aces) == 1 and os.environ["USERNAME"].lower() in out.lower(), out
    else:
        assert (p.stat().st_mode & 0o777) == 0o600


def test_public_and_login_url():
    st = {"ui_url": "http://127.0.0.1:5/", "browser_token": "a b", "otlp_key": "k", "api_key": "q",
          "pid": 3}
    assert aspire.public(st) == {"ui_url": "http://127.0.0.1:5/", "pid": 3}
    assert aspire.login_url(st) == "http://127.0.0.1:5/login?t=a%20b"


def test_build_env_sets_auth_and_strips_inherited():
    base = {"PATH": "p", "OTEL_EXPORTER_OTLP_ENDPOINT": "x", "Dashboard__Otlp__AuthMode": "Unsecured",
            "ASPNETCORE_ENVIRONMENT": "Development", "ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL": "y"}
    env = aspire.build_env(ui_port=1, otlp_port=2, browser_token="b", otlp_key="o", api_key="a",
                           base=base)
    assert env["PATH"] == "p" and "OTEL_EXPORTER_OTLP_ENDPOINT" not in env
    assert "ASPNETCORE_ENVIRONMENT" not in env and "ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL" not in env
    assert env["ASPNETCORE_URLS"] == "http://127.0.0.1:1"
    assert env["ASPIRE_DASHBOARD_OTLP_HTTP_ENDPOINT_URL"] == "http://127.0.0.1:2"
    assert env["Dashboard__Otlp__AuthMode"] == "ApiKey" and env["Dashboard__Otlp__PrimaryApiKey"] == "o"
    assert env["Dashboard__Frontend__AuthMode"] == "BrowserToken"
    assert env["Dashboard__Frontend__BrowserToken"] == "b"
    assert env["Dashboard__Api__Enabled"] == "true" and env["Dashboard__Api__PrimaryApiKey"] == "a"
    assert env["AllowedHosts"] == "127.0.0.1;localhost"
    assert env["ASPIRE_DASHBOARD_SUPPRESS_BROWSER_TOKEN_IN_OUTPUT"] == "true"
    assert any(k.startswith("Dashboard__TelemetryLimits__") for k in env)
    grpc = aspire.build_env(ui_port=1, otlp_port=2, grpc_port=3, browser_token="b", otlp_key="o",
                            api_key="a", base={})
    assert grpc["ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL"] == "http://127.0.0.1:3"


class _FakeProc:
    def __init__(self, rc=None):
        self.pid, self.returncode = 424242, rc

    def poll(self):
        return self.returncode


def test_up_writes_state_and_hides_secrets(tmp_path, monkeypatch):
    sp = tmp_path / "dashboard.json"
    monkeypatch.setattr(aspire, "ensure_installed", lambda *a, **k: tmp_path / EXE)
    seen = {}

    def spawn(exe, env, log):
        seen.update(env=env, exe=exe, log=log)
        return _FakeProc()

    monkeypatch.setattr(aspire, "probe", lambda st, timeout=1.0: True)
    res = aspire.up(path=sp, spawn=spawn, rid="linux-x64")
    st = json.loads(sp.read_text("utf-8"))
    assert res["running"] and not res["already_running"]
    assert not set(aspire.SECRET_FIELDS) & set(res)
    assert all(len(st[f]) >= 40 for f in aspire.SECRET_FIELDS)
    assert len({st[f] for f in aspire.SECRET_FIELDS}) == 3
    assert seen["env"]["Dashboard__Api__PrimaryApiKey"] == st["api_key"]
    assert seen["env"]["Dashboard__Otlp__PrimaryApiKey"] == st["otlp_key"]
    assert st["api_url"] == st["ui_url"] and st["ui_url"].startswith("http://127.0.0.1:")
    assert st["otlp_grpc_url"] is None and seen["log"] == tmp_path / "dashboard.log"
    for v in (st[f] for f in aspire.SECRET_FIELDS):
        assert v not in json.dumps(res)


def test_up_reuses_live_instance(tmp_path, monkeypatch):
    live = {"pid": 1, "ui_url": "u", "api_key": "k"}
    monkeypatch.setattr(aspire, "live_state", lambda *a, **k: live)
    res = aspire.up(path=tmp_path / "s.json", spawn=lambda *a: pytest.fail("spawned"))
    assert res == {"pid": 1, "ui_url": "u", "running": True, "already_running": True}


def test_up_reports_early_exit(tmp_path, monkeypatch):
    sp = tmp_path / "dashboard.json"
    monkeypatch.setattr(aspire, "ensure_installed", lambda *a, **k: tmp_path / EXE)
    (tmp_path / "dashboard.log").write_text("Failed to bind\n", encoding="utf-8")
    with pytest.raises(aspire.DashboardError, match="Failed to bind"):
        aspire.up(path=sp, spawn=lambda *a: _FakeProc(rc=1), rid="linux-x64")
    assert not sp.exists()


def test_down_status_without_state(tmp_path):
    sp = tmp_path / "none.json"
    assert aspire.down(sp) == {"stopped": False, "reason": "not running"}
    assert aspire.status(sp) == {"running": False, "state_file": str(sp)}


def test_down_removes_stale_state(tmp_path, monkeypatch):
    sp = aspire.write_state({"pid": 999999, "exe": "x"}, tmp_path / "s.json")
    monkeypatch.setattr(aspire, "pid_alive", lambda pid: False)
    assert aspire.down(sp)["reason"] == "stale state removed" and not sp.exists()


@pytest.fixture
def sleeper():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield proc
    proc.kill()
    proc.wait(10)


def test_down_refuses_foreign_process(tmp_path, sleeper):
    other = tmp_path / EXE
    other.write_bytes(b"x")
    sp = aspire.write_state({"pid": sleeper.pid, "exe": str(other)}, tmp_path / "s.json")
    res = aspire.down(sp)
    assert res["stopped"] is False and "left alone" in res["reason"]
    assert sleeper.poll() is None and not sp.exists()


def test_down_stops_verified_process(tmp_path, sleeper):
    time.sleep(0.2)
    image = aspire.process_image(sleeper.pid)
    assert image and aspire._same_file(image, image)
    sp = aspire.write_state({"pid": sleeper.pid, "exe": image, "api_url": "http://127.0.0.1:9",
                             "api_key": "k"}, tmp_path / "s.json")
    st = aspire.status(sp)
    assert st["running"] is True and st["ready"] is False and "api_key" not in st
    assert aspire.down(sp) == {"stopped": True, "pid": sleeper.pid}
    assert sleeper.wait(10) is not None and not sp.exists()


def test_probe_and_live_state_use_api_key(monkeypatch, tmp_path):
    seen = []

    def fake_get(url, headers, timeout):
        seen.append((url, headers))
        return (200 if headers.get("x-api-key") == "good" else 401), b"[]"

    monkeypatch.setattr(aspire, "http_get", fake_get)
    st = {"api_url": "http://127.0.0.1:7/", "api_key": "good", "pid": os.getpid()}
    assert aspire.probe(st) and seen[-1][0] == "http://127.0.0.1:7/api/telemetry/resources"
    assert not aspire.probe({**st, "api_key": "bad"})
    sp = aspire.write_state(st, tmp_path / "s.json")
    assert aspire.live_state(sp) == st
    assert aspire.query("traces", st) == []
    aspire.write_state({**st, "pid": 0}, sp)
    assert aspire.live_state(sp) is None