"""Host-wide admission control for local s1 judge backends (llama-server).

Every process that judges through the same llama-server (parallel ASSERT suites, campaign
arms, ``ci-lab judge``) takes a *slot lease* before sending work, so the server never sees
more than ``capacity`` judge calls at once and queued calls wait *here* instead of inside
a caller's per-request timeout.

* Leases are OS file locks on ``<lock root>/s1-<sha256(url)[:12]>/slot-<i>.lock``
  (``msvcrt.locking`` on Windows, ``fcntl.flock`` on POSIX). The OS drops them when the
  holder dies, so a killed process never wedges the queue.
* Lock root: ``$CI_S1_LOCK_DIR``, else ``%LOCALAPPDATA%\\ci-lab\\locks`` (Windows) or
  ``${XDG_RUNTIME_DIR:-${XDG_CACHE_HOME:-~/.cache}}/ci-lab/locks`` (POSIX).
* Capacity: ``$CI_S1_MAX_INFLIGHT`` (``0`` disables admission), else the server's
  ``/props`` ``total_slots`` capped at :data:`DEFAULT_MAX_INFLIGHT`, else 1. Measured on a
  CPU llama-server (``-np 4``, Qwen3.5-4B, real judge calls): 0.61 calls/min at 1 in flight
  vs 0.41-0.65 at 2-4, while per-call service time grows from ~80 s to ~290-450 s, so more
  in flight only adds latency (see docs/judge.md). All processes sharing a server should
  resolve the same capacity.
* Fairness: waiters drop a locked ticket file ``queue/<time_ns>-<pid>-...`` and only the
  ``capacity`` oldest live tickets compete for free slots, so service is FIFO up to
  windows of ``capacity`` (approximately FIFO). Tickets of dead waiters are reaped.
* Reentrancy: the held lease lives in a context variable. Nested ``hold`` calls for the
  same server in the same task, or in threads started from it (``asyncio.to_thread``
  copies the context), share the lease via a reference count instead of deadlocking.
  If the owner exits while a nested holder is still running (e.g. the caller's timeout
  cancelled an ``await asyncio.to_thread(...)``), the lease is *abandoned*: it stays
  locked until the nested holder finishes, and :meth:`Lease.check` raises
  :class:`LeaseAbandoned` so the orphaned work stops at its next request.
* Slot pinning (opt-in, ``$CI_S1_PIN_SLOT=1``): :attr:`Lease.id_slot` is the lease index
  when it is a valid server slot, and the llama.cpp backend sends it as ``id_slot``.
  llama-server accepts it on ``/v1/chat/completions``, but its own similarity-based slot
  choice already hit the prompt cache on every measured request, and pinning every lease
  to slot 0 at capacity 1 would make interleaved suites evict each other's cache.
* Waits are logged at INFO every :data:`LOG_EVERY_S` seconds (``logging``, never stdout);
  ``$CI_S1_ADMISSION_LOG=<dir>`` additionally appends one JSON line per lease
  (wait/held seconds, slot, capacity) to ``<dir>/admission-<pid>.jsonl``.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from ci_lab import obs

if os.name == "nt":
    import msvcrt
else:
    import fcntl

log = logging.getLogger(__name__)

MAX_INFLIGHT_ENV = "CI_S1_MAX_INFLIGHT"
LOCK_DIR_ENV = "CI_S1_LOCK_DIR"
PIN_SLOT_ENV = "CI_S1_PIN_SLOT"
ADMISSION_LOG_ENV = "CI_S1_ADMISSION_LOG"
DEFAULT_MAX_INFLIGHT = 1
LOG_EVERY_S = 30.0
TICKET_GRACE_S = 2.0
POLL_MIN_S = 0.05
POLL_MAX_S = 1.0
PROPS_TIMEOUT_S = 5.0
_PROPS_RETRY_S = 60.0
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


class LeaseAbandoned(RuntimeError):
    """The lease owner gave up (e.g. its timeout fired) while nested work was still running.

    Deliberately *not* a ``BackendError``: the provider must not fall back to another judge
    for work nobody is waiting for."""


# ------------------------------------------------------------------ OS file locks

def _try_lock(path: Path, *, create: bool = True) -> int | None:
    """Open ``path`` and take a non-blocking exclusive lock; the fd, or None if held elsewhere."""
    flags = os.O_RDWR | (os.O_CREAT if create else 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileNotFoundError:
        return None
    except PermissionError:
        return None  # Windows: being deleted / open without sharing
    try:
        if os.name == "nt":
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _unlock(fd: int) -> None:
    try:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        os.close(fd)


# ------------------------------------------------------------------ configuration

def normalize_url(base_url: str) -> str:
    """Canonical server identity: scheme://host:port with loopback spellings unified."""
    raw = (base_url or "").strip().rstrip("/")
    raw = raw.removesuffix("/v1")
    parts = urlsplit(raw if "://" in raw else "http://" + raw)
    host = (parts.hostname or "").lower()
    if host in _LOCAL_HOSTS:
        host = "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return f"{parts.scheme or 'http'}://{host}:{port}"


def lock_root() -> Path:
    env = os.environ.get(LOCK_DIR_ENV)
    if env:
        return Path(env)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "ci-lab" / "locks"
    base = os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ci-lab" / "locks"


def lock_dir(base_url: str) -> Path:
    url = normalize_url(base_url)
    return lock_root() / f"s1-{hashlib.sha256(url.encode()).hexdigest()[:12]}"


_props_cache: dict[str, tuple[float, int | None]] = {}
_props_lock = threading.Lock()


def server_slots(base_url: str, *, transport: httpx.BaseTransport | None = None) -> int | None:
    """``/props`` ``total_slots`` of the server (cached per process; None if unreachable)."""
    url = normalize_url(base_url)
    now = time.monotonic()
    with _props_lock:
        hit = _props_cache.get(url)
        if hit is not None and (hit[1] is not None or now - hit[0] < _PROPS_RETRY_S):
            return hit[1]
    slots: int | None = None
    try:
        with httpx.Client(timeout=PROPS_TIMEOUT_S, transport=transport) as c:
            r = c.get(url + "/props")
        if r.status_code == 200:
            v = r.json().get("total_slots")
            slots = int(v) if isinstance(v, int | float) and int(v) > 0 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        slots = None
    with _props_lock:
        _props_cache[url] = (now, slots)
    return slots


def _env_capacity() -> int | None:
    raw = (os.environ.get(MAX_INFLIGHT_ENV) or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError as e:
        raise ValueError(f"{MAX_INFLIGHT_ENV} must be an integer >= 0, got {raw!r}") from e
    if v < 0:
        raise ValueError(f"{MAX_INFLIGHT_ENV} must be an integer >= 0, got {raw!r}")
    return v


def resolve_capacity(base_url: str, *, total_slots: int | None = None,
                     transport: httpx.BaseTransport | None = None) -> tuple[int, int | None]:
    """(capacity, server total_slots). Capacity 0 means admission is disabled.

    ``total_slots`` short-circuits the ``/props`` lookup when the caller already knows it."""
    env = _env_capacity()
    if total_slots is None and env != 0:
        total_slots = server_slots(base_url, transport=transport)
    if env is not None:
        return env, total_slots
    return (min(total_slots, DEFAULT_MAX_INFLIGHT) if total_slots else 1), total_slots


def pin_enabled() -> bool:
    """Opt-in (``$CI_S1_PIN_SLOT=1``): measured no gain, and it defeats llama.cpp's own slot choice."""
    return os.environ.get(PIN_SLOT_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------------ leases

@dataclass(eq=False)
class Lease:
    base_url: str
    index: int
    capacity: int
    total_slots: int | None = None
    waited_s: float = 0.0
    _fd: int | None = field(default=None, repr=False)
    _refs: int = field(default=1, repr=False)
    _owner_done: bool = field(default=False, repr=False)
    _orphaned: bool = field(default=False, repr=False)
    _acquired: float = field(default_factory=time.monotonic, repr=False)
    _mu: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def held(self) -> bool:
        return self._fd is not None or self.index < 0

    @property
    def abandoned(self) -> bool:
        return self._owner_done

    @property
    def id_slot(self) -> int | None:
        """llama.cpp slot to pin, when the lease index is a valid server slot."""
        if self.index < 0 or not self.total_slots or self.index >= self.total_slots or not pin_enabled():
            return None
        return self.index

    def check(self) -> None:
        if self._owner_done:
            raise LeaseAbandoned(f"s1 judge lease {self.index} on {self.base_url} was abandoned by its owner")

    def retain(self) -> bool:
        with self._mu:
            if self._owner_done or not self.held:
                return False
            self._refs += 1
            return True

    def release(self) -> None:
        fd = None
        with self._mu:
            self._refs -= 1
            if self._refs <= 0 and self._fd is not None:
                fd, self._fd = self._fd, None
        if fd is not None:
            _unlock(fd)
            _record(self)

    def _owner_exit(self) -> None:
        with self._mu:
            self._owner_done = True
            self._orphaned = self._refs > 1
        self.release()


_current: contextvars.ContextVar[Lease | None] = contextvars.ContextVar("ci_s1_lease", default=None)
_slot_pref: dict[str, int] = {}


def current_lease(base_url: str | None = None) -> Lease | None:
    lease = _current.get()
    if lease is None or (base_url is not None and lease.base_url != normalize_url(base_url)):
        return None
    return lease


def _record(lease: Lease) -> None:
    held = time.monotonic() - lease._acquired
    d = os.environ.get(ADMISSION_LOG_ENV)
    if not d:
        return
    row = {"ts": time.time(), "pid": os.getpid(), "url": lease.base_url, "slot": lease.index,
           "capacity": lease.capacity, "total_slots": lease.total_slots,
           "wait_s": round(lease.waited_s, 3), "held_s": round(held, 3), "abandoned": lease._orphaned}
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        with open(Path(d) / f"admission-{os.getpid()}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        log.debug("could not write admission log to %s", d)


class _Waiter:
    """One queued acquisition: a locked ticket plus non-blocking slot attempts."""

    def __init__(self, base_url: str, capacity: int, total_slots: int | None) -> None:
        self.url = base_url
        self.capacity = capacity
        self.total_slots = total_slots
        self.dir = lock_dir(base_url)
        self.qdir = self.dir / "queue"
        self.qdir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        self._last_log = self.t0
        self._delay = POLL_MIN_S
        self.position = 0
        name = f"{time.time_ns():020d}-{os.getpid()}-{threading.get_ident() % 10**8:08d}-{os.urandom(3).hex()}"
        self.ticket = self.qdir / f"{name}.ticket"
        self._tfd = _try_lock(self.ticket)

    def _ahead(self) -> int:
        """Live tickets older than ours (reaping tickets whose waiter died)."""
        mine = self.ticket.name
        ahead = 0
        now_ns = time.time_ns()
        try:
            names = sorted(p.name for p in self.qdir.iterdir() if p.suffix == ".ticket")
        except OSError:
            return 0
        for n in names:
            if n >= mine:
                break
            try:
                born = int(n.split("-", 1)[0])
            except ValueError:
                born = 0
            if (now_ns - born) / 1e9 < TICKET_GRACE_S:
                ahead += 1
                continue
            path = self.qdir / n
            fd = _try_lock(path, create=False)
            if fd is None:
                ahead += int(path.exists())
                continue
            _unlock(fd)
            try:
                path.unlink()
            except OSError:
                pass
        return ahead

    def attempt(self) -> Lease | None:
        self.position = self._ahead()
        if self.position < self.capacity:
            pref = _slot_pref.get(self.url, 0) % self.capacity
            for k in range(self.capacity):
                i = (pref + k) % self.capacity
                fd = _try_lock(self.dir / f"slot-{i}.lock")
                if fd is not None:
                    _slot_pref[self.url] = i
                    return Lease(self.url, i, self.capacity, self.total_slots,
                                 waited_s=time.monotonic() - self.t0, _fd=fd)
        now = time.monotonic()
        if now - self._last_log >= LOG_EVERY_S:
            self._last_log = now
            log.info("s1 judge queue: waited %.0fs for %s (%d ahead, capacity %d)",
                     now - self.t0, self.url, self.position, self.capacity)
        return None

    def next_delay(self) -> float:
        d = min(self._delay * (0.5 + random.random()), POLL_MAX_S)
        self._delay = min(self._delay * 1.5, POLL_MAX_S)
        return d

    def close(self) -> None:
        if self._tfd is not None:
            _unlock(self._tfd)
            self._tfd = None
        try:
            self.ticket.unlink()
        except OSError:
            pass


def _acquired(lease: Lease) -> Lease:
    if lease.waited_s >= LOG_EVERY_S:
        log.info("s1 judge queue: lease %d/%d on %s after %.0fs", lease.index, lease.capacity,
                 lease.base_url, lease.waited_s)
    obs.annotate({"ci.s1.queue_wait_s": round(lease.waited_s, 3), "ci.s1.slot": lease.index,
                  "ci.s1.capacity": lease.capacity})
    return lease


def _unmanaged(url: str, total_slots: int | None) -> Lease:
    return Lease(url, -1, 0, total_slots)


def acquire(base_url: str, *, total_slots: int | None = None) -> Lease:
    """Block (outside any request timeout) until a slot lease is free."""
    url = normalize_url(base_url)
    cap, slots = resolve_capacity(url, total_slots=total_slots)
    if cap == 0:
        return _unmanaged(url, slots)
    w = _Waiter(url, cap, slots)
    try:
        while True:
            lease = w.attempt()
            if lease is not None:
                return _acquired(lease)
            time.sleep(w.next_delay())
    finally:
        w.close()


async def acquire_async(base_url: str, *, total_slots: int | None = None) -> Lease:
    url = normalize_url(base_url)
    cap, slots = await asyncio.to_thread(resolve_capacity, url, total_slots=total_slots)
    if cap == 0:
        return _unmanaged(url, slots)
    w = _Waiter(url, cap, slots)
    try:
        while True:
            lease = w.attempt()
            if lease is not None:
                return _acquired(lease)
            await asyncio.sleep(w.next_delay())
    finally:
        w.close()


def _reset(token: contextvars.Token) -> None:
    try:
        _current.reset(token)
    except ValueError:  # exited from another context (e.g. generator finalised by GC)
        _current.set(None)


def _reenter(url: str) -> Lease | None:
    cur = _current.get()
    if cur is None or cur.base_url != url:
        return None
    cur.check()
    return cur if cur.retain() else None


@contextmanager
def hold(base_url: str, *, total_slots: int | None = None) -> Iterator[Lease]:
    """Hold a slot lease for ``base_url`` (reentrant within a task / its threads)."""
    url = normalize_url(base_url)
    cur = _reenter(url)
    if cur is not None:
        try:
            yield cur
        finally:
            cur.release()
        return
    lease = acquire(url, total_slots=total_slots)
    token = _current.set(lease)
    try:
        yield lease
    finally:
        _reset(token)
        lease._owner_exit()


@asynccontextmanager
async def hold_async(base_url: str, *, total_slots: int | None = None) -> AsyncIterator[Lease]:
    url = normalize_url(base_url)
    cur = _reenter(url)
    if cur is not None:
        try:
            yield cur
        finally:
            cur.release()
        return
    lease = await acquire_async(url, total_slots=total_slots)
    token = _current.set(lease)
    try:
        yield lease
    finally:
        _reset(token)
        lease._owner_exit()


def s1_local_url(model: str, api_base: str | None = None) -> str | None:
    """Base URL of a local llama.cpp s1 judge model string (``s1/llamacpp/...``), else None."""
    from ci_lab.judge.backends import DEFAULT_LLAMA_URL, LLAMA_URL_ENV

    m = (model or "").strip()
    if not m.startswith("s1/"):
        return None
    kind = m[3:].split("/", 1)[0]
    if kind not in ("llamacpp", "local"):
        return None
    return api_base or os.environ.get(LLAMA_URL_ENV) or DEFAULT_LLAMA_URL


def describe(base_url: str) -> dict[str, Any]:
    """Resolved admission settings for diagnostics / run summaries."""
    url = normalize_url(base_url)
    cap, slots = resolve_capacity(url)
    return {"url": url, "capacity": cap, "total_slots": slots, "lock_dir": str(lock_dir(url)),
            "pin_slot": pin_enabled()}
