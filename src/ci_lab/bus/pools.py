"""Global weighted async resource pools (bus contract v2 §0.8, §9).

``ResourcePools({"s1": 1, "llm": 4, "cpu": 8})`` gives one weighted semaphore per named
resource. ``async with pools.acquire("llm", weight=2):`` waits in strict FIFO order (a
heavy waiter at the head is never starved by lighter late arrivals) and releases on exit,
including on cancellation. Callers start their own timeouts *after* admission.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

__all__ = ["PoolStats", "ResourcePools"]


@dataclass(frozen=True)
class PoolStats:
    capacity: int
    in_use: int
    waiting: int
    granted: int
    peak_in_use: int


class _Pool:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.in_use = 0
        self.granted = 0
        self.peak = 0
        self.waiters: deque[tuple[int, asyncio.Future[None]]] = deque()

    def _grant(self, weight: int) -> None:
        self.in_use += weight
        self.granted += 1
        self.peak = max(self.peak, self.in_use)

    def _wake(self) -> None:
        while self.waiters:
            weight, fut = self.waiters[0]
            if fut.done():
                self.waiters.popleft()
            elif self.in_use + weight <= self.capacity:
                self.waiters.popleft()
                self._grant(weight)
                fut.set_result(None)
            else:
                return

    async def take(self, weight: int) -> None:
        if not self.waiters and self.in_use + weight <= self.capacity:
            self._grant(weight)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (weight, fut)
        self.waiters.append(entry)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.give(weight)  # granted, then cancelled before the caller saw it
            else:
                if entry in self.waiters:
                    self.waiters.remove(entry)
                self._wake()
            raise

    def give(self, weight: int) -> None:
        self.in_use -= weight
        self._wake()

    def stats(self) -> PoolStats:
        return PoolStats(self.capacity, self.in_use, sum(not f.done() for _, f in self.waiters),
                         self.granted, self.peak)


class ResourcePools:
    """Named weighted semaphores; one instance is shared by every voter/agent of a run."""

    def __init__(self, capacities: Mapping[str, int]) -> None:
        self._pools: dict[str, _Pool] = {}
        for name, cap in capacities.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"bad pool name {name!r}")
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise ValueError(f"pool {name!r}: capacity must be an int >= 1, got {cap!r}")
            self._pools[name] = _Pool(cap)

    def __contains__(self, name: object) -> bool:
        return name in self._pools

    @property
    def capacities(self) -> dict[str, int]:
        return {k: p.capacity for k, p in self._pools.items()}

    def _pool(self, name: str, weight: int) -> _Pool:
        if name not in self._pools:
            raise KeyError(f"unknown resource pool {name!r}; have {sorted(self._pools)}")
        pool = self._pools[name]
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise ValueError(f"pool {name!r}: weight must be an int >= 1, got {weight!r}")
        if weight > pool.capacity:
            raise ValueError(f"pool {name!r}: weight {weight} exceeds capacity {pool.capacity}")
        return pool

    @asynccontextmanager
    async def acquire(self, name: str, weight: int = 1) -> AsyncIterator[None]:
        pool = self._pool(name, weight)
        await pool.take(weight)
        try:
            yield
        finally:
            pool.give(weight)

    def stats(self) -> dict[str, PoolStats]:
        return {k: p.stats() for k, p in self._pools.items()}
