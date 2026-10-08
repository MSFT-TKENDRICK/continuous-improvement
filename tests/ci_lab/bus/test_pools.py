"""ResourcePools: weighted FIFO admission, validation, cancellation, stats."""

import asyncio

import pytest

from ci_lab.bus.pools import ResourcePools


def test_validation():
    with pytest.raises(ValueError):
        ResourcePools({"llm": 0})
    with pytest.raises(ValueError):
        ResourcePools({"llm": True})
    pools = ResourcePools({"llm": 2})
    assert pools.capacities == {"llm": 2} and "llm" in pools and "s1" not in pools

    async def go(name, weight):
        async with pools.acquire(name, weight):
            pass

    with pytest.raises(ValueError, match="exceeds capacity"):
        asyncio.run(go("llm", 3))
    with pytest.raises(ValueError):
        asyncio.run(go("llm", 0))
    with pytest.raises(KeyError):
        asyncio.run(go("s1", 1))


def test_weighted_fifo_no_barging():
    pools = ResourcePools({"llm": 3})
    order: list[str] = []

    async def job(tag, weight, hold):
        async with pools.acquire("llm", weight):
            order.append(tag)
            await asyncio.sleep(hold)

    async def main():
        first = asyncio.create_task(job("a", 2, 0.05))
        await asyncio.sleep(0)
        heavy = asyncio.create_task(job("heavy", 3, 0))  # must wait for "a"
        await asyncio.sleep(0)
        light = asyncio.create_task(job("light", 1, 0))  # would fit now, but FIFO: no barging
        await asyncio.sleep(0.01)
        assert order == ["a"]
        assert pools.stats()["llm"].waiting == 2
        await asyncio.gather(first, heavy, light)

    asyncio.run(main())
    assert order == ["a", "heavy", "light"]
    s = pools.stats()["llm"]
    assert (s.in_use, s.waiting, s.granted, s.peak_in_use) == (0, 0, 3, 3)


def test_concurrency_bounded_by_capacity():
    pools = ResourcePools({"cpu": 2})
    live = peak = 0

    async def job():
        nonlocal live, peak
        async with pools.acquire("cpu"):
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1

    async def main():
        await asyncio.gather(*(job() for _ in range(7)))

    asyncio.run(main())
    assert peak == 2 and pools.stats()["cpu"].granted == 7


def test_cancelled_waiter_releases_queue_position():
    pools = ResourcePools({"s1": 1})
    got: list[str] = []

    async def job(tag):
        async with pools.acquire("s1"):
            got.append(tag)
            await asyncio.sleep(0.02)

    async def main():
        holder = asyncio.create_task(job("holder"))
        await asyncio.sleep(0)
        doomed = asyncio.create_task(job("doomed"))
        nxt = asyncio.create_task(job("next"))
        await asyncio.sleep(0.005)
        doomed.cancel()
        await asyncio.gather(holder, nxt)
        with pytest.raises(asyncio.CancelledError):
            await doomed

    asyncio.run(main())
    assert got == ["holder", "next"]
    assert pools.stats()["s1"].in_use == 0
