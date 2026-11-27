import asyncio

import pytest

from switchboard.lifecycle import Lifecycle
from switchboard.readiness import NamedProbe, ReadinessChecker


async def test_track_counts_inflight_work() -> None:
    lifecycle = Lifecycle()
    assert lifecycle.inflight == 0
    with lifecycle.track():
        with lifecycle.track():
            assert lifecycle.inflight == 2
        assert lifecycle.inflight == 1
    assert lifecycle.inflight == 0


async def test_track_decrements_on_error() -> None:
    lifecycle = Lifecycle()
    with pytest.raises(RuntimeError), lifecycle.track():
        raise RuntimeError
    assert lifecycle.inflight == 0


async def test_wait_idle_returns_when_work_finishes() -> None:
    lifecycle = Lifecycle()

    async def work() -> None:
        with lifecycle.track():
            await asyncio.sleep(0.05)

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    assert lifecycle.inflight == 1
    async with asyncio.timeout(1):
        await lifecycle.wait_idle()
    assert lifecycle.inflight == 0
    await task


async def test_wait_idle_can_be_bounded() -> None:
    lifecycle = Lifecycle()
    with lifecycle.track(), pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await lifecycle.wait_idle()


async def test_draining_flag() -> None:
    lifecycle = Lifecycle()
    assert not lifecycle.draining
    lifecycle.start_draining()
    assert lifecycle.draining


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_probe_failures_and_timeouts_are_reported() -> None:
    async def ok() -> None:
        return None

    async def broken() -> None:
        raise ConnectionError("refused")

    async def hangs() -> None:
        await asyncio.sleep(10)

    checker = ReadinessChecker(
        [NamedProbe("a", ok), NamedProbe("b", broken), NamedProbe("c", hangs)],
        timeout=0.05,
        cache_seconds=0,
    )
    results = await checker.check()
    assert results["a"].ok
    assert not results["b"].ok
    assert results["b"].detail == "ConnectionError: refused"
    assert not results["c"].ok
    assert results["c"].detail == "timed out after 0.05s"


async def test_results_are_cached_briefly() -> None:
    calls = 0

    async def probe() -> None:
        nonlocal calls
        calls += 1

    clock = FakeClock()
    checker = ReadinessChecker([NamedProbe("p", probe)], timeout=1, cache_seconds=1, clock=clock)
    await checker.check()
    await checker.check()
    assert calls == 1
    clock.now = 1.5
    await checker.check()
    assert calls == 2


async def test_concurrent_checks_share_one_round_of_probes() -> None:
    calls = 0

    async def slow_probe() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)

    checker = ReadinessChecker([NamedProbe("p", slow_probe)], timeout=1, cache_seconds=5)
    await asyncio.gather(*(checker.check() for _ in range(10)))
    assert calls == 1
