"""Dependency checks behind ``/ready``.

Each probe is an async callable that raises on failure. Probes run concurrently,
each with a timeout, and the combined result is cached briefly so frequent
health checks can't turn into a load source for Redis or Postgres.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

Probe = Callable[[], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class NamedProbe:
    name: str
    check: Probe


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    detail: str
    latency_ms: float


class ReadinessChecker:
    def __init__(
        self,
        probes: Sequence[NamedProbe],
        *,
        timeout: float,
        cache_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probes = tuple(probes)
        self._timeout = timeout
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._cached: dict[str, ProbeResult] | None = None
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    async def check(self) -> dict[str, ProbeResult]:
        async with self._lock:  # concurrent /ready calls share one round of probes
            now = self._clock()
            if self._cached is not None and now - self._cached_at < self._cache_seconds:
                return self._cached
            results = await asyncio.gather(*(self._run(p) for p in self._probes))
            self._cached = {p.name: r for p, r in zip(self._probes, results, strict=True)}
            self._cached_at = self._clock()
            return self._cached

    async def _run(self, probe: NamedProbe) -> ProbeResult:
        start = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout):
                await probe.check()
        except TimeoutError:
            return ProbeResult(False, f"timed out after {self._timeout:g}s", _ms_since(start))
        except Exception as exc:  # a probe failure of any kind means "not ready"
            return ProbeResult(False, f"{type(exc).__name__}: {exc}", _ms_since(start))
        return ProbeResult(True, "ok", _ms_since(start))


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
