"""Process lifecycle: readiness, draining and in-flight tracking for graceful shutdown.

On SIGTERM the server calls :meth:`Lifecycle.start_draining` before uvicorn stops
accepting connections. From then on ``/ready`` answers 503, so a load balancer stops
routing new traffic here. Streaming handlers (Phase 1+) wrap their work in
:meth:`Lifecycle.track`, so shutdown can wait (bounded) for them to finish.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager


class Lifecycle:
    def __init__(self) -> None:
        self._draining = False
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def inflight(self) -> int:
        return self._inflight

    def start_draining(self) -> None:
        self._draining = True

    @contextmanager
    def track(self) -> Iterator[None]:
        """Count a unit of in-flight work (typically one streamed response)."""
        self._inflight += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()

    async def wait_idle(self) -> None:
        """Wait until no work is in flight. Bound it with ``asyncio.timeout``."""
        await self._idle.wait()
