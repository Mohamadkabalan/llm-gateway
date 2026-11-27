"""The idle timeout: the longest gap allowed between upstream reads after the commit point.

Checked lazily, so streaming stays cheap: per chunk the hot path only stores timestamps
(adapters set ``CallTimings.last_read`` on every network read; the responder marks when
it is busy sending to the client). One timer per idle period checks the gap and, if it
is too long, expires the surrounding ``asyncio.timeout``.

Time spent writing to a slow client is not counted: while the gateway is not reading,
the upstream cannot be blamed for silence.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

from switchboard.providers.base import CallTimings
from switchboard.providers.errors import ErrorKind, ProviderError


class IdleWatchdog:
    def __init__(self, timeout: asyncio.Timeout, idle: float, timings: CallTimings) -> None:
        self._loop = asyncio.get_running_loop()
        self._timeout = timeout
        self._idle = idle
        self._timings = timings
        self.sending = False
        self.ready_at = time.perf_counter()
        self.fired = False
        self._handle = self._loop.call_later(idle, self._check)

    def sent(self) -> None:
        """The responder finished a write and is waiting on upstream again."""
        self.sending = False
        self.ready_at = time.perf_counter()

    def cancel(self) -> None:
        self._handle.cancel()

    def _check(self) -> None:
        if self.sending:
            self._handle = self._loop.call_later(self._idle, self._check)
            return
        last = max(self._timings.last_read, self.ready_at)
        remaining = last + self._idle - time.perf_counter()
        if remaining > 0:
            self._handle = self._loop.call_later(remaining, self._check)
            return
        self.fired = True
        self._timeout.reschedule(self._loop.time())


@contextlib.asynccontextmanager
async def idle_guard(idle: float, timings: CallTimings) -> AsyncIterator[IdleWatchdog]:
    """Run the body with an idle watchdog; a silent upstream raises a TIMEOUT error."""
    watchdog: IdleWatchdog | None = None
    try:
        async with asyncio.timeout(None) as timeout:
            watchdog = IdleWatchdog(timeout, idle, timings)
            try:
                yield watchdog
            finally:
                watchdog.cancel()
    except TimeoutError:
        if watchdog is not None and watchdog.fired:
            message = f"no data from upstream for {idle:g}s"
            raise ProviderError(ErrorKind.TIMEOUT, message) from None
        raise
