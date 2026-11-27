"""Usage events and the asynchronous ledger writer (PLAN.md §8, ADR-0006).

Every authenticated request produces one ``UsageEvent``, whatever its outcome. The
request path only calls ``put``, which never blocks: a full queue drops the event and
counts it in ``switchboard_usage_events_dropped_total``. Budgets stay correct
regardless, because Redis counters are updated at settlement, not from the ledger.

``UsageWriter`` batches events (up to ``batch_rows``, or whatever arrived within
``flush_seconds``) and writes them with ``COPY``. If Postgres is down, the batch is
retried with backoff while new events wait in the queue, so a short outage loses
nothing. A hard crash can lose what was queued (at most one flush interval of rows at
steady load); ADR-0006 records that trade-off.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import astuple, dataclass, field, fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol
from uuid import UUID

import asyncpg
import structlog

from switchboard.observability.metrics import (
    USAGE_DROPPED,
    USAGE_QUEUE_DEPTH,
    USAGE_WRITE_ERRORS,
    USAGE_WRITTEN,
)

_log = structlog.get_logger("switchboard.accounting.usage")
_MAX_BACKOFF = 30.0


@dataclass(slots=True)
class UsageEvent:
    """One ``usage_events`` row; field order is the table's column order."""

    id: UUID
    created_at: datetime
    request_id: str
    trace_id: str | None
    key_id: UUID
    team_id: UUID
    endpoint: str  # chat | embeddings
    requested_model: str
    route_rule: str | None
    provider: str | None
    model: str | None
    status: str
    http_status: int
    error_type: str | None
    stream: bool
    cache_status: str = "disabled"
    fallback_used: bool = False
    attempts: str = "[]"  # JSON
    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: Decimal = Decimal(0)
    cost_estimated: bool = False
    saved_usd: Decimal = Decimal(0)
    hedge_waste_usd: Decimal = Decimal(0)
    pricing_version: str | None = None
    ttft_ms: int | None = None
    latency_ms: int = 0
    gateway_overhead_ms: float | None = None
    prompt_name: str | None = None
    prompt_version: int | None = None
    tags: list[str] = field(default_factory=list)
    end_user: str | None = None


COLUMNS = tuple(f.name for f in fields(UsageEvent))


class UsageSink(Protocol):
    def put(self, event: UsageEvent) -> None: ...


class MemoryUsageSink:
    """Keeps events in a list (tests, and runs without a database)."""

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def put(self, event: UsageEvent) -> None:
        self.events.append(event)


class UsageWriter:
    def __init__(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        *,
        queue_size: int = 10_000,
        batch_rows: int = 500,
        flush_seconds: float = 0.25,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._pool = pool
        self._queue: asyncio.Queue[UsageEvent] = asyncio.Queue(maxsize=queue_size)
        self._batch_rows = batch_rows
        self._flush = flush_seconds
        self._clock = clock
        self._partitions_for: str | None = None  # the month partitions were last ensured
        self._task: asyncio.Task[None] | None = None
        self._failing = False
        USAGE_QUEUE_DEPTH.set_function(self._queue.qsize)

    def put(self, event: UsageEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            USAGE_DROPPED.inc()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def aclose(self, timeout: float = 5.0) -> None:  # noqa: ASYNC109 (a bounded drain)
        """Stop and flush what is queued, for at most ``timeout`` seconds."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait({self._task})
        try:
            async with asyncio.timeout(timeout):
                while not self._queue.empty():
                    await self._write_with_retry(self._drain())
        except TimeoutError:
            _log.error("usage.lost_at_shutdown", rows=self._queue.qsize())

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            await asyncio.sleep(self._flush)  # gather what arrives meanwhile
            await self._write_with_retry([first, *self._drain(self._batch_rows - 1)])

    def _drain(self, limit: int | None = None) -> list[UsageEvent]:
        batch: list[UsageEvent] = []
        limit = self._batch_rows if limit is None else limit
        while len(batch) < limit:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _write_with_retry(self, batch: list[UsageEvent]) -> None:
        if not batch:
            return
        delay = 0.5
        while True:
            try:
                await self._ensure_partitions()
                async with self._pool.acquire(timeout=5) as conn:
                    await conn.copy_records_to_table(
                        "usage_events",
                        records=[astuple(e) for e in batch],
                        columns=COLUMNS,
                        timeout=10,
                    )
            except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError) as exc:
                USAGE_WRITE_ERRORS.inc()
                if not self._failing:
                    _log.warning("usage.write_failed", detail=str(exc), rows=len(batch))
                self._failing = True
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAX_BACKOFF)
                continue
            if self._failing:
                _log.info("usage.write_recovered", rows=len(batch))
                self._failing = False
            USAGE_WRITTEN.inc(len(batch))
            return

    async def _ensure_partitions(self) -> None:
        """This month's and next month's partitions, checked once per month."""
        now = self._clock()
        month = now.strftime("%Y%m")
        if self._partitions_for == month:
            return
        next_month = (now.replace(day=1) + _MONTH).replace(day=1)
        with contextlib.suppress(asyncpg.PostgresError):
            async with self._pool.acquire(timeout=5) as conn:
                for day in (now.date(), next_month.date()):
                    await conn.execute("SELECT switchboard_ensure_usage_partition($1)", day)
                self._partitions_for = month


_MONTH = timedelta(days=32)  # from the 1st, always lands in the next month


def ledger_spend(
    pool: asyncpg.Pool[asyncpg.Record],
) -> Callable[[str, UUID, datetime, datetime], Awaitable[Decimal]]:
    """A ``SpendSource`` over ``usage_events``, to rebuild lost budget counters."""

    async def spend(scope: str, scope_id: UUID, start: datetime, end: datetime) -> Decimal:
        column = "key_id" if scope == "key" else "team_id"
        value = await pool.fetchval(
            f"SELECT coalesce(sum(cost_usd), 0) FROM usage_events "  # noqa: S608 (fixed column)
            f"WHERE {column} = $1 AND created_at >= $2 AND created_at < $3",
            scope_id,
            start,
            end,
        )
        return Decimal(value)

    return spend
