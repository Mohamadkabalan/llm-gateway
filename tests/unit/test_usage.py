"""The usage writer (batching, retry, drops, partitions) and the ledger's schema."""

import asyncio
import contextlib
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from prometheus_client import REGISTRY

from switchboard.accounting.usage import COLUMNS, UsageEvent, UsageWriter
from switchboard.storage.migrate import MigrationError, discover
from tests.support import REPO_ROOT

MIGRATIONS = REPO_ROOT / "migrations"


def migration_columns() -> list[str]:
    sql = (MIGRATIONS / "0001_keys_and_usage.sql").read_text()
    block = sql.split("CREATE TABLE usage_events (", 1)[1].split(") PARTITION BY", 1)[0]
    names = [re.match(r"\s*([a-z_]+)\s", line) for line in block.splitlines()]
    return [m.group(1) for m in names if m and m.group(1) != "PRIMARY"]


def test_event_fields_are_the_tables_columns_in_order() -> None:
    assert list(COLUMNS) == migration_columns()


def test_migrations_are_well_formed() -> None:
    names = [m.name for m in discover(MIGRATIONS)]
    assert names[0] == "0001_keys_and_usage.sql"


def test_bad_migration_names_are_rejected(tmp_path: Any) -> None:
    (tmp_path / "1_x.sql").write_text("SELECT 1;")
    with pytest.raises(MigrationError, match="bad migration file name"):
        discover(tmp_path)


def event(n: int = 0) -> UsageEvent:
    return UsageEvent(
        id=uuid.uuid7(),
        created_at=datetime(2026, 11, 27, tzinfo=UTC),
        request_id=f"req-{n}",
        trace_id=None,
        key_id=uuid.uuid4(),
        team_id=uuid.uuid4(),
        endpoint="chat",
        requested_model="fake",
        route_rule=None,
        provider="fake",
        model="fake/fake-chat",
        status="ok",
        http_status=200,
        error_type=None,
        stream=True,
    )


class FakeConn:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool

    async def copy_records_to_table(
        self,
        table: str,
        *,
        records: list[tuple[Any, ...]],
        columns: tuple[str, ...],
        timeout: float,  # noqa: ASYNC109 (mirrors asyncpg's signature)
    ) -> None:
        if self.pool.failures:
            self.pool.failures -= 1
            raise asyncpg.PostgresConnectionError("connection refused")
        assert table == "usage_events"
        assert columns == COLUMNS
        self.pool.batches.append(records)

    async def execute(self, query: str, *args: Any) -> str:
        self.pool.partitions.append(args[0])
        return "SELECT 1"


class FakePool:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.batches: list[list[tuple[Any, ...]]] = []
        self.partitions: list[Any] = []

    @contextlib.asynccontextmanager
    async def acquire(self, timeout: float | None = None) -> AsyncIterator[FakeConn]:  # noqa: ASYNC109
        yield FakeConn(self)


def writer(pool: FakePool, **kwargs: Any) -> UsageWriter:
    kwargs.setdefault("flush_seconds", 0.01)
    return UsageWriter(pool, **kwargs)  # type: ignore[arg-type]


async def wait_for(condition: Any, within: float = 3.0) -> None:
    async with asyncio.timeout(within):
        while not condition():  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_events_are_written_in_batches() -> None:
    pool = FakePool()
    w = writer(pool, batch_rows=500)
    w.start()
    for n in range(1200):
        w.put(event(n))
    await wait_for(lambda: sum(len(b) for b in pool.batches) == 1200)
    await w.aclose()
    assert max(len(b) for b in pool.batches) <= 500
    assert len(pool.batches) <= 4  # 1,200 rows in batches of at most 500
    assert [row[2] for row in pool.batches[0][:2]] == ["req-0", "req-1"]  # request_id, in order


async def test_a_failing_database_is_retried_until_it_recovers() -> None:
    pool = FakePool(failures=2)
    w = writer(pool)
    w.start()
    w.put(event())
    await wait_for(lambda: len(pool.batches) == 1, within=5)
    await w.aclose()
    errors = REGISTRY.get_sample_value("switchboard_usage_write_errors_total") or 0
    assert errors >= 2


async def test_a_full_queue_drops_and_counts() -> None:
    before = REGISTRY.get_sample_value("switchboard_usage_events_dropped_total") or 0
    w = writer(FakePool(), queue_size=2)  # not started: nothing drains the queue
    for n in range(5):
        w.put(event(n))
    after = REGISTRY.get_sample_value("switchboard_usage_events_dropped_total") or 0
    assert after - before == 3


async def test_shutdown_flushes_what_is_queued() -> None:
    pool = FakePool()
    w = writer(pool)  # never started
    for n in range(3):
        w.put(event(n))
    await w.aclose()
    assert sum(len(b) for b in pool.batches) == 3


async def test_partitions_are_ensured_once_per_month() -> None:
    pool = FakePool()
    now = datetime(2026, 12, 15, tzinfo=UTC)
    w = writer(pool, clock=lambda: now)
    w.start()
    for n in range(3):
        w.put(event(n))
        await wait_for(lambda n=n: sum(len(b) for b in pool.batches) == n + 1)
    await w.aclose()
    assert [d.isoformat() for d in pool.partitions] == ["2026-12-15", "2027-01-01"]
