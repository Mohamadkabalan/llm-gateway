"""Connection factories for Redis and PostgreSQL.

Both are created lazily: no connection is opened at startup, so the process can come
up before its dependencies and report "not ready" until they're reachable, instead of
crash-looping.
"""

import asyncpg
from redis.asyncio import Redis


def create_redis(url: str) -> Redis:
    return Redis.from_url(
        url,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        health_check_interval=30,
        decode_responses=False,
    )


async def create_pg_pool(dsn: str) -> asyncpg.Pool[asyncpg.Record]:
    return await asyncpg.create_pool(
        dsn,
        min_size=0,
        max_size=10,
        command_timeout=5.0,
        timeout=2.0,  # connection establishment
    )


async def ping_redis(client: Redis) -> None:
    await client.ping()


async def ping_postgres(pool: asyncpg.Pool[asyncpg.Record]) -> None:
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
