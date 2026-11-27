"""Where keys live: Postgres in production, memory in tests."""

from collections.abc import Iterable
from typing import Any, Protocol

import asyncpg

from switchboard.auth.context import KeyContext, Limits, combine_allowed


class KeyStore(Protocol):
    async def lookup(self, key_hash: bytes) -> KeyContext | None:
        """The key with this hash, revoked or not; None if there is none."""
        ...


_LOOKUP = """
SELECT k.id, k.name, k.key_prefix, k.status, k.expires_at, k.tags,
       k.rpm_limit, k.tpm_limit, k.monthly_budget_usd, k.soft_budget_usd, k.allowed_models,
       t.id AS team_id, t.name AS team_name,
       t.rpm_limit AS team_rpm, t.tpm_limit AS team_tpm,
       t.monthly_budget_usd AS team_budget, t.soft_budget_usd AS team_soft,
       t.allowed_models AS team_models
FROM api_keys k JOIN teams t ON t.id = k.team_id
WHERE k.key_hash = $1
"""


class PostgresKeyStore:
    def __init__(self, pool: asyncpg.Pool[asyncpg.Record]) -> None:
        self._pool = pool

    async def lookup(self, key_hash: bytes) -> KeyContext | None:
        row = await self._pool.fetchrow(_LOOKUP, key_hash)
        return None if row is None else context_from_row(row)


def context_from_row(row: asyncpg.Record | dict[str, Any]) -> KeyContext:
    return KeyContext(
        key_id=row["id"],
        key_name=row["name"],
        key_prefix=row["key_prefix"],
        team_id=row["team_id"],
        team_name=row["team_name"],
        key_limits=Limits(
            rpm=row["rpm_limit"],
            tpm=row["tpm_limit"],
            hard_usd=row["monthly_budget_usd"],
            soft_usd=row["soft_budget_usd"],
        ),
        team_limits=Limits(
            rpm=row["team_rpm"],
            tpm=row["team_tpm"],
            hard_usd=row["team_budget"],
            soft_usd=row["team_soft"],
        ),
        allowed_models=combine_allowed(row["allowed_models"], row["team_models"]),
        tags=frozenset(row["tags"] or ()),
        expires_at=row["expires_at"],
        revoked=row["status"] != "active",
    )


class MemoryKeyStore:
    """Keys held in a dict, for tests and for running without a database."""

    def __init__(self, keys: Iterable[tuple[bytes, KeyContext]] = ()) -> None:
        self.keys = dict(keys)
        self.lookups = 0

    async def lookup(self, key_hash: bytes) -> KeyContext | None:
        self.lookups += 1
        return self.keys.get(key_hash)


class NoKeyStore:
    """No database: every key is unknown."""

    async def lookup(self, key_hash: bytes) -> KeyContext | None:
        return None
