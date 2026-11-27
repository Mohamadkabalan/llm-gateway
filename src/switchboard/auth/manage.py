"""Creating teams and keys, and revoking keys.

Used by the CLI today and by the admin API in Phase 6. A secret is returned exactly
once, from ``create_key``; only its HMAC is stored.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg
from redis.asyncio import Redis

from switchboard.auth.keys import display_prefix, generate_secret, hash_secret, plausible
from switchboard.auth.resolver import publish_invalidation


class ManageError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class LimitSettings:
    monthly_budget_usd: Decimal | None = None
    soft_budget_usd: Decimal | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    allowed_models: list[str] | None = None

    def values(self) -> tuple[Any, ...]:
        return (
            self.monthly_budget_usd,
            self.soft_budget_usd,
            self.rpm_limit,
            self.tpm_limit,
            self.allowed_models,
        )


@dataclass(frozen=True, slots=True)
class CreatedKey:
    id: UUID
    team_id: UUID
    name: str
    prefix: str
    secret: str | None  # None when an existing key was kept (ensure_key)


type Conn = asyncpg.Connection[asyncpg.Record] | asyncpg.Pool[asyncpg.Record]


async def create_team(conn: Conn, name: str, limits: LimitSettings = LimitSettings()) -> UUID:
    team_id = uuid.uuid7()
    try:
        await conn.execute(
            "INSERT INTO teams (id, name, monthly_budget_usd, soft_budget_usd, rpm_limit, "
            "tpm_limit, allowed_models) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            team_id,
            name,
            *limits.values(),
        )
    except asyncpg.UniqueViolationError:
        raise ManageError(f"a team named {name!r} already exists") from None
    return team_id


async def ensure_team(conn: Conn, name: str, limits: LimitSettings = LimitSettings()) -> UUID:
    """The team's id, creating it with ``limits`` if it does not exist (limits of an
    existing team are left alone)."""
    existing: UUID | None = await conn.fetchval("SELECT id FROM teams WHERE name = $1", name)
    if existing is not None:
        return existing
    try:
        return await create_team(conn, name, limits)
    except ManageError:  # created concurrently
        found: UUID = await conn.fetchval("SELECT id FROM teams WHERE name = $1", name)
        return found


async def team_id_for(conn: Conn, name: str) -> UUID:
    team_id: UUID | None = await conn.fetchval("SELECT id FROM teams WHERE name = $1", name)
    if team_id is None:
        raise ManageError(f"no team named {name!r}")
    return team_id


async def create_key(
    conn: Conn,
    pepper: bytes,
    *,
    team: str,
    name: str,
    limits: LimitSettings = LimitSettings(),
    tags: list[str] | None = None,
    expires_at: datetime | None = None,
    secret: str | None = None,
) -> CreatedKey:
    """Create a key. ``secret`` is for automation (``ensure-key``); normally one is
    generated, and this is the only time it is ever available."""
    team_id = await team_id_for(conn, team)
    secret = secret or generate_secret()
    if not plausible(secret):
        raise ManageError("a key must start with 'sb_' and be 20 to 200 characters long")
    key_id = uuid.uuid7()
    try:
        await conn.execute(
            "INSERT INTO api_keys (id, team_id, name, key_prefix, key_hash, monthly_budget_usd, "
            "soft_budget_usd, rpm_limit, tpm_limit, allowed_models, tags, expires_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
            key_id,
            team_id,
            name,
            display_prefix(secret),
            hash_secret(pepper, secret),
            *limits.values(),
            tags or [],
            expires_at,
        )
    except asyncpg.UniqueViolationError:
        raise ManageError(f"team {team!r} already has a key named {name!r}") from None
    return CreatedKey(key_id, team_id, name, display_prefix(secret), secret)


async def ensure_key(
    conn: Conn,
    pepper: bytes,
    *,
    team: str,
    name: str,
    secret: str,
    limits: LimitSettings = LimitSettings(),
    team_limits: LimitSettings = LimitSettings(),
) -> CreatedKey:
    """Make sure a key with exactly this secret exists (idempotent, for local stacks
    and benchmarks whose secret lives in their environment)."""
    key_hash = hash_secret(pepper, secret)
    row = await conn.fetchrow(
        "SELECT k.id, k.team_id, k.name, k.key_prefix, k.status FROM api_keys k "
        "WHERE k.key_hash = $1",
        key_hash,
    )
    if row is not None:
        if row["status"] != "active":
            raise ManageError(f"the key {row['key_prefix']}… exists but is revoked")
        return CreatedKey(row["id"], row["team_id"], row["name"], row["key_prefix"], None)
    await ensure_team(conn, team, team_limits)
    created = await create_key(conn, pepper, team=team, name=name, limits=limits, secret=secret)
    return CreatedKey(created.id, created.team_id, created.name, created.prefix, None)


async def revoke_key(
    conn: Conn, redis: Redis | None, *, key_id: UUID | None = None, prefix: str | None = None
) -> tuple[UUID, bool]:
    """Revoke a key by id or display prefix. Returns ``(key id, pushed)``, where
    ``pushed`` says whether replicas were told at once (else within the cache TTL)."""
    if (key_id is None) == (prefix is None):
        raise ManageError("give exactly one of a key id or a key prefix")
    where, value = ("id = $1", key_id) if key_id is not None else ("key_prefix = $1", prefix)
    rows = await conn.fetch(f"SELECT id, key_hash FROM api_keys WHERE {where}", value)  # noqa: S608 (fixed column names)
    if not rows:
        raise ManageError("no such key")
    if len(rows) > 1:
        raise ManageError("that prefix matches more than one key; use the key id")
    row = rows[0]
    await conn.execute(
        "UPDATE api_keys SET status = 'revoked', revoked_at = coalesce(revoked_at, now()) "
        "WHERE id = $1",
        row["id"],
    )
    pushed = await publish_invalidation(redis, key_hash=bytes(row["key_hash"]))
    return row["id"], pushed
