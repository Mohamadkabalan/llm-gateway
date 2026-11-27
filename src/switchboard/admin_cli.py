"""The database commands of the CLI: migrations, teams and keys.

``switchboard migrate | create-team | create-key | revoke-key | ensure-key``. They read
``SWITCHBOARD_DATABASE_URL`` and ``SWITCHBOARD_KEY_PEPPER`` like the gateway, and
``SWITCHBOARD_REDIS_URL`` (optional) to push revocations to running replicas.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

import asyncpg

from switchboard.auth.manage import (
    LimitSettings,
    ManageError,
    create_key,
    create_team,
    ensure_key,
    revoke_key,
)
from switchboard.settings import Settings
from switchboard.storage.clients import create_redis
from switchboard.storage.migrate import MigrationError, migrate

EXIT_OK = 0
EXIT_FAILED = 3

COMMANDS = ("migrate", "create-team", "create-key", "revoke-key", "ensure-key")


def add_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    commands.add_parser("migrate", help="apply pending database migrations")

    team = commands.add_parser("create-team", help="create a team")
    team.add_argument("name")
    _limit_flags(team)

    key = commands.add_parser("create-key", help="create an API key (the secret is shown once)")
    key.add_argument("--team", required=True)
    key.add_argument("--name", required=True)
    key.add_argument("--tags", type=_csv, help="comma-separated tags for routing rules")
    key.add_argument("--expires-in-days", type=int, help="the key stops working after this")
    _limit_flags(key)

    revoke = commands.add_parser("revoke-key", help="revoke a key everywhere within a second")
    which = revoke.add_mutually_exclusive_group(required=True)
    which.add_argument("--id", type=UUID)
    which.add_argument("--prefix", help="the key's display prefix, e.g. sb_live_4f2a")

    ensure = commands.add_parser(
        "ensure-key",
        help="make sure a key with the secret in an environment variable exists "
        "(idempotent; for local stacks and benchmarks)",
    )
    ensure.add_argument("--team", required=True)
    ensure.add_argument("--name", required=True)
    ensure.add_argument("--secret-env", required=True, metavar="VAR")
    _limit_flags(ensure)
    _limit_flags(ensure, prefix="team-")


def run(args: argparse.Namespace, settings: Settings) -> int:
    if settings.database_url is None:
        print("SWITCHBOARD_DATABASE_URL is not set.", file=sys.stderr)
        return EXIT_FAILED
    command: Callable[[asyncpg.Connection[asyncpg.Record]], Awaitable[None]] = {
        "migrate": lambda conn: _migrate(conn, settings),
        "create-team": lambda conn: _create_team(conn, args),
        "create-key": lambda conn: _create_key(conn, args, settings),
        "revoke-key": lambda conn: _revoke_key(conn, args, settings),
        "ensure-key": lambda conn: _ensure_key(conn, args, settings),
    }[args.command]
    try:
        asyncio.run(_with_connection(str(settings.database_url), command))
    except (ManageError, MigrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except (OSError, asyncpg.PostgresError) as exc:
        print(f"error: database: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


async def _with_connection(
    dsn: str, command: Callable[[asyncpg.Connection[asyncpg.Record]], Awaitable[None]]
) -> None:
    conn = await asyncpg.connect(dsn, timeout=10)
    try:
        await command(conn)
    finally:
        await conn.close()


async def _migrate(conn: asyncpg.Connection[asyncpg.Record], settings: Settings) -> None:
    applied = await migrate(conn, settings.migrations_dir)
    if applied:
        for name in applied:
            print(f"applied {name}")
    else:
        print("database is up to date")


async def _create_team(conn: asyncpg.Connection[asyncpg.Record], args: argparse.Namespace) -> None:
    team_id = await create_team(conn, args.name, _limits(args))
    print(f"created team {args.name!r} ({team_id})")


async def _create_key(
    conn: asyncpg.Connection[asyncpg.Record], args: argparse.Namespace, settings: Settings
) -> None:
    expires = None
    if args.expires_in_days is not None:
        expires = datetime.now(UTC) + timedelta(days=args.expires_in_days)
    created = await create_key(
        conn,
        _pepper(settings),
        team=args.team,
        name=args.name,
        limits=_limits(args),
        tags=args.tags,
        expires_at=expires,
    )
    print(f"created key {created.name!r} for team {args.team!r} ({created.id}, {created.prefix}…)")
    print("The secret is shown once; store it now:")
    print(created.secret)


async def _revoke_key(
    conn: asyncpg.Connection[asyncpg.Record], args: argparse.Namespace, settings: Settings
) -> None:
    redis = create_redis(str(settings.redis_url)) if settings.redis_url else None
    try:
        key_id, pushed = await revoke_key(conn, redis, key_id=args.id, prefix=args.prefix)
    finally:
        if redis is not None:
            await redis.aclose()
    when = "now" if pushed else f"within {settings.key_cache_ttl_seconds:g}s (no Redis to push)"
    print(f"revoked key {key_id}; replicas stop accepting it {when}")


async def _ensure_key(
    conn: asyncpg.Connection[asyncpg.Record], args: argparse.Namespace, settings: Settings
) -> None:
    secret = os.environ.get(args.secret_env)
    if not secret:
        raise ManageError(f"environment variable {args.secret_env} is empty")
    key = await ensure_key(
        conn,
        _pepper(settings),
        team=args.team,
        name=args.name,
        secret=secret,
        limits=_limits(args),
        team_limits=_limits(args, prefix="team_"),
    )
    print(f"key {key.name!r} ({key.prefix}…) is ready in team {args.team!r}")


def _pepper(settings: Settings) -> bytes:
    if settings.key_pepper is None:
        raise ManageError("SWITCHBOARD_KEY_PEPPER is not set")
    return settings.key_pepper.get_secret_value().encode()


def _limit_flags(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    who = "the team" if prefix else "this"
    parser.add_argument(f"--{prefix}budget", type=_usd, help=f"monthly hard limit for {who}, USD")
    parser.add_argument(f"--{prefix}soft-budget", type=_usd, help="monthly alert threshold, USD")
    parser.add_argument(f"--{prefix}rpm", type=int, help="requests per minute")
    parser.add_argument(f"--{prefix}tpm", type=int, help="tokens per minute")
    parser.add_argument(f"--{prefix}models", type=_csv, help="allowed aliases and models")


def _limits(args: argparse.Namespace, prefix: str = "") -> LimitSettings:
    def get(name: str) -> Any:
        return getattr(args, f"{prefix}{name}", None)

    return LimitSettings(
        monthly_budget_usd=get("budget"),
        soft_budget_usd=get("soft_budget"),
        rpm_limit=get("rpm"),
        tpm_limit=get("tpm"),
        allowed_models=get("models"),
    )


def _usd(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not an amount: {value!r}") from None
    if amount < 0:
        raise argparse.ArgumentTypeError("amounts cannot be negative")
    return amount


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]
