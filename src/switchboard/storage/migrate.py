"""A small migration runner: plain SQL files applied in order, once each.

Files are ``migrations/NNNN_description.sql``. ``switchboard migrate`` applies every file
not yet recorded in ``schema_migrations``, all in one transaction under an advisory
lock, so concurrent runs (several replicas starting at once) serialize and a failure
leaves nothing half applied. Each applied file's SHA-256 is recorded; a file changed
after it was applied is an error, because the database no longer matches the repo.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import asyncpg
from asyncpg.pool import PoolConnectionProxy

type Conn = asyncpg.Connection[asyncpg.Record] | PoolConnectionProxy[asyncpg.Record]

_FILE_RE = re.compile(r"^(?P<version>\d{4})_[a-z0-9_]+\.sql$")
_LOCK_ID = 0x5357_4954_4348  # "SWITCH": any constant; only migrators take it

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    text PRIMARY KEY,
  name       text NOT NULL,
  checksum   text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()


def discover(directory: Path) -> list[Migration]:
    if not directory.is_dir():
        raise MigrationError(f"migrations directory not found: {directory}")
    migrations: list[Migration] = []
    for path in sorted(directory.iterdir()):
        if path.suffix != ".sql":
            continue
        match = _FILE_RE.match(path.name)
        if match is None:
            raise MigrationError(f"bad migration file name (want NNNN_name.sql): {path.name}")
        migrations.append(Migration(match["version"], path.name, path.read_text("utf-8")))
    versions = [m.version for m in migrations]
    if len(set(versions)) != len(versions):
        raise MigrationError("two migration files share a version number")
    return migrations


async def migrate(conn: Conn, directory: Path) -> list[str]:
    """Apply pending migrations; return the names applied (empty if up to date)."""
    pending = discover(directory)
    applied_now: list[str] = []
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _LOCK_ID)
        await conn.execute(_BOOTSTRAP)
        rows = await conn.fetch("SELECT version, name, checksum FROM schema_migrations")
        applied = {row["version"]: row for row in rows}
        for migration in pending:
            row = applied.get(migration.version)
            if row is not None:
                if row["checksum"] != migration.checksum:
                    raise MigrationError(
                        f"{migration.name} changed after it was applied; add a new "
                        "migration instead of editing an applied one"
                    )
                continue
            await conn.execute(migration.sql)
            await conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES ($1, $2, $3)",
                migration.version,
                migration.name,
                migration.checksum,
            )
            applied_now.append(migration.name)
    return applied_now


async def pending(conn: Conn, directory: Path) -> list[str]:
    """Names of migrations not yet applied (for readiness and the CLI)."""
    exists = await conn.fetchval("SELECT to_regclass('schema_migrations') IS NOT NULL")
    applied: set[str] = set()
    if exists:
        rows = await conn.fetch("SELECT version FROM schema_migrations")
        applied = {row["version"] for row in rows}
    return [m.name for m in discover(directory) if m.version not in applied]
