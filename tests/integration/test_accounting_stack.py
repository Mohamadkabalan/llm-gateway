"""Phase 4 acceptance against the compose stack (`make up`, then `make test-integration`).

Each test creates its own team and key directly in Postgres (with the stack's pepper,
from .env) and removes them afterwards. Usage rows are left in place: they are the
ledger, and nothing references them.
"""

import asyncio
import os
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from decimal import Decimal
from typing import Any

import asyncpg
import httpx
import openai
import pytest
from redis.asyncio import Redis

from switchboard.auth.manage import LimitSettings, create_key, create_team, revoke_key
from switchboard.storage.migrate import migrate, pending
from tests.support import REPO_ROOT

pytestmark = pytest.mark.integration

SONNET_IN, SONNET_OUT = Decimal(2), Decimal(10)  # fake models are priced like Sonnet 5.5
COMPOSE = ["docker", "compose", "-f", "deploy/compose.yaml", "--env-file", ".env"]


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set (run via `make test-integration`)")
    return value


@pytest.fixture
async def pg() -> AsyncIterator[asyncpg.Connection[asyncpg.Record]]:
    conn = await asyncpg.connect(env("SB_DATABASE_URL"))
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = Redis.from_url(env("SB_REDIS_URL"))
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def gateway() -> str:
    return os.environ.get("SB_GATEWAY_URL", "http://127.0.0.1:7080")


type NewKey = Callable[..., Awaitable[tuple[str, uuid.UUID]]]


@pytest.fixture
async def new_key(pg: asyncpg.Connection[asyncpg.Record]) -> AsyncIterator[NewKey]:
    """Create a team and one key in it; returns ``(secret, key id)``."""
    pepper = env("SWITCHBOARD_KEY_PEPPER").encode()
    teams: list[uuid.UUID] = []

    async def make(**limits: Any) -> tuple[str, uuid.UUID]:
        name = f"it-{uuid.uuid4().hex[:12]}"
        teams.append(await create_team(pg, name))
        created = await create_key(
            pg, pepper, team=name, name="key", limits=LimitSettings(**limits)
        )
        assert created.secret is not None
        return created.secret, created.id

    yield make
    for team_id in teams:
        await pg.execute("DELETE FROM api_keys WHERE team_id = $1", team_id)
        await pg.execute(
            "DELETE FROM budget_alerts WHERE scope = 'team' AND scope_id = $1", team_id
        )
        await pg.execute("DELETE FROM teams WHERE id = $1", team_id)


def body(**extra: Any) -> dict[str, Any]:
    return {"model": "fake", "messages": [{"role": "user", "content": "hello"}], **extra}


async def post(gateway: str, secret: str, **headers: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as http:
        return await http.post(
            f"{gateway}/v1/chat/completions",
            json=body(),
            headers={"authorization": f"Bearer {secret}", **headers},
        )


async def rows(
    pg: asyncpg.Connection[asyncpg.Record], key_id: uuid.UUID, *, at_least: int, within: float = 10
) -> list[asyncpg.Record]:
    """The key's usage rows, once at least ``at_least`` have been written."""
    deadline = time.monotonic() + within
    while True:
        found = await pg.fetch(
            "SELECT * FROM usage_events WHERE key_id = $1 ORDER BY created_at", key_id
        )
        if len(found) >= at_least or time.monotonic() > deadline:
            return found
        await asyncio.sleep(0.1)


# --- migrations and the ledger ---------------------------------------------------------


async def test_migrations_are_applied_and_rerunning_is_a_no_op(
    pg: asyncpg.Connection[asyncpg.Record],
) -> None:
    assert await pending(pg, REPO_ROOT / "migrations") == []
    assert await migrate(pg, REPO_ROOT / "migrations") == []


async def test_one_usage_row_per_request_with_tokens_and_cost(
    gateway: str, new_key: NewKey, pg: asyncpg.Connection[asyncpg.Record]
) -> None:
    secret, key_id = await new_key()
    request_id = f"it-{uuid.uuid4().hex}"
    response = await post(gateway, secret, **{"x-request-id": request_id})
    assert response.status_code == 200
    usage = response.json()["usage"]
    (row,) = await rows(pg, key_id, at_least=1)
    assert row["request_id"] == request_id
    assert (row["status"], row["http_status"], row["model"]) == ("ok", 200, "fake/fake-chat")
    assert row["input_tokens"] == usage["prompt_tokens"]
    assert row["output_tokens"] == usage["completion_tokens"]
    expected = (
        usage["prompt_tokens"] * SONNET_IN + usage["completion_tokens"] * SONNET_OUT
    ) / 10**6
    assert row["cost_usd"] == expected
    assert row["pricing_version"] == "2026-11-27"


# --- budgets ---------------------------------------------------------------------------


async def test_hard_budget_is_429_insufficient_quota_and_the_sdk_does_not_retry(
    gateway: str, new_key: NewKey, pg: asyncpg.Connection[asyncpg.Record]
) -> None:
    # The admission estimate (prompt plus 64 output tokens at Sonnet prices) is ~$0.0007.
    secret, key_id = await new_key(monthly_budget_usd=Decimal("0.0005"))
    client = openai.AsyncOpenAI(base_url=f"{gateway}/v1", api_key=secret, max_retries=3)
    try:
        with pytest.raises(openai.RateLimitError) as caught:
            await client.chat.completions.create(**body())
    finally:
        await client.close()
    assert caught.value.code == "insufficient_quota"
    assert caught.value.response.headers["x-should-retry"] == "false"
    found = await rows(pg, key_id, at_least=1)
    await asyncio.sleep(1)  # room for retries to show up, if the SDK made any
    found = await rows(pg, key_id, at_least=1)
    assert [r["status"] for r in found] == ["budget_blocked"]


async def test_soft_alert_fires_exactly_once(
    gateway: str, new_key: NewKey, pg: asyncpg.Connection[asyncpg.Record]
) -> None:
    secret, key_id = await new_key(soft_budget_usd=Decimal("0.00001"))
    for _ in range(3):
        assert (await post(gateway, secret)).status_code == 200
    fake = os.environ.get("SB_FAKE_URL", "http://127.0.0.1:7800")
    deadline = time.monotonic() + 10
    alerts: list[asyncpg.Record] = []
    while time.monotonic() < deadline and not alerts:
        alerts = await pg.fetch("SELECT * FROM budget_alerts WHERE scope_id = $1", key_id)
        await asyncio.sleep(0.1)
    await asyncio.sleep(1)  # room for a duplicate, if there were one
    alerts = await pg.fetch("SELECT * FROM budget_alerts WHERE scope_id = $1", key_id)
    assert len(alerts) == 1
    assert alerts[0]["scope"] == "key"
    async with httpx.AsyncClient() as http:
        webhooks = (await http.get(f"{fake}/_control/stats")).json()["webhooks"]
    mine = [w for w in webhooks if w.get("switchboard", {}).get("scope_id") == str(key_id)]
    assert len(mine) == 1
    await pg.execute("DELETE FROM budget_alerts WHERE scope_id = $1", key_id)


# --- rate limits -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("limits", "kind"),
    [({"rpm_limit": 2}, "requests"), ({"tpm_limit": 50}, "tokens")],
)
async def test_rate_limits_are_429_with_retry_after_and_ratelimit_headers(
    gateway: str, new_key: NewKey, limits: dict[str, int], kind: str
) -> None:
    secret, _ = await new_key(**limits)
    statuses = [(await post(gateway, secret)) for _ in range(3)]
    refused = statuses[-1]
    assert refused.status_code == 429
    assert refused.json()["error"]["type"] == kind
    assert int(refused.headers["retry-after"]) >= 1
    assert f"x-ratelimit-limit-{kind}" in refused.headers
    assert f"x-ratelimit-remaining-{kind}" in refused.headers
    assert f"x-ratelimit-reset-{kind}" in refused.headers


# --- resilience and revocation ---------------------------------------------------------


async def test_requests_succeed_while_postgres_is_paused_and_rows_land_after(
    gateway: str, new_key: NewKey, pg: asyncpg.Connection[asyncpg.Record]
) -> None:
    secret, key_id = await new_key()
    assert (await post(gateway, secret)).status_code == 200  # the key is now cached
    await rows(pg, key_id, at_least=1)
    await compose("pause", "postgres")
    try:
        during = [await post(gateway, secret) for _ in range(3)]
    finally:
        await compose("unpause", "postgres")
    assert [r.status_code for r in during] == [200, 200, 200]
    found = await rows(pg, key_id, at_least=4, within=60)  # the writer backs off to 30 s
    assert len(found) == 4


async def compose(*args: str) -> None:
    await asyncio.to_thread(subprocess.run, [*COMPOSE, *args], cwd=REPO_ROOT, check=True)


async def test_a_revoked_key_is_rejected_within_a_second(
    gateway: str, new_key: NewKey, pg: asyncpg.Connection[asyncpg.Record], redis: Redis
) -> None:
    secret, key_id = await new_key()
    assert (await post(gateway, secret)).status_code == 200  # cached for 30 s
    _, pushed = await revoke_key(pg, redis, key_id=key_id)
    assert pushed
    started = time.monotonic()
    while (await post(gateway, secret)).status_code != 401:
        assert time.monotonic() - started < 1.0, "the revoked key was still accepted"
        await asyncio.sleep(0.02)
    assert time.monotonic() - started < 1.0
