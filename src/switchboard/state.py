"""Per-process state shared by request handlers, created in the app lifespan."""

from dataclasses import dataclass
from typing import cast

import asyncpg
from fastapi import Request
from redis.asyncio import Redis

from switchboard.accounting.usage import UsageSink
from switchboard.auth.resolver import KeyResolver
from switchboard.lifecycle import Lifecycle
from switchboard.limits.limiter import Limiter
from switchboard.readiness import ReadinessChecker
from switchboard.reliability.executor import Executor
from switchboard.reload import ConfigManager
from switchboard.settings import Settings


@dataclass(slots=True)
class AppState:
    settings: Settings
    lifecycle: Lifecycle
    readiness: ReadinessChecker
    config: ConfigManager  # the active routing snapshot; swapped on reload
    executor: Executor
    keys: KeyResolver
    limiter: Limiter | None  # None without Redis: no rate limits or budgets
    usage: UsageSink
    redis: Redis | None = None
    pg_pool: asyncpg.Pool[asyncpg.Record] | None = None


def get_state(request: Request) -> AppState:
    return cast(AppState, request.app.state.switchboard)
