"""Per-process state shared by request handlers, created in the app lifespan."""

from dataclasses import dataclass
from typing import cast

import asyncpg
from fastapi import Request
from redis.asyncio import Redis

from switchboard.lifecycle import Lifecycle
from switchboard.providers.base import ProviderAdapter
from switchboard.readiness import ReadinessChecker
from switchboard.routing import LoadedConfig
from switchboard.routing.table import RoutingTable
from switchboard.settings import Settings


@dataclass(slots=True)
class AppState:
    settings: Settings
    routing: LoadedConfig
    lifecycle: Lifecycle
    readiness: ReadinessChecker
    routing_table: RoutingTable
    adapters: dict[str, ProviderAdapter]
    redis: Redis | None = None
    pg_pool: asyncpg.Pool[asyncpg.Record] | None = None


def get_state(request: Request) -> AppState:
    return cast(AppState, request.app.state.switchboard)
