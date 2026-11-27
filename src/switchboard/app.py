"""Application factory for the gateway data plane."""

import asyncio
import contextlib
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from wsgiref.simple_server import WSGIServer

import asyncpg
import structlog
from fastapi import FastAPI
from prometheus_client import start_http_server
from redis.asyncio import Redis
from starlette.types import ASGIApp

from switchboard import __version__
from switchboard.accounting.usage import (
    MemoryUsageSink,
    UsageSink,
    UsageWriter,
    ledger_spend,
)
from switchboard.api import chat, embeddings, health, models
from switchboard.api.errors import install_error_handlers
from switchboard.api.middleware import RequestContextMiddleware
from switchboard.auth.resolver import KeyResolver
from switchboard.auth.store import KeyStore, NoKeyStore, PostgresKeyStore
from switchboard.lifecycle import Lifecycle
from switchboard.limits.alerts import Alerts
from switchboard.limits.limiter import Limiter
from switchboard.observability.logging import configure_logging
from switchboard.providers.thinking import MemoryThinkingSidecar, RedisThinkingSidecar
from switchboard.readiness import NamedProbe, ReadinessChecker
from switchboard.reliability.breaker import Breakers
from switchboard.reliability.executor import Executor
from switchboard.reload import ConfigManager
from switchboard.routing import ConfigError, LoadedConfig, load_routing_config
from switchboard.settings import Settings
from switchboard.state import AppState
from switchboard.storage.clients import create_pg_pool, create_redis, ping_postgres, ping_redis
from switchboard.storage.migrate import pending

_log = structlog.get_logger("switchboard.app")

_POOL_CLOSE_TIMEOUT = 5.0
_DRAIN_TIMEOUT_AFTER_SERVER = 5.0


@dataclass(frozen=True, slots=True)
class Overrides:
    """Dependencies a test (or an embedding application) supplies instead of the ones
    built from settings."""

    key_store: KeyStore | None = None
    usage_sink: UsageSink | None = None
    redis: Redis | None = None


def build_api(
    settings: Settings,
    routing: LoadedConfig,
    *,
    lifecycle: Lifecycle | None = None,
    config_path: Path | None = None,
    overrides: Overrides | None = None,
) -> FastAPI:
    """Build the FastAPI app. Most callers want :func:`create_app` instead.

    ``config_path`` is the file ``routing`` was loaded from; with it, the config
    reloads on SIGHUP and, unless ``settings.config_watch_seconds`` is 0, when the
    file changes.
    """
    lifecycle = lifecycle or Lifecycle()
    given = overrides or Overrides()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis = given.redis
        if redis is None and settings.redis_url:
            redis = create_redis(str(settings.redis_url))
        pg_pool = (
            await create_pg_pool(str(settings.database_url)) if settings.database_url else None
        )
        probes = _probes(settings, redis, pg_pool)

        sidecar = RedisThinkingSidecar(redis) if redis is not None else MemoryThinkingSidecar()
        breakers = Breakers(redis)
        config = ConfigManager(routing, path=config_path, sidecar=sidecar, breakers=breakers)
        config.start(watch_seconds=settings.config_watch_seconds)

        services = _Services.start(settings, given, redis=redis, pg_pool=pg_pool)
        app.state.switchboard = AppState(
            settings=settings,
            lifecycle=lifecycle,
            readiness=ReadinessChecker(
                probes,
                timeout=settings.ready_check_timeout_seconds,
                cache_seconds=settings.ready_cache_seconds,
            ),
            config=config,
            executor=Executor(breakers),
            keys=services.keys,
            limiter=services.limiter,
            usage=services.usage,
            redis=redis,
            pg_pool=pg_pool,
        )
        metrics_server = _start_metrics_server(settings)
        _log_startup(settings, routing, config, given, redis=redis, pg_pool=pg_pool)
        try:
            yield
        finally:
            # uvicorn has already stopped accepting connections and waited up to the grace
            # period for open ones; this catches work that outlived its connection.
            _log.info("gateway.stopping", inflight=lifecycle.inflight)
            drained = await _wait_idle(lifecycle, _DRAIN_TIMEOUT_AFTER_SERVER)
            await services.aclose()
            await config.aclose()
            await breakers.aclose()
            if metrics_server is not None:
                await asyncio.to_thread(metrics_server.shutdown)
                metrics_server.server_close()
            if redis is not None and given.redis is None:
                await redis.aclose()
            if pg_pool is not None:
                try:
                    await asyncio.wait_for(pg_pool.close(), _POOL_CLOSE_TIMEOUT)
                except TimeoutError:
                    pg_pool.terminate()
            _log.info("gateway.stopped", drained=drained)

    api = FastAPI(
        title="Switchboard",
        version=__version__,
        lifespan=lifespan,
        # The data plane is OpenAI-compatible; it publishes no docs of its own.
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    install_error_handlers(api)
    api.include_router(health.router)
    api.include_router(chat.router)
    api.include_router(embeddings.router)
    api.include_router(models.router)
    return api


def create_app(
    settings: Settings,
    routing: LoadedConfig,
    *,
    lifecycle: Lifecycle | None = None,
    config_path: Path | None = None,
    overrides: Overrides | None = None,
) -> ASGIApp:
    return RequestContextMiddleware(
        build_api(
            settings, routing, lifecycle=lifecycle, config_path=config_path, overrides=overrides
        )
    )


def app_factory() -> ASGIApp:
    """Zero-argument factory for ``uvicorn --factory`` (used by ``make dev``)."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    try:
        routing = load_routing_config(settings.routing_config)
    except ConfigError as exc:
        raise SystemExit(exc.render()) from None
    return create_app(settings, routing, config_path=settings.routing_config)


@dataclass(slots=True)
class _Services:
    """Keys, limits, alerts and the usage ledger: started together, stopped together."""

    keys: KeyResolver
    limiter: Limiter | None
    alerts: Alerts
    usage: UsageSink
    writer: UsageWriter | None
    listener: asyncio.Task[None] | None

    @classmethod
    def start(
        cls,
        settings: Settings,
        given: Overrides,
        *,
        redis: Redis | None,
        pg_pool: asyncpg.Pool[asyncpg.Record] | None,
    ) -> _Services:
        store = given.key_store or (PostgresKeyStore(pg_pool) if pg_pool else NoKeyStore())
        keys = KeyResolver(store, _pepper(settings), ttl=settings.key_cache_ttl_seconds)
        listener = asyncio.create_task(keys.listen(redis)) if redis is not None else None
        webhook = str(settings.alert_webhook_url) if settings.alert_webhook_url else None
        alerts = Alerts(pool=pg_pool, webhook_url=webhook)
        limiter = None
        if redis is not None:
            limiter = Limiter(
                redis,
                spend_source=ledger_spend(pg_pool) if pg_pool is not None else None,
                budget_on_failure=settings.budget_on_redis_failure,
                on_alert=alerts,
            )
        writer = None
        usage = given.usage_sink
        if usage is None and pg_pool is not None:
            usage = writer = UsageWriter(
                pg_pool,
                queue_size=settings.usage_queue_size,
                batch_rows=settings.usage_batch_rows,
                flush_seconds=settings.usage_flush_seconds,
            )
            writer.start()
        return cls(keys, limiter, alerts, usage or MemoryUsageSink(), writer, listener)

    async def aclose(self) -> None:
        if self.listener is not None:
            self.listener.cancel()
            await asyncio.wait({self.listener})
        if self.limiter is not None:
            await self.limiter.aclose()  # settlements still in flight
        await self.alerts.aclose()
        if self.writer is not None:
            await self.writer.aclose()  # flush the ledger last


def _pepper(settings: Settings) -> bytes:
    if settings.key_pepper is not None:
        return settings.key_pepper.get_secret_value().encode()
    # No database means no stored keys, so nothing needs a stable pepper.
    return secrets.token_bytes(32)


def _probes(
    settings: Settings, redis: Redis | None, pg_pool: asyncpg.Pool[asyncpg.Record] | None
) -> list[NamedProbe]:
    probes: list[NamedProbe] = []
    if redis is not None:
        probes.append(NamedProbe("redis", partial(ping_redis, redis)))
    if pg_pool is not None:
        probes.append(NamedProbe("postgres", partial(ping_postgres, pg_pool)))
        probes.append(
            NamedProbe("migrations", partial(_migrations_applied, pg_pool, settings.migrations_dir))
        )
    return probes


async def _migrations_applied(pool: asyncpg.Pool[asyncpg.Record], directory: Path) -> None:
    async with pool.acquire() as conn:
        missing = await pending(conn, directory)
    if missing:
        raise RuntimeError(f"pending migrations: {', '.join(missing)} (run `switchboard migrate`)")


def _log_startup(
    settings: Settings,
    routing: LoadedConfig,
    config: ConfigManager,
    given: Overrides,
    *,
    redis: Redis | None,
    pg_pool: asyncpg.Pool[asyncpg.Record] | None,
) -> None:
    for warning in routing.warnings:
        _log.warning("config.warning", problem=str(warning))
    if pg_pool is None and given.key_store is None:
        _log.warning("auth.no_keys", detail="no database configured; /v1 rejects every key")
    if redis is None:
        _log.warning("limits.disabled", detail="no Redis configured; rate limits and budgets off")
    if routing.pricing is None:
        _log.warning("pricing.none", detail="no pricing file in the routing config; costs are 0")
    _log.info(
        "gateway.started",
        version=__version__,
        config_source=routing.source,
        config_version=routing.version,
        pricing_version=routing.pricing.version if routing.pricing else None,
        providers=sorted(config.current.adapters),
        disabled_providers=sorted(routing.disabled_providers),
        port=settings.port,
        metrics_port=settings.metrics_port or None,
    )


async def _wait_idle(lifecycle: Lifecycle, limit_seconds: float) -> bool:
    try:
        async with asyncio.timeout(limit_seconds):
            await lifecycle.wait_idle()
    except TimeoutError:
        return False
    return True


def _start_metrics_server(settings: Settings) -> WSGIServer | None:
    if not settings.metrics_port:
        return None
    server, _thread = start_http_server(settings.metrics_port, addr=settings.host)
    return server
