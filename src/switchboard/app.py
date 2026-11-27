"""Application factory for the gateway data plane."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from functools import partial
from wsgiref.simple_server import WSGIServer

import structlog
from fastapi import FastAPI
from prometheus_client import start_http_server
from starlette.types import ASGIApp

from switchboard import __version__
from switchboard.api import chat, embeddings, health, models
from switchboard.api.errors import install_error_handlers
from switchboard.api.middleware import RequestContextMiddleware
from switchboard.lifecycle import Lifecycle
from switchboard.observability.logging import configure_logging
from switchboard.observability.metrics import CONFIG_INFO
from switchboard.providers.registry import build_adapters
from switchboard.providers.thinking import MemoryThinkingSidecar, RedisThinkingSidecar
from switchboard.readiness import NamedProbe, ReadinessChecker
from switchboard.routing import ConfigError, LoadedConfig, load_routing_config
from switchboard.routing.table import RoutingTable
from switchboard.settings import Settings
from switchboard.state import AppState
from switchboard.storage.clients import create_pg_pool, create_redis, ping_postgres, ping_redis

_log = structlog.get_logger("switchboard.app")

_POOL_CLOSE_TIMEOUT = 5.0
_DRAIN_TIMEOUT_AFTER_SERVER = 5.0


def build_api(
    settings: Settings, routing: LoadedConfig, *, lifecycle: Lifecycle | None = None
) -> FastAPI:
    """Build the FastAPI app. Most callers want :func:`create_app` instead."""
    lifecycle = lifecycle or Lifecycle()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis = create_redis(str(settings.redis_url)) if settings.redis_url else None
        pg_pool = (
            await create_pg_pool(str(settings.database_url)) if settings.database_url else None
        )
        probes: list[NamedProbe] = []
        if redis is not None:
            probes.append(NamedProbe("redis", partial(ping_redis, redis)))
        if pg_pool is not None:
            probes.append(NamedProbe("postgres", partial(ping_postgres, pg_pool)))

        sidecar = RedisThinkingSidecar(redis) if redis is not None else MemoryThinkingSidecar()
        adapters = build_adapters(routing, sidecar=sidecar)
        app.state.switchboard = AppState(
            settings=settings,
            routing=routing,
            lifecycle=lifecycle,
            readiness=ReadinessChecker(
                probes,
                timeout=settings.ready_check_timeout_seconds,
                cache_seconds=settings.ready_cache_seconds,
            ),
            routing_table=RoutingTable(routing),
            adapters=adapters,
            redis=redis,
            pg_pool=pg_pool,
        )
        CONFIG_INFO.labels(version=routing.version).set(1)
        metrics_server = _start_metrics_server(settings)

        for warning in routing.warnings:
            _log.warning("config.warning", problem=str(warning))
        if settings.dev_api_key is None:
            _log.warning("auth.no_keys", detail="SWITCHBOARD_DEV_API_KEY is unset; /v1 rejects all")
        _log.info(
            "gateway.started",
            version=__version__,
            config_source=routing.source,
            config_version=routing.version,
            providers=sorted(adapters),
            disabled_providers=sorted(routing.disabled_providers),
            port=settings.port,
            metrics_port=settings.metrics_port or None,
        )
        try:
            yield
        finally:
            # uvicorn has already stopped accepting connections and waited up to the grace
            # period for open ones; this catches work that outlived its connection.
            _log.info("gateway.stopping", inflight=lifecycle.inflight)
            drained = await _wait_idle(lifecycle, _DRAIN_TIMEOUT_AFTER_SERVER)
            await asyncio.gather(*(adapter.aclose() for adapter in adapters.values()))
            if metrics_server is not None:
                await asyncio.to_thread(metrics_server.shutdown)
                metrics_server.server_close()
            if redis is not None:
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
    settings: Settings, routing: LoadedConfig, *, lifecycle: Lifecycle | None = None
) -> ASGIApp:
    return RequestContextMiddleware(build_api(settings, routing, lifecycle=lifecycle))


def app_factory() -> ASGIApp:
    """Zero-argument factory for ``uvicorn --factory`` (used by ``make dev``)."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    try:
        routing = load_routing_config(settings.routing_config)
    except ConfigError as exc:
        raise SystemExit(exc.render()) from None
    return create_app(settings, routing)


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
