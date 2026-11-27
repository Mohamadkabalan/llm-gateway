"""Health, readiness, request ids and the OpenAI error envelope, over real ASGI."""

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import structlog
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from switchboard.api.middleware import RequestContextMiddleware
from switchboard.app import build_api, create_app
from switchboard.lifecycle import Lifecycle
from switchboard.routing import LoadedConfig
from tests.conftest import SettingsFactory


async def test_health_is_always_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_without_dependencies(client: httpx.AsyncClient, routing: LoadedConfig) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "config_version": routing.version, "checks": {}}


async def test_ready_returns_503_while_draining(
    client: httpx.AsyncClient, lifecycle: Lifecycle
) -> None:
    lifecycle.start_draining()
    response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "draining"


async def test_ready_reports_unreachable_dependencies(
    make_settings: SettingsFactory, routing: LoadedConfig
) -> None:
    # Port 1 on localhost refuses connections immediately: real client code, no docker.
    settings = make_settings(
        redis_url="redis://127.0.0.1:1/0",
        database_url="postgresql://u:p@127.0.0.1:1/db",
        ready_check_timeout_seconds=2,
    )
    app = create_app(settings, routing)
    async with (
        LifespanManager(app) as manager,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app), base_url="http://gateway"
        ) as http,
    ):
        response = await http.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["redis"]["ok"] is False
    assert body["checks"]["postgres"]["ok"] is False


async def test_request_id_is_generated_as_uuid7(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    request_id = uuid.UUID(response.headers["x-request-id"])
    assert request_id.version == 7


async def test_valid_incoming_request_id_is_reused(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "trace-abc.123:4"})
    assert response.headers["x-request-id"] == "trace-abc.123:4"


@pytest.mark.parametrize("bad", ["a" * 129, "has space", "semi;colon", "<script>"])
async def test_unsafe_incoming_request_id_is_replaced(client: httpx.AsyncClient, bad: str) -> None:
    response = await client.get("/health", headers={"X-Request-ID": bad})
    assert response.headers["x-request-id"] != bad
    assert uuid.UUID(response.headers["x-request-id"]).version == 7


async def test_unknown_route_uses_openai_error_shape(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {
        "error": {"message": "Not Found", "type": "not_found_error", "param": None, "code": None}
    }
    assert "x-request-id" in response.headers


async def test_wrong_method_uses_openai_error_shape(client: httpx.AsyncClient) -> None:
    response = await client.post("/health")
    assert response.status_code == 405
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.fixture
async def boom_client(
    make_settings: SettingsFactory, routing: LoadedConfig
) -> AsyncIterator[httpx.AsyncClient]:
    api: FastAPI = build_api(make_settings(), routing)

    async def boom() -> None:
        raise RuntimeError("kaboom")

    api.add_api_route("/boom", boom)
    app = RequestContextMiddleware(api)
    async with (
        LifespanManager(app) as manager,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app, raise_app_exceptions=False),
            base_url="http://gateway",
        ) as http,
    ):
        yield http


async def test_unhandled_error_is_500_with_request_id_and_logged(
    boom_client: httpx.AsyncClient,
) -> None:
    with structlog.testing.capture_logs() as logs:
        response = await boom_client.get("/boom", headers={"X-Request-ID": "req-1"})
    assert response.status_code == 500
    assert response.headers["x-request-id"] == "req-1"
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert "req-1" in error["message"]
    assert "kaboom" not in error["message"]  # internals are logged, never returned
    logged = [entry for entry in logs if entry["event"] == "http.unhandled_exception"]
    assert len(logged) == 1
    assert logged[0]["log_level"] == "error"


async def test_access_log_has_request_id_and_skips_probes(client: httpx.AsyncClient) -> None:
    with structlog.testing.capture_logs() as logs:
        await client.get("/health")
        await client.get("/nope", headers={"X-Request-ID": "req-2"})
    access = [entry for entry in logs if entry["event"] == "http.request"]
    assert [entry["path"] for entry in access] == ["/nope"]
    assert access[0]["status"] == 404
    assert access[0]["duration_ms"] >= 0
