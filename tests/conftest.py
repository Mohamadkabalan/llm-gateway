import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from switchboard.app import create_app
from switchboard.lifecycle import Lifecycle
from switchboard.routing import LoadedConfig, parse_routing_config
from switchboard.settings import Settings
from tests.support import MINIMAL_CONFIG


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's SWITCHBOARD_* and provider key variables out of tests."""
    for name in list(os.environ):
        if name.startswith("SWITCHBOARD_") or name.endswith("_API_KEY"):
            monkeypatch.delenv(name)


@pytest.fixture
def routing() -> LoadedConfig:
    return parse_routing_config(MINIMAL_CONFIG, source="test.yaml", env={})


SettingsFactory = Callable[..., Settings]


@pytest.fixture
def make_settings() -> SettingsFactory:
    def factory(**overrides: Any) -> Settings:
        values: dict[str, Any] = {"metrics_port": 0, "ready_cache_seconds": 0}
        values.update(overrides)
        return Settings(**values)

    return factory


@pytest.fixture
def lifecycle() -> Lifecycle:
    return Lifecycle()


@pytest.fixture
async def client(
    make_settings: SettingsFactory, routing: LoadedConfig, lifecycle: Lifecycle
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(make_settings(), routing, lifecycle=lifecycle)
    async with (
        LifespanManager(app) as manager,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app), base_url="http://gateway"
        ) as http,
    ):
        yield http
