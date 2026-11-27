"""Real servers on loopback: the gateway and the fake provider, both under uvicorn.

The servers run as tasks in the test's own event loop on ephemeral ports, so these
tests need no Docker and stay fast, while exercising real sockets, streaming and
client disconnects.
"""

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
import pytest
import uvicorn
from starlette.types import ASGIApp

from fakeprovider.app import FakeProvider
from switchboard.app import create_app
from switchboard.routing import parse_routing_config
from tests.conftest import SettingsFactory

API_KEY = "sb-test-key"


@contextlib.asynccontextmanager
async def serve(app: ASGIApp) -> AsyncIterator[str]:
    """Run ``app`` under uvicorn on 127.0.0.1 and yield its base URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app, log_config=None, lifespan="on", timeout_graceful_shutdown=1, timeout_keep_alive=75
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                task.result()  # surface startup errors
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()


def routing_yaml(
    fake_url: str, *, defaults: str = "", providers: str = "", models: str = "", aliases: str = ""
) -> str:
    """The base test config plus extra YAML lines; ``{fake_url}`` in them is filled in."""
    providers = providers.replace("{fake_url}", fake_url)
    return f"""\
version: 1
{defaults}
providers:
  fake:
    type: openai
    base_url: {fake_url}/v1
    api_key_env: FAKE_UPSTREAM_KEY
{providers}
models:
  fake/fake-chat:
    upstream_model: fake-chat
    defaults: {{ max_tokens: 64 }}
  fake/fake-embed:
    upstream_model: fake-embed
    kind: embedding
{models}
aliases:
  fake:
    targets: [{{ model: fake/fake-chat }}]
  embed:
    targets: [{{ model: fake/fake-embed }}]
{aliases}
"""


@dataclass
class Stack:
    gateway_url: str
    fake: FakeProvider
    fake_url: str
    _sdk_clients: list[openai.AsyncOpenAI] = field(default_factory=list)

    def http(self, **kwargs: Any) -> httpx.AsyncClient:
        """A raw client with the API key set; use it as an async context manager."""
        headers = {"authorization": f"Bearer {API_KEY}", **kwargs.pop("headers", {})}
        return httpx.AsyncClient(base_url=self.gateway_url, headers=headers, timeout=10, **kwargs)

    def sdk(self) -> openai.AsyncOpenAI:
        """An OpenAI SDK client pointed at the gateway (closed when the stack stops)."""
        client = openai.AsyncOpenAI(
            base_url=f"{self.gateway_url}/v1", api_key=API_KEY, max_retries=0, timeout=10
        )
        self._sdk_clients.append(client)
        return client

    async def aclose(self) -> None:
        for client in self._sdk_clients:
            await client.close()


StackFactory = Callable[..., contextlib.AbstractAsyncContextManager[Stack]]


@pytest.fixture
def make_stack(make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch) -> StackFactory:
    # The gateway reads provider keys from the environment, as in production.
    monkeypatch.setenv("FAKE_UPSTREAM_KEY", "sk-upstream-test")

    @contextlib.asynccontextmanager
    async def factory(
        *,
        defaults: str = "",
        providers: str = "",
        models: str = "",
        aliases: str = "",
        **settings: Any,
    ) -> AsyncIterator[Stack]:
        fake = FakeProvider(seed=1)
        async with serve(fake) as fake_url:
            routing = parse_routing_config(
                routing_yaml(
                    fake_url, defaults=defaults, providers=providers, models=models, aliases=aliases
                ),
                source="test.yaml",
                env={"FAKE_UPSTREAM_KEY": "sk-upstream-test"},
            )
            settings.setdefault("dev_api_key", API_KEY)
            app = create_app(make_settings(**settings), routing)
            async with serve(app) as gateway_url:
                running = Stack(gateway_url, fake, fake_url)
                try:
                    yield running
                finally:
                    await running.aclose()

    return factory


@pytest.fixture
async def stack(make_stack: StackFactory) -> AsyncIterator[Stack]:
    async with make_stack() as running:
        yield running
