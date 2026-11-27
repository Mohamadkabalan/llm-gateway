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
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
import uvicorn
from redis.asyncio import Redis
from starlette.types import ASGIApp

from fakeprovider.app import FakeProvider
from switchboard.accounting.usage import MemoryUsageSink
from switchboard.app import Overrides, create_app
from switchboard.auth.store import MemoryKeyStore
from switchboard.routing import load_routing_config, parse_routing_config
from tests.conftest import SettingsFactory
from tests.support import REPO_ROOT, TEST_KEY, TEST_PEPPER, key_store

API_KEY = TEST_KEY


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
    fake_url: str,
    *,
    defaults: str = "",
    providers: str = "",
    models: str = "",
    aliases: str = "",
    rules: str = "",
) -> str:
    """The base test config plus extra YAML lines; ``{fake_url}`` in them is filled in."""
    providers = providers.replace("{fake_url}", fake_url)
    rules = f"rules:\n{rules}" if rules else ""
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
{rules}
"""


@dataclass
class Stack:
    gateway_url: str
    fake: FakeProvider
    fake_url: str
    usage: MemoryUsageSink
    keys: MemoryKeyStore
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
        rules: str = "",
        config_file: Path | None = None,
        keys: MemoryKeyStore | None = None,
        redis: Redis | None = None,
        priced: bool = False,
        alert_webhook: bool = False,
        **settings: Any,
    ) -> AsyncIterator[Stack]:
        """With ``config_file``, the config is written there and the gateway watches it.
        ``keys`` replaces the one test key; ``redis`` (e.g. fakeredis) turns on rate
        limits and budgets; ``priced`` uses the shipped pricing file; ``alert_webhook``
        sends budget alerts to the fake provider's webhook sink."""
        fake = FakeProvider(seed=1)
        async with serve(fake) as fake_url:
            text = routing_yaml(
                fake_url,
                defaults=defaults,
                providers=providers,
                models=models,
                aliases=aliases,
                rules=rules,
            )
            env = {"FAKE_UPSTREAM_KEY": "sk-upstream-test"}
            if priced:
                text = text.replace(
                    "version: 1\n", "version: 1\npricing: pricing/2026-11-27.yaml\n", 1
                )
            if config_file is None:
                routing = parse_routing_config(
                    text, source="test.yaml", env=env, base_dir=REPO_ROOT / "config"
                )
            else:
                config_file.write_text(text)
                routing = load_routing_config(config_file, env=env)
            settings.setdefault("key_pepper", TEST_PEPPER)
            if alert_webhook:
                settings["alert_webhook_url"] = f"{fake_url}/_webhook"
            store = keys if keys is not None else key_store()
            usage = MemoryUsageSink()
            overrides = Overrides(key_store=store, usage_sink=usage, redis=redis)
            app = create_app(
                make_settings(**settings), routing, config_path=config_file, overrides=overrides
            )
            async with serve(app) as gateway_url:
                running = Stack(gateway_url, fake, fake_url, usage, store)
                try:
                    yield running
                finally:
                    await running.aclose()

    return factory


@pytest.fixture
async def stack(make_stack: StackFactory) -> AsyncIterator[Stack]:
    async with make_stack() as running:
        yield running
