"""Live tests: the real gateway, in-process, against the real providers.

Run with `make test-live`, which loads provider keys from .env. Each provider's tests
are skipped when its key is not set. A full run makes about a dozen short calls.
"""

import os
from collections.abc import AsyncIterator

import openai
import pytest
from pydantic import SecretStr

from switchboard.app import Overrides, create_app
from switchboard.routing import parse_routing_config
from switchboard.settings import Settings
from tests.component.conftest import serve
from tests.support import TEST_KEY, TEST_PEPPER, key_context, key_store

KEY = TEST_KEY
CONFIG = """\
version: 1
providers:
  anthropic:
    type: anthropic
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    api_version: "2026-10-21"
    optional: true
  openai:
    type: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    optional: true
models:
  anthropic/claude-sonnet-5-5:
    upstream_model: claude-sonnet-5-5
    capabilities: { tools: true, vision: true, forced_tool_choice: false, json_schema: true,
                    json_object: false }
    defaults: { max_tokens: 512 }
  openai/gpt-5.4-mini:
    upstream_model: gpt-5.4-mini
    capabilities: { tools: true, vision: true, json_schema: true }
  openai/text-embedding-3-small:
    upstream_model: text-embedding-3-small
    kind: embedding
"""


@pytest.fixture(autouse=True)
def _isolate_env() -> None:
    """Live tests need the real provider keys, so they skip the root isolation."""


def require(variable: str) -> None:
    if not os.environ.get(variable):
        pytest.skip(f"{variable} is not set")


@pytest.fixture
async def gateway() -> AsyncIterator[openai.AsyncOpenAI]:
    routing = parse_routing_config(CONFIG, source="live.yaml", env=dict(os.environ))
    settings = Settings(key_pepper=SecretStr(TEST_PEPPER), metrics_port=0)
    overrides = Overrides(key_store=key_store((KEY, key_context(KEY))))
    app = create_app(settings, routing, overrides=overrides)
    async with serve(app) as url:
        client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key=KEY, max_retries=0, timeout=60)
        try:
            yield client
        finally:
            await client.close()
