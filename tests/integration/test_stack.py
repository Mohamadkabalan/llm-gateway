"""Against the running compose stack (`make up`, then `make test-integration`).

The stack's `fake` alias routes to the fake-provider container, so these run without
any provider keys.
"""

import asyncio
import os
from collections.abc import AsyncIterator

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from redis.asyncio import Redis

from switchboard.reliability.breaker import BreakerState, RedisBreakerStore, redis_key

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> openai.OpenAI:
    key = os.environ.get("SWITCHBOARD_DEMO_KEY")
    if not key:
        pytest.skip("SWITCHBOARD_DEMO_KEY is not set (run via `make test-integration`)")
    base = os.environ.get("SB_GATEWAY_URL", "http://127.0.0.1:7080")
    return openai.OpenAI(base_url=f"{base}/v1", api_key=key, max_retries=0, timeout=30)


def test_sdk_streams_through_the_stack(client: openai.OpenAI) -> None:
    stream = client.chat.completions.create(
        model="fake",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text.strip()
    assert chunks[-1].usage is not None


def test_sdk_non_streaming_through_the_stack(client: openai.OpenAI) -> None:
    completion = client.chat.completions.create(
        model="fake", messages=[{"role": "user", "content": "hello"}]
    )
    assert completion.choices[0].message.content
    assert completion.usage is not None


def test_python_sdk_tool_round_trip_through_the_anthropic_translation(
    client: openai.OpenAI,
) -> None:
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ]
    ask: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Weather in Paris?"}]
    first = client.chat.completions.create(model="fake-claude", messages=ask, tools=tools)
    assert first.choices[0].finish_reason == "tool_calls"
    call = (first.choices[0].message.tool_calls or [])[0]
    history: list[ChatCompletionMessageParam] = [
        *ask,
        first.choices[0].message.model_dump(exclude_none=True),  # type: ignore[list-item]
        {"role": "tool", "tool_call_id": call.id, "content": "18C"},
    ]
    second = client.chat.completions.create(model="fake-claude", messages=history, tools=tools)
    assert second.choices[0].finish_reason == "stop"
    assert second.choices[0].message.content


def test_embeddings_and_models_through_the_stack(client: openai.OpenAI) -> None:
    vectors = client.embeddings.create(model="fake-embed", input=["a", "b"])
    assert len(vectors.data) == 2
    ids = {model.id for model in client.models.list()}
    assert {"fake", "fake-claude", "fake-embed"} <= ids


# --- Phase 3: failover, the Redis-backed breaker and rules, through the stack --------

FLAKY = "fake-flaky/fake-claude-flaky"  # the fake-fallback alias's primary


@pytest.fixture
def fake_url() -> str:
    return os.environ.get("SB_FAKE_URL", "http://127.0.0.1:7800")


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    url = os.environ.get("SB_REDIS_URL")
    if not url:
        pytest.skip("SB_REDIS_URL is not set (run via `make test-integration`)")
    client = Redis.from_url(url)
    try:
        yield client
    finally:
        await client.aclose()


def test_failover_through_the_stack(client: openai.OpenAI) -> None:
    raw = client.chat.completions.with_raw_response.create(
        model="fake-fallback",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={
            # Only the Anthropic-format primary sees this fault.
            "provider_options": {"anthropic": {"fake": {"fault": "status", "status": 503}}}
        },
    )
    assert raw.parse().choices[0].message.content
    assert raw.headers["x-switchboard-model"] == "fake/fake-chat"
    assert raw.headers["x-switchboard-fallback"] == "true"


def test_tags_select_a_rule_through_the_stack(client: openai.OpenAI) -> None:
    raw = client.chat.completions.with_raw_response.create(
        model="fake",
        messages=[{"role": "user", "content": "hello"}],
        extra_headers={"x-switchboard-tags": "via-claude"},
    )
    assert raw.headers["x-switchboard-model"] == "fake-anthropic/fake-claude"


async def test_breaker_opens_and_recovers_through_the_stack(
    client: openai.OpenAI, redis: Redis, fake_url: str
) -> None:
    """fake-flaky's breaker (min_requests 5, cooldown 3s, 2 probes) lives in Redis."""
    key = redis_key(FLAKY)
    await redis.delete(key)

    def served_by() -> str:
        raw = client.chat.completions.with_raw_response.create(
            model="fake-fallback", messages=[{"role": "user", "content": "hello"}]
        )
        return raw.headers["x-switchboard-model"]

    async with httpx.AsyncClient(base_url=fake_url, timeout=5) as fake:
        try:
            fault = {"fault": "status", "status": 503, "fault_models": ["fake-claude-flaky"]}
            (await fake.put("/_control/behavior", json=fault)).raise_for_status()
            assert [served_by() for _ in range(5)] == ["fake/fake-chat"] * 5
            assert await RedisBreakerStore(redis).state(FLAKY) is BreakerState.OPEN

            seen = await primary_requests(fake)
            assert served_by() == "fake/fake-chat"
            assert await primary_requests(fake) == seen  # skipped without a call

            (await fake.delete("/_control/behavior")).raise_for_status()
            await asyncio.sleep(3.2)  # the cooldown
            assert [served_by() for _ in range(3)] == [FLAKY] * 3  # 2 probes, then closed
            assert await RedisBreakerStore(redis).state(FLAKY) is BreakerState.CLOSED
        finally:
            await fake.delete("/_control/behavior")
            await redis.delete(key)


async def primary_requests(fake: httpx.AsyncClient) -> int:
    """Anthropic-format calls the fake has received (the fallback uses the OpenAI path)."""
    stats = (await fake.get("/_control/stats")).json()
    return int(stats["requests"].get("/v1/messages", 0))
