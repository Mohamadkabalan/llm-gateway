"""Phase 4 through real sockets: virtual keys, rate limits, budgets, alerts and the ledger.

Rate limits and budgets run the real Lua scripts on fakeredis; costs use the shipped
pricing file, where the fake models are priced like Claude Sonnet 5.5 ($2 in, $10 out
per million tokens).
"""

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import fakeredis
import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam
from prometheus_client import REGISTRY

from switchboard.accounting.usage import UsageEvent
from switchboard.auth.context import Limits
from tests.component.conftest import API_KEY, Stack, StackFactory
from tests.support import key_context, key_store

USER: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]
FAST = {"fake": {"ttft_ms": 0, "interval_ms": 0, "chunks": 4}}
SONNET_IN, SONNET_OUT = Decimal(2), Decimal(10)  # USD per million tokens


async def wait_for(condition: Any, within: float = 3.0) -> None:
    async with asyncio.timeout(within):
        while not condition():  # noqa: ASYNC110 (another server's state)
            await asyncio.sleep(0.01)


def limited(**limits: Any) -> Any:
    """A stack whose one key has these limits, with fakeredis behind the limiter."""
    return key_store((API_KEY, key_context(key_limits=Limits(**limits))))


@pytest.fixture
async def priced(make_stack: StackFactory) -> AsyncIterator[Stack]:
    async with make_stack(priced=True, redis=fakeredis.FakeAsyncRedis()) as stack:
        yield stack


def events(stack: Stack, status: str | None = None) -> list[UsageEvent]:
    return [e for e in stack.usage.events if status is None or e.status == status]


# --- the ledger ------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
async def test_one_usage_row_per_request_with_tokens_and_cost(priced: Stack, stream: bool) -> None:
    create = priced.sdk().chat.completions.with_raw_response.create
    if stream:
        raw = await create(
            model="fake",
            messages=USER,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=FAST,
        )
        chunks = [chunk async for chunk in raw.parse()]
        usage = chunks[-1].usage
        request_id = raw.headers["x-request-id"]
    else:
        plain = await create(model="fake", messages=USER, extra_body=FAST)
        usage = plain.parse().usage
        request_id = plain.headers["x-request-id"]
    assert usage is not None
    await wait_for(lambda: len(events(priced)) == 1)
    (event,) = events(priced)
    assert (event.status, event.http_status, event.endpoint) == ("ok", 200, "chat")
    assert (event.provider, event.model, event.requested_model) == (
        "fake",
        "fake/fake-chat",
        "fake",
    )
    assert event.input_tokens == usage.prompt_tokens
    assert event.output_tokens == usage.completion_tokens == 4
    expected = (usage.prompt_tokens * SONNET_IN + 4 * SONNET_OUT) / 1_000_000
    assert event.cost_usd == expected
    assert not event.cost_estimated
    assert event.pricing_version == "2026-11-27"
    assert event.request_id == request_id
    assert event.stream is stream
    assert event.latency_ms >= 0


async def test_refusals_are_recorded_too(priced: Stack) -> None:
    async with priced.http() as http:
        missing = await http.post("/v1/chat/completions", json={"model": "nope", "messages": USER})
        invalid = await http.post("/v1/chat/completions", json={"model": "fake"})
    assert (missing.status_code, invalid.status_code) == (404, 400)
    await wait_for(lambda: len(events(priced)) == 2)
    first, second = events(priced)
    assert (first.status, first.http_status, first.error_type) == (
        "invalid_request",
        404,
        "model_not_found",
    )
    assert first.requested_model == "nope"
    assert (second.status, second.http_status) == ("invalid_request", 400)
    assert first.cost_usd == second.cost_usd == 0


async def test_a_stream_cut_short_is_priced_as_an_estimate(priced: Stack) -> None:
    stream = await priced.sdk().chat.completions.create(
        model="fake",
        messages=USER,
        stream=True,
        extra_body={"fake": {"chunks": 8, "fault": "disconnect", "fault_after": 3}},
    )
    with pytest.raises(openai.APIError):
        async for _ in stream:
            pass
    await wait_for(lambda: len(events(priced)) == 1)
    (event,) = events(priced)
    assert event.status == "upstream_error"
    assert event.output_tokens is None  # no usage arrived
    assert event.cost_estimated
    assert event.cost_usd > 0


async def test_tokens_and_cost_reach_the_metrics(priced: Stack) -> None:
    labels = {"team": "test-team", "provider": "fake", "model": "fake/fake-chat"}
    before = REGISTRY.get_sample_value("switchboard_cost_usd_total", labels) or 0.0
    out_before = REGISTRY.get_sample_value("switchboard_tokens_total", {"type": "output"}) or 0.0
    await priced.sdk().chat.completions.create(model="fake", messages=USER, extra_body=FAST)
    await wait_for(lambda: len(events(priced)) == 1)
    after = REGISTRY.get_sample_value("switchboard_cost_usd_total", labels) or 0.0
    assert after - before == pytest.approx(float(events(priced)[0].cost_usd))
    out_after = REGISTRY.get_sample_value("switchboard_tokens_total", {"type": "output"}) or 0.0
    assert out_after - out_before == 4


# --- keys ------------------------------------------------------------------------------


async def test_keys_are_looked_up_once_then_cached(stack: Stack) -> None:
    client = stack.sdk()
    for _ in range(5):
        await client.chat.completions.create(model="fake", messages=USER, extra_body=FAST)
    assert stack.keys.lookups == 1


async def test_a_revoked_key_is_refused(make_stack: StackFactory) -> None:
    store = key_store((API_KEY, key_context(revoked=True)))
    async with make_stack(keys=store) as stack, stack.http() as http:
        response = await http.post("/v1/chat/completions", json={"model": "fake", "messages": USER})
    assert response.status_code == 401
    assert "revoked" in response.json()["error"]["message"]


async def test_allowed_models_restrict_requests_and_the_model_list(
    make_stack: StackFactory,
) -> None:
    store = key_store((API_KEY, key_context(allowed_models=frozenset({"fake"}))))
    async with make_stack(keys=store) as stack, stack.http() as http:
        allowed = await http.post(
            "/v1/chat/completions", json={"model": "fake", "messages": USER, **FAST}
        )
        refused = await http.post("/v1/embeddings", json={"model": "embed", "input": "hi"})
        listed = (await http.get("/v1/models")).json()["data"]
    assert allowed.status_code == 200
    assert refused.status_code == 404
    assert refused.json()["error"]["code"] == "model_not_found"
    assert [m["id"] for m in listed] == ["fake"]


# --- rate limits and budgets ------------------------------------------------------------


async def test_rpm_limit_returns_429_with_rate_limit_headers(make_stack: StackFactory) -> None:
    async with (
        make_stack(keys=limited(rpm=2), redis=fakeredis.FakeAsyncRedis()) as stack,
        stack.http() as http,
    ):
        body = {"model": "fake", "messages": USER, **FAST}
        ok = [await http.post("/v1/chat/completions", json=body) for _ in range(2)]
        refused = await http.post("/v1/chat/completions", json=body)
    assert [r.status_code for r in ok] == [200, 200]
    assert ok[0].headers["x-ratelimit-limit-requests"] == "2"
    assert ok[0].headers["x-ratelimit-remaining-requests"] == "1"
    assert refused.status_code == 429
    error = refused.json()["error"]
    assert (error["type"], error["code"]) == ("requests", "rate_limit_exceeded")
    assert refused.headers["retry-after"] == "30"
    assert refused.headers["x-ratelimit-remaining-requests"] == "0"
    await wait_for(lambda: len(events(stack, "rate_limited")) == 1)


async def test_tpm_limit_returns_429_for_tokens(make_stack: StackFactory) -> None:
    async with (
        make_stack(keys=limited(tpm=50), redis=fakeredis.FakeAsyncRedis()) as stack,
        stack.http() as http,
    ):
        body = {"model": "fake", "messages": USER, **FAST}
        first = await http.post("/v1/chat/completions", json=body)  # needs a full bucket
        second = await http.post("/v1/chat/completions", json=body)
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["type"] == "tokens"
    assert second.headers["x-ratelimit-limit-tokens"] == "50"
    assert "retry-after" in second.headers


async def test_hard_budget_blocks_and_the_sdk_does_not_retry(make_stack: StackFactory) -> None:
    # The estimate (prompt + 64 output tokens at Sonnet prices) is about $0.0007.
    store = limited(hard_usd=Decimal("0.0005"))
    async with make_stack(keys=store, redis=fakeredis.FakeAsyncRedis(), priced=True) as stack:
        client = openai.AsyncOpenAI(
            base_url=f"{stack.gateway_url}/v1", api_key=API_KEY, max_retries=3, timeout=10
        )
        try:
            with pytest.raises(openai.RateLimitError) as caught:
                await client.chat.completions.create(model="fake", messages=USER, extra_body=FAST)
        finally:
            await client.close()
        assert caught.value.code == "insufficient_quota"
        assert caught.value.response.headers["x-should-retry"] == "false"
        await wait_for(lambda: len(events(stack)) >= 1)
        await asyncio.sleep(0.2)
        assert [e.status for e in events(stack)] == ["budget_blocked"]  # one request, no retries
        assert stack.fake.stats.requests["/v1/chat/completions"] == 0


async def test_soft_alert_fires_exactly_once(make_stack: StackFactory) -> None:
    before = REGISTRY.get_sample_value("switchboard_budget_alerts_total", {"scope": "key"}) or 0
    store = limited(soft_usd=Decimal("0.00001"))
    async with make_stack(
        keys=store, redis=fakeredis.FakeAsyncRedis(), priced=True, alert_webhook=True
    ) as stack:
        client = stack.sdk()
        for _ in range(3):
            await client.chat.completions.create(model="fake", messages=USER, extra_body=FAST)
        await wait_for(lambda: len(stack.fake.stats.webhooks) == 1)
        await asyncio.sleep(0.3)  # time for a duplicate, if there were one
        (alert,) = stack.fake.stats.webhooks
    assert "soft budget" in alert["text"]
    assert alert["switchboard"]["event"] == "budget.soft_limit_reached"
    after = REGISTRY.get_sample_value("switchboard_budget_alerts_total", {"scope": "key"}) or 0
    assert after - before == 1
