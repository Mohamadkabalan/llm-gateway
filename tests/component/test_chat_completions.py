"""``/v1/chat/completions`` end to end: real gateway, real fake provider, real sockets.

The OpenAI Python SDK is the contract test: if it can stream, aggregate, parse tool
calls and raise the right exceptions through the gateway, clients will work.
"""

import asyncio
import io
import json
from typing import Any

import openai
import orjson
import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from prometheus_client import REGISTRY

from switchboard.observability.logging import configure_logging
from tests.component.conftest import Stack, StackFactory

USER: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]
FAST = {"ttft_ms": 0, "interval_ms": 0, "chunks": 4}
TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
    }
]


def fake(**overrides: Any) -> dict[str, Any]:
    """Per-request fake-provider behavior, sent through the gateway in the body."""
    return {"fake": {**FAST, **overrides}}


async def wait_for(condition: Any, within: float = 3.0) -> None:
    """Poll another server's counters (they live in a different coroutine's world)."""
    async with asyncio.timeout(within):
        while not condition():  # noqa: ASYNC110 (there is no event to wait on)
            await asyncio.sleep(0.01)


# --- the OpenAI SDK contract -----------------------------------------------------------


async def test_sdk_non_streaming(stack: Stack) -> None:
    completion = await stack.sdk().chat.completions.create(
        model="fake", messages=USER, extra_body=fake()
    )
    assert completion.choices[0].message.content == "lorem ipsum dolor sit "
    assert completion.choices[0].finish_reason == "stop"
    assert completion.model == "fake-chat"
    assert completion.id.startswith("chatcmpl-")
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 4
    assert completion.usage.total_tokens == completion.usage.prompt_tokens + 4


async def test_sdk_streaming_with_usage(stack: Stack) -> None:
    stream = await stack.sdk().chat.completions.create(
        model="fake",
        messages=USER,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=fake(),
    )
    chunks = [chunk async for chunk in stream]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "lorem ipsum dolor sit "
    assert chunks[0].choices[0].delta.role == "assistant"
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 4
    assert len({chunk.id for chunk in chunks}) == 1


async def test_streaming_without_include_usage_sends_no_usage_chunk(stack: Stack) -> None:
    stream = await stack.sdk().chat.completions.create(
        model="fake", messages=USER, stream=True, extra_body=fake()
    )
    chunks = [chunk async for chunk in stream]
    assert all(chunk.usage is None for chunk in chunks)
    assert all(chunk.choices for chunk in chunks)


@pytest.mark.parametrize("stream", [False, True])
async def test_sdk_tool_calls(stack: Stack, stream: bool) -> None:
    client = stack.sdk()
    body = fake(scenario="tool_call")
    if stream:
        response = await client.chat.completions.create(
            model="fake",
            messages=USER,
            tools=TOOLS,
            stream=True,
            extra_body=body,
        )
        chunks = [chunk async for chunk in response]
        calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
        assert calls[0].id is not None
        assert calls[0].function is not None
        assert calls[0].function.name == "get_weather"
        arguments = "".join(tc.function.arguments or "" for tc in calls if tc.function)
        finish = [c.choices[0].finish_reason for c in chunks if c.choices][-1]
    else:
        completion = await client.chat.completions.create(
            model="fake",
            messages=USER,
            tools=TOOLS,
            extra_body=body,
        )
        message = completion.choices[0].message
        assert message.tool_calls is not None
        call = message.tool_calls[0]
        assert call.type == "function"
        assert call.function.name == "get_weather"
        arguments = call.function.arguments
        finish = completion.choices[0].finish_reason
    assert json.loads(arguments) == {"location": "Paris", "unit": "celsius"}
    assert finish == "tool_calls"


async def test_upstream_request_is_rewritten_for_openai(stack: Stack) -> None:
    await stack.sdk().chat.completions.create(
        model="fake",
        messages=USER,
        max_tokens=33,
        temperature=0.5,
        extra_body={**fake(), "some_new_openai_param": {"x": 1}},
    )
    sent = stack.fake.stats.recent[-1]
    body = sent["body"]
    assert body["model"] == "fake-chat"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_completion_tokens"] == 33
    assert "max_tokens" not in body
    assert body["temperature"] == 0.5
    assert body["some_new_openai_param"] == {"x": 1}  # unknown fields pass through
    assert sent["headers"]["authorization"] == "Bearer sk-upstream-test"
    assert sent["headers"]["x-request-id"]


async def test_model_default_max_tokens_applies_when_client_sets_none(stack: Stack) -> None:
    await stack.sdk().chat.completions.create(model="fake", messages=USER, extra_body=fake())
    assert stack.fake.stats.recent[-1]["body"]["max_completion_tokens"] == 64


async def test_concrete_model_name_routes_directly(stack: Stack) -> None:
    async with stack.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            json={"model": "fake/fake-chat", "messages": USER, **fake()},
        )
    assert response.status_code == 200
    assert response.headers["x-switchboard-model"] == "fake/fake-chat"
    assert response.headers["x-switchboard-provider"] == "fake"
    assert response.headers["x-switchboard-attempts"] == "1"


# --- failures before the commit point become HTTP errors ------------------------------


@pytest.mark.parametrize(
    ("upstream", "status", "code"),
    [
        ({"status": 503}, 502, "upstream_error"),
        ({"status": 500}, 502, "upstream_error"),
        ({"status": 401}, 502, "upstream_auth_failed"),
        ({"status": 429, "retry_after": 7}, 429, "upstream_rate_limited"),
        ({"status": 429, "error_code": "insufficient_quota"}, 503, "upstream_quota_exhausted"),
        ({"status": 400}, 400, "upstream_bad_request"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_http_errors_map_to_client_errors(
    stack: Stack, upstream: dict[str, Any], status: int, code: str, stream: bool
) -> None:
    async with stack.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            json={
                "model": "fake",
                "messages": USER,
                "stream": stream,
                **fake(fault="status", **upstream),
            },
        )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.headers["x-switchboard-provider"] == "fake"
    if "retry_after" in upstream:
        assert response.headers["retry-after"] == "7"


async def test_upstream_auth_failure_message_does_not_leak_provider_details(stack: Stack) -> None:
    async with stack.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            json={"model": "fake", "messages": USER, **fake(fault="status", status=401)},
        )
    assert "fake 401" not in response.text  # the provider's own message stays in the logs


async def test_error_event_before_any_content_is_still_an_http_error(stack: Stack) -> None:
    async with stack.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            json={
                "model": "fake",
                "messages": USER,
                "stream": True,
                **fake(fault="mid_stream_error", fault_after=0),
            },
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


async def test_first_token_timeout_returns_504_and_cancels_upstream(
    make_stack: StackFactory,
) -> None:
    async with make_stack(defaults="defaults: { timeouts: { first_token: 300ms } }") as stack:
        async with stack.http() as http:
            response = await http.post(
                "/v1/chat/completions",
                json={"model": "fake", "messages": USER, "stream": True, **fake(ttft_ms=5000)},
            )
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "upstream_timeout"
        await wait_for(lambda: stack.fake.stats.client_disconnects == 1)


async def test_connection_refused_is_502(make_stack: StackFactory) -> None:
    # Nothing listens on the discard port, so the connection is refused.
    providers = "  down: { type: openai, base_url: 'http://127.0.0.1:9/v1' }"
    models = "  down/m: { upstream_model: m }"
    async with make_stack(providers=providers, models=models) as stack, stack.http() as http:
        response = await http.post(
            "/v1/chat/completions", json={"model": "down/m", "messages": USER}
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unavailable"


# --- failures after the commit point become in-stream error events ---------------------


@pytest.mark.parametrize("fault", ["mid_stream_error", "disconnect"])
async def test_failure_after_first_token_raises_in_sdk(stack: Stack, fault: str) -> None:
    stream = await stack.sdk().chat.completions.create(
        model="fake", messages=USER, stream=True, extra_body=fake(fault=fault, fault_after=2)
    )
    received: list[str] = []

    async def consume() -> None:
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                received.append(chunk.choices[0].delta.content)

    with pytest.raises(openai.APIError, match="upstream provider"):
        await consume()
    assert received == ["lorem ", "ipsum "]


async def test_total_timeout_mid_stream_ends_with_error_event(make_stack: StackFactory) -> None:
    defaults = "defaults: { timeouts: { connect: 500ms, first_token: 500ms, total: 1s } }"
    async with make_stack(defaults=defaults) as stack:
        stream = await stack.sdk().chat.completions.create(
            model="fake",
            messages=USER,
            stream=True,
            extra_body=fake(interval_ms=300, chunks=20),
        )
        with pytest.raises(openai.APIError, match="timed out"):
            async for _ in stream:
                pass
        await wait_for(lambda: stack.fake.stats.client_disconnects == 1)


async def test_idle_timeout_mid_stream_ends_with_error_event(make_stack: StackFactory) -> None:
    defaults = "defaults: { timeouts: { first_token: 300ms, idle: 300ms } }"
    async with make_stack(defaults=defaults) as stack:
        stream = await stack.sdk().chat.completions.create(
            model="fake", messages=USER, stream=True, extra_body=fake(interval_ms=1500, chunks=3)
        )
        with pytest.raises(openai.APIError, match="timed out"):
            async for _ in stream:
                pass


# --- client disconnects cancel the upstream call ---------------------------------------


@pytest.mark.parametrize("during", ["first_token_wait", "streaming"])
async def test_client_disconnect_cancels_upstream(stack: Stack, during: str) -> None:
    behavior = (
        fake(ttft_ms=5000) if during == "first_token_wait" else fake(interval_ms=200, chunks=50)
    )
    labels = {"alias": "fake", "provider": "fake", "model": "fake/fake-chat"}
    before = _requests(**labels, outcome="client_disconnect")

    async with stack.http() as http:
        request = http.build_request(
            "POST",
            "/v1/chat/completions",
            json={"model": "fake", "messages": USER, "stream": True, **behavior},
        )
        if during == "streaming":
            response = await http.send(request, stream=True)
            async for _ in response.aiter_lines():
                break  # read one line, then hang up
            await response.aclose()
        else:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.3):
                    await http.send(request)

    await wait_for(lambda: stack.fake.stats.client_disconnects == 1)
    await wait_for(lambda: _requests(**labels, outcome="client_disconnect") == before + 1)


def _requests(**labels: str) -> float:
    total = 0.0
    for metric in REGISTRY.collect():
        if metric.name != "switchboard_requests":
            continue
        for sample in metric.samples:
            if sample.name.endswith("_total") and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                total += sample.value
    return total


# --- checks that fail before any upstream call ----------------------------------------


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({}, "missing_api_key"),
        ({"authorization": "Bearer wrong"}, "invalid_api_key"),
        ({"authorization": "Basic sb-test-key"}, "invalid_api_key"),
    ],
)
async def test_api_key_is_required(stack: Stack, headers: dict[str, str], code: str) -> None:
    async with stack.http() as http:
        http.headers.pop("authorization")
        response = await http.post(
            "/v1/chat/completions", json={"model": "fake", "messages": USER}, headers=headers
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == code
    assert stack.fake.stats.requests["/v1/chat/completions"] == 0


async def test_without_a_configured_key_every_request_is_rejected(make_stack: StackFactory) -> None:
    async with make_stack(dev_api_key=None) as stack, stack.http() as http:
        response = await http.post("/v1/chat/completions", json={"model": "fake", "messages": USER})
    assert response.status_code == 401


@pytest.mark.parametrize(
    ("body", "status", "param", "code"),
    [
        ({"messages": USER}, 400, "model", None),
        ({"model": "fake"}, 400, "messages", None),
        ({"model": "fake", "messages": []}, 400, "messages", None),
        (
            {"model": "fake", "messages": [{"role": "robot", "content": "x"}]},
            400,
            "messages.0.role",
            None,
        ),
        ({"model": "fake", "messages": USER, "n": 2}, 400, "n", "unsupported_parameter"),
        (
            {"model": "fake", "messages": USER, "logprobs": True},
            400,
            "logprobs",
            "unsupported_parameter",
        ),
        ({"model": "nope", "messages": USER}, 404, "model", "model_not_found"),
    ],
)
async def test_invalid_requests(
    stack: Stack, body: dict[str, Any], status: int, param: str, code: str | None
) -> None:
    async with stack.http() as http:
        response = await http.post("/v1/chat/completions", json=body)
    assert response.status_code == status
    error = response.json()["error"]
    assert error["param"] == param
    assert error["code"] == code


async def test_invalid_json_is_400(stack: Stack) -> None:
    async with stack.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


@pytest.mark.parametrize("chunked", [False, True])
async def test_oversized_body_is_413(make_stack: StackFactory, chunked: bool) -> None:
    async with make_stack(max_request_bytes=2048) as stack:
        payload = orjson.dumps(
            {"model": "fake", "messages": [{"role": "user", "content": "x" * 4096}]}
        )

        async def body() -> Any:
            yield payload  # no content-length: the limit is enforced while reading

        async with stack.http() as http:
            response = await http.post(
                "/v1/chat/completions", content=body() if chunked else payload
            )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


# --- observability ---------------------------------------------------------------------


async def test_one_access_log_line_per_request_without_prompt_content(stack: Stack) -> None:
    sink = io.StringIO()
    configure_logging("info", stream=sink)
    secret_prompt = "PROMPT-CONTENT-MUST-NOT-BE-LOGGED"
    stream = await stack.sdk().chat.completions.create(
        model="fake",
        messages=[{"role": "user", "content": secret_prompt}],  # inline literal: typed by mypy
        stream=True,
        stream_options={"include_usage": True},
        extra_body=fake(),
    )
    async for _ in stream:
        pass
    await wait_for(lambda: "http.request" in sink.getvalue())
    lines = [json.loads(line) for line in sink.getvalue().splitlines() if '"http.request"' in line]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["status"] == 200
    assert entry["alias"] == "fake"
    assert entry["model"] == "fake/fake-chat"
    assert entry["outcome"] == "success"
    assert entry["completion_tokens"] == 4
    assert entry["prompt_tokens"] > 0
    assert entry["finish_reason"] == "stop"
    assert entry["ttft_ms"] >= 0
    assert secret_prompt not in sink.getvalue()


async def test_metrics_record_requests_ttft_and_overhead(stack: Stack) -> None:
    labels = {"alias": "fake", "provider": "fake", "model": "fake/fake-chat"}
    ttft_before = REGISTRY.get_sample_value("switchboard_ttft_seconds_count", labels) or 0
    pre_before = (
        REGISTRY.get_sample_value(
            "switchboard_gateway_overhead_seconds_count", {"phase": "pre_upstream"}
        )
        or 0
    )
    ok_before = _requests(**labels, status="200", outcome="success")

    stream = await stack.sdk().chat.completions.create(
        model="fake", messages=USER, stream=True, extra_body=fake()
    )
    async for _ in stream:
        pass

    await wait_for(lambda: _requests(**labels, status="200", outcome="success") == ok_before + 1)
    assert REGISTRY.get_sample_value("switchboard_ttft_seconds_count", labels) == ttft_before + 1
    pre_after = REGISTRY.get_sample_value(
        "switchboard_gateway_overhead_seconds_count", {"phase": "pre_upstream"}
    )
    assert pre_after == pre_before + 1
    assert REGISTRY.get_sample_value("switchboard_inflight_requests") == 0
