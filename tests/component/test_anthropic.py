"""The OpenAI SDK through the gateway to the fake Anthropic Messages API.

These are the Phase 2 contract tests: streaming, aggregation, a full tool-call round
trip, thinking pass-back across that round trip, refusals and error mapping. The fake
rejects what the real API rejects (sampling parameters under 2026-10-21, prefill, a
missing max_tokens), so a translation slip shows up here as a 400.
"""

import json
from typing import Any

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from prometheus_client import REGISTRY

from tests.component.conftest import Stack, StackFactory

PROVIDER = "  claude: { type: anthropic, base_url: '{fake_url}' }"
MODELS = """\
  claude/sonnet:
    upstream_model: claude-sonnet-5-5
    capabilities: { tools: true, vision: true, forced_tool_choice: false, json_schema: true,
                    json_object: false, max_output_tokens: 128000 }
    defaults: { max_tokens: 1024 }
  claude/haiku:
    upstream_model: claude-haiku-5-5
    capabilities: { tools: true, vision: true, forced_tool_choice: true, json_object: false }
"""
USER: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "What's the weather in Paris?"}
]
WEATHER: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
    }
]


def fake(**behavior: Any) -> dict[str, Any]:
    """Fake behavior rides the escape hatch, merged into the Anthropic request body."""
    return {"provider_options": {"anthropic": {"fake": {"chunks": 4, **behavior}}}}


@pytest.fixture
async def claude(make_stack: StackFactory) -> Any:
    async with make_stack(providers=PROVIDER, models=MODELS) as stack:
        yield stack


def last_request(stack: Stack) -> dict[str, Any]:
    sent = stack.fake.stats.recent[-1]
    assert sent["path"] == "/v1/messages"
    return dict(sent["body"])


def passback(result: str) -> float:
    return REGISTRY.get_sample_value("switchboard_thinking_passback_total", {"result": result}) or 0


async def test_streaming_text(claude: Stack) -> None:
    stream = await claude.sdk().chat.completions.create(
        model="claude/sonnet",
        messages=USER,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=fake(),
    )
    chunks = [chunk async for chunk in stream]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "lorem ipsum dolor sit "
    assert chunks[0].model == "claude-sonnet-5-5"
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 4
    sent = claude.fake.stats.recent[-1]
    assert sent["headers"]["anthropic-version"] == "2026-10-21"
    assert last_request(claude)["max_tokens"] == 1024


async def test_non_streaming_is_aggregated(claude: Stack) -> None:
    completion = await claude.sdk().chat.completions.create(
        model="claude/sonnet", messages=USER, extra_body=fake()
    )
    assert completion.choices[0].message.content == "lorem ipsum dolor sit "
    assert completion.usage is not None
    assert completion.usage.prompt_tokens > 0


async def test_python_sdk_tool_call_round_trip(claude: Stack) -> None:
    client = claude.sdk()
    first = await client.chat.completions.create(
        model="claude/sonnet", messages=USER, tools=WEATHER, extra_body=fake(scenario="auto")
    )
    call = first.choices[0].message.tool_calls[0]  # type: ignore[index]
    assert first.choices[0].finish_reason == "tool_calls"
    assert call.function.name == "get_weather"  # type: ignore[union-attr]
    arguments = json.loads(call.function.arguments)  # type: ignore[union-attr]
    assert arguments == {"location": "Paris", "unit": "celsius"}

    history: list[ChatCompletionMessageParam] = [
        *USER,
        first.choices[0].message.model_dump(exclude_none=True),  # type: ignore[list-item]
        {"role": "tool", "tool_call_id": call.id, "content": "18C and sunny"},
    ]
    second = await client.chat.completions.create(
        model="claude/sonnet", messages=history, tools=WEATHER, extra_body=fake(scenario="auto")
    )
    assert second.choices[0].finish_reason == "stop"
    assert second.choices[0].message.content
    messages = last_request(claude)["messages"]
    assert messages[1]["content"] == [
        {"type": "tool_use", "id": call.id, "name": "get_weather", "input": arguments}
    ]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": call.id, "content": "18C and sunny"}
    ]


async def test_thinking_blocks_are_passed_back_across_the_tool_loop(claude: Stack) -> None:
    client = claude.sdk()
    hits = passback("hit")
    stream = await client.chat.completions.create(
        model="claude/sonnet",
        messages=USER,
        tools=WEATHER,
        stream=True,
        extra_body=fake(scenario="thinking_tool_call"),
    )
    chunks = [chunk async for chunk in stream]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
    call_id = calls[0].id
    arguments = "".join(tc.function.arguments or "" for tc in calls if tc.function)
    assistant: ChatCompletionMessageParam = {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": call_id or "",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments},
            }
        ],
    }
    history: list[ChatCompletionMessageParam] = [
        *USER,
        assistant,
        {"role": "tool", "tool_call_id": call_id or "", "content": "18C"},
    ]
    await client.chat.completions.create(
        model="claude/sonnet", messages=history, tools=WEATHER, extra_body=fake(scenario="auto")
    )
    turn = last_request(claude)["messages"][1]["content"]
    assert [block["type"] for block in turn] == ["thinking", "text", "tool_use"]
    assert turn[0]["signature"].startswith("sig-msg_fake_")
    assert passback("hit") == hits + 1


async def test_a_changed_history_is_sent_without_thinking(claude: Stack) -> None:
    client = claude.sdk()
    mismatches = passback("mismatch")
    first = await client.chat.completions.create(
        model="claude/sonnet",
        messages=USER,
        tools=WEATHER,
        extra_body=fake(scenario="thinking_tool_call"),
    )
    call = first.choices[0].message.tool_calls[0]  # type: ignore[index]
    edited = {"role": "assistant", "content": "Something else.", "tool_calls": [call.model_dump()]}
    history: list[Any] = [
        *USER,
        edited,
        {"role": "tool", "tool_call_id": call.id, "content": "18C"},
    ]
    await client.chat.completions.create(
        model="claude/sonnet", messages=history, tools=WEATHER, extra_body=fake(scenario="auto")
    )
    turn = last_request(claude)["messages"][1]["content"]
    assert [block["type"] for block in turn] == ["text", "tool_use"]
    assert passback("mismatch") == mismatches + 1


async def test_parallel_tool_calls(claude: Stack) -> None:
    tools: list[ChatCompletionToolParam] = [
        *WEATHER,
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}},
    ]
    completion = await claude.sdk().chat.completions.create(
        model="claude/sonnet",
        messages=USER,
        tools=tools,
        extra_body=fake(scenario="parallel_tool_calls"),
    )
    names = [c.function.name for c in completion.choices[0].message.tool_calls or []]  # type: ignore[union-attr]
    assert names == ["get_weather", "get_time"]


async def test_refusal_is_content_filter(claude: Stack) -> None:
    completion = await claude.sdk().chat.completions.create(
        model="claude/sonnet", messages=USER, extra_body=fake(scenario="refusal")
    )
    assert completion.choices[0].finish_reason == "content_filter"
    assert completion.choices[0].message.content is None


async def test_dropped_sampling_parameters_are_reported(claude: Stack) -> None:
    async with claude.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            json={
                "model": "claude/sonnet",
                "messages": USER,
                "temperature": 0.2,
                "seed": 1,
                **fake(),
            },
        )
    assert response.status_code == 200  # the fake would 400 if temperature reached it
    assert response.headers["x-switchboard-dropped-params"] == "seed,temperature"
    assert "temperature" not in last_request(claude)


async def test_forced_tool_choice_depends_on_the_model(claude: Stack) -> None:
    client = claude.sdk()
    with pytest.raises(openai.BadRequestError, match="forced tool_choice"):
        await client.chat.completions.create(
            model="claude/sonnet",
            messages=USER,
            tools=WEATHER,
            tool_choice="required",
            extra_body=fake(),
        )
    await client.chat.completions.create(
        model="claude/haiku",
        messages=USER,
        tools=WEATHER,
        tool_choice="required",
        extra_body=fake(scenario="auto"),
    )
    assert last_request(claude)["tool_choice"] == {"type": "any"}


async def test_prefill_is_rejected_before_the_upstream_call(claude: Stack) -> None:
    sent_before = claude.fake.stats.requests["/v1/messages"]
    with pytest.raises(openai.BadRequestError, match="prefill"):
        await claude.sdk().chat.completions.create(
            model="claude/sonnet", messages=[*USER, {"role": "assistant", "content": "Sure:"}]
        )
    assert claude.fake.stats.requests["/v1/messages"] == sent_before


@pytest.mark.parametrize(
    ("upstream", "status", "code"),
    [
        ({"status": 529}, 502, "upstream_error"),
        ({"status": 429, "retry_after": 3}, 429, "upstream_rate_limited"),
        (
            {"status": 429, "error_code": "enforced_spend_limit_reached"},
            503,
            "upstream_quota_exhausted",
        ),
        ({"status": 401}, 502, "upstream_auth_failed"),
        ({"status": 400}, 400, "upstream_bad_request"),
    ],
)
async def test_anthropic_errors_map_to_openai_errors(
    claude: Stack, upstream: dict[str, Any], status: int, code: str
) -> None:
    async with claude.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            json={
                "model": "claude/sonnet",
                "messages": USER,
                "stream": True,
                **fake(fault="status", **upstream),
            },
        )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


async def test_mid_stream_error_event_reaches_the_sdk(claude: Stack) -> None:
    stream = await claude.sdk().chat.completions.create(
        model="claude/sonnet",
        messages=USER,
        stream=True,
        extra_body=fake(fault="mid_stream_error", fault_after=2),
    )

    async def consume() -> None:
        async for _ in stream:
            pass

    with pytest.raises(openai.APIError, match="upstream provider"):
        await consume()
