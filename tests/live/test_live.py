"""Real Claude and real OpenAI through the gateway (PLAN.md §15.4). Catches API drift."""

import json

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from tests.live.conftest import require

pytestmark = pytest.mark.live

WEATHER: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
ASK: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Use the get_weather tool to check the weather in Paris."}
]
MODELS = {"anthropic": "anthropic/claude-sonnet-5-5", "openai": "openai/gpt-5.4-mini"}
KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
async def test_streams_text(gateway: openai.AsyncOpenAI, provider: str) -> None:
    require(KEYS[provider])
    stream = await gateway.chat.completions.create(
        model=MODELS[provider],
        messages=[{"role": "user", "content": "Reply with exactly: switchboard ok"}],
        stream=True,
        stream_options={"include_usage": True},
        max_completion_tokens=200,
    )
    chunks = [chunk async for chunk in stream]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert "switchboard ok" in text.lower()
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens > 0
    assert chunks[-1].usage.completion_tokens > 0


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
async def test_tool_call_round_trip(gateway: openai.AsyncOpenAI, provider: str) -> None:
    require(KEYS[provider])
    first = await gateway.chat.completions.create(
        model=MODELS[provider], messages=ASK, tools=WEATHER, max_completion_tokens=400
    )
    message = first.choices[0].message
    assert first.choices[0].finish_reason == "tool_calls"
    assert message.tool_calls
    call = message.tool_calls[0]
    assert call.type == "function"
    assert json.loads(call.function.arguments)["city"].lower().startswith("paris")

    history: list[ChatCompletionMessageParam] = [
        *ASK,
        message.model_dump(exclude_none=True),  # type: ignore[list-item]
        {"role": "tool", "tool_call_id": call.id, "content": "18C and sunny"},
    ]
    second = await gateway.chat.completions.create(
        model=MODELS[provider], messages=history, tools=WEATHER, max_completion_tokens=400
    )
    assert second.choices[0].finish_reason == "stop"
    assert "18" in (second.choices[0].message.content or "")


async def test_embeddings(gateway: openai.AsyncOpenAI) -> None:
    require("OPENAI_API_KEY")
    response = await gateway.embeddings.create(
        model="openai/text-embedding-3-small", input=["hello", "world"]
    )
    assert len(response.data) == 2
    assert len(response.data[0].embedding) == 1536
