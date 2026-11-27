"""Against the running compose stack (`make up`, then `make test-integration`).

The stack's `fake` alias routes to the fake-provider container, so these run without
any provider keys.
"""

import os

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> openai.OpenAI:
    key = os.environ.get("SWITCHBOARD_DEV_API_KEY")
    if not key:
        pytest.skip("SWITCHBOARD_DEV_API_KEY is not set (run via `make test-integration`)")
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
