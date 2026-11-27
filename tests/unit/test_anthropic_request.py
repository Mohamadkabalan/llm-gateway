"""OpenAI → Anthropic request translation (PLAN.md §3.3), one behaviour per test."""

from typing import Any

import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st

from switchboard.providers.anthropic_request import Passback, Translation, build_request
from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.routing.config import Capabilities, ModelDefaults, ModelKind, Timeouts
from switchboard.routing.table import ModelTarget
from switchboard.schemas.chat import ChatRequest

TARGET = ModelTarget(
    name="anthropic/claude-sonnet-5-5",
    provider="anthropic",
    upstream_model="claude-sonnet-5-5",
    kind=ModelKind.CHAT,
    capabilities=Capabilities(max_output_tokens=128000),
    defaults=ModelDefaults(max_tokens=8192),
    timeouts=Timeouts(),
)
USER = {"role": "user", "content": "hi"}


def translate(
    api_version: str = "2026-10-21",
    policy: Any = "drop",
    passback: Passback | None = None,
    **fields: Any,
) -> Translation:
    request = ChatRequest.model_validate({"model": "smart", "messages": [USER], **fields})
    return build_request(
        request, TARGET, api_version=api_version, unsupported_params=policy, passback=passback
    )


def body(**fields: Any) -> dict[str, Any]:
    return translate(**fields).body


def invalid(**fields: Any) -> ProviderError:
    with pytest.raises(ProviderError) as caught:
        translate(**fields)
    assert caught.value.kind is ErrorKind.INVALID_REQUEST
    return caught.value


def test_minimal_request() -> None:
    assert body() == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "stream": True,
    }


def test_max_tokens_precedence() -> None:
    assert body(max_completion_tokens=10, max_tokens=20)["max_tokens"] == 10
    assert body(max_tokens=20)["max_tokens"] == 20


def test_leading_system_and_developer_messages_become_system() -> None:
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "developer", "content": [{"type": "text", "text": "Use metric units."}]},
        USER,
    ]
    result = body(messages=messages)
    assert result["system"] == [
        {"type": "text", "text": "Be terse."},
        {"type": "text", "text": "Use metric units."},
    ]
    assert [m["role"] for m in result["messages"]] == ["user"]


def test_mid_conversation_system_stays_in_place_where_allowed() -> None:
    messages = [
        USER,
        {"role": "system", "content": "Now answer in French."},
        {"role": "assistant", "content": "Oui."},
        USER,
    ]
    roles = [m["role"] for m in body(messages=messages)["messages"]]
    assert roles == ["user", "system", "assistant", "user"]


def test_mid_conversation_system_in_a_disallowed_place_becomes_user_text() -> None:
    # After an assistant turn is not allowed, so it joins the next user turn as text.
    messages = [
        USER,
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": "Be brief."},
        USER,
    ]
    result = body(messages=messages)["messages"]
    assert [m["role"] for m in result] == ["user", "assistant", "user"]
    assert result[2]["content"] == [
        {"type": "text", "text": "[system] Be brief."},
        {"type": "text", "text": "hi"},
    ]


def test_images_base64_and_url() -> None:
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]
    blocks = body(messages=[{"role": "user", "content": content}])["messages"][0]["content"]
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
    }
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
    }


def test_unsupported_image_type_and_audio_are_rejected() -> None:
    bmp = {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,AAAA"}}
    assert invalid(messages=[{"role": "user", "content": [bmp]}]).param == "messages[0].content[0]"
    audio = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
    invalid(messages=[{"role": "user", "content": [audio]}])


TOOL_LOOP: list[dict[str, Any]] = [
    {"role": "user", "content": "Weather in Paris and Rome?"},
    {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "Rome"}'},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
    {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "21C"}]},
]


def test_tool_loop_round_trip() -> None:
    messages = body(messages=TOOL_LOOP)["messages"]
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Paris"}},
            {"type": "tool_use", "id": "call_2", "name": "weather", "input": {"city": "Rome"}},
        ],
    }
    # Consecutive tool messages become one user turn of tool_result blocks.
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "18C"},
            {"type": "tool_result", "tool_use_id": "call_2", "content": "21C"},
        ],
    }


def test_invalid_tool_arguments_name_the_call() -> None:
    bad = [
        USER,
        {
            **TOOL_LOOP[1],
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c", "content": "x"},
    ]
    assert invalid(messages=bad).param == "messages[1].tool_calls[0].function.arguments"


def test_prefill_is_rejected_before_sending() -> None:
    error = invalid(messages=[USER, {"role": "assistant", "content": "Sure, here"}])
    assert "prefill" in error.message


WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        "strict": True,
    },
}


def test_tools_and_tool_choice() -> None:
    result = body(tools=[WEATHER])
    assert result["tools"] == [
        {
            "name": "weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            "description": "Current weather",
            "strict": True,
        }
    ]
    assert "tool_choice" not in result  # auto is the API default
    assert body(tools=[WEATHER], tool_choice="none")["tool_choice"] == {"type": "none"}
    assert body(tools=[WEATHER], tool_choice="required")["tool_choice"] == {"type": "any"}
    named = {"type": "function", "function": {"name": "weather"}}
    assert body(tools=[WEATHER], tool_choice=named)["tool_choice"] == {
        "type": "tool",
        "name": "weather",
    }
    assert body(tools=[WEATHER], parallel_tool_calls=False)["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_output_config_effort_and_json_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }
    result = body(
        reasoning_effort="minimal",
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
    )
    assert result["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert body(reasoning_effort="xhigh")["output_config"] == {"effort": "xhigh"}
    assert "output_config" not in body(response_format={"type": "text"})
    invalid(response_format={"type": "json_object"})
    invalid(reasoning_effort="extreme")


def test_stop_and_user() -> None:
    result = body(stop="END", user="user-42")
    assert result["stop_sequences"] == ["END"]
    assert result["metadata"] == {"user_id": "user-42"}
    assert body(stop=["a", "b"])["stop_sequences"] == ["a", "b"]


def test_sampling_parameters_are_dropped_for_current_api_version() -> None:
    result = translate(temperature=0.3, top_p=0.9, seed=7, presence_penalty=0.1)
    assert result.dropped == ["presence_penalty", "seed", "temperature", "top_p"]
    assert not {"temperature", "top_p", "seed"} & set(result.body)


def test_sampling_parameters_pass_to_older_api_versions_clamped_to_one() -> None:
    result = translate(api_version="2023-06-01", temperature=1.5, top_p=0.5)
    assert result.body["temperature"] == 1.0
    assert result.body["top_p"] == 0.5
    assert result.dropped == []


def test_error_policy_rejects_instead_of_dropping() -> None:
    error = invalid(policy="error", temperature=0.2)
    assert error.param == "temperature"
    assert error.client_error.code == "unsupported_parameter"


def test_provider_options_escape_hatch_merges_last() -> None:
    result = body(
        reasoning_effort="high",
        provider_options={
            "anthropic": {
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "max"},
            }
        },
    )
    assert result["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert result["output_config"] == {"effort": "max"}
    assert "provider_options" not in result


def test_passback_replaces_the_translated_assistant_turn() -> None:
    stored: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Paris"}},
    ]
    messages = TOOL_LOOP[:2] + TOOL_LOOP[2:3]
    messages[1] = {**TOOL_LOOP[1], "tool_calls": TOOL_LOOP[1]["tool_calls"][:1]}
    result = body(messages=messages, passback=Passback(message_index=1, content=stored))
    assert result["messages"][1] == {"role": "assistant", "content": stored}


MESSAGE = st.fixed_dictionaries(
    {
        "role": st.sampled_from(["system", "developer", "user", "assistant"]),
        "content": st.text(min_size=1, max_size=40),
    }
)


@given(st.lists(MESSAGE, min_size=1, max_size=12))
def test_translation_is_deterministic(messages: list[dict[str, str]]) -> None:
    """The thinking sidecar relies on byte-identical prefixes (ADR-0009)."""
    history = [*messages, USER]
    try:
        first = orjson.dumps(body(messages=history, temperature=0.5))
    except ProviderError:
        return  # some random histories are invalid; determinism is about valid ones
    assert orjson.dumps(body(messages=history, temperature=0.5)) == first
