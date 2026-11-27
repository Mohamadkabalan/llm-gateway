"""The OpenAI adapter's pure parts: chunk mapping, usage, request body, error classification.

The chunk fixtures follow the documented Chat Completions streaming format
(source: docs). Phase 2 adds fixtures recorded from the real API.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import orjson
import pytest

from switchboard.providers.errors import ErrorKind, parse_retry_after
from switchboard.providers.openai import (
    build_body,
    chunk_events,
    classify_response,
    normalize_usage,
)
from switchboard.routing.config import Capabilities, ModelDefaults, ModelKind, Timeouts
from switchboard.routing.table import ModelTarget
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    RefusalDelta,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStart,
    Usage,
)

TARGET = ModelTarget(
    name="openai/gpt-x",
    provider="openai",
    upstream_model="gpt-x-2026",
    kind=ModelKind.CHAT,
    capabilities=Capabilities(),
    defaults=ModelDefaults(max_tokens=1000),
    timeouts=Timeouts(),
)


def chunk(
    delta: Mapping[str, object], finish: str | None = None, **extra: object
) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gpt-x-2026",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }


def test_text_and_role_chunks() -> None:
    assert chunk_events(chunk({"role": "assistant", "content": ""})) == []
    assert chunk_events(chunk({"content": "Hi"})) == [TextDelta("Hi")]


def test_finish_reasons() -> None:
    assert chunk_events(chunk({}, "stop")) == [Finish(FinishReason.STOP)]
    assert chunk_events(chunk({}, "length")) == [Finish(FinishReason.LENGTH)]
    assert chunk_events(chunk({}, "function_call")) == [Finish(FinishReason.TOOL_CALLS)]
    assert chunk_events(chunk({}, "something_new")) == [Finish(FinishReason.STOP)]


def test_tool_call_deltas() -> None:
    start = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_a",
                "type": "function",
                "function": {"name": "f", "arguments": ""},
            },
            {
                "index": 1,
                "id": "call_b",
                "type": "function",
                "function": {"name": "g", "arguments": "{"},
            },
        ]
    }
    assert chunk_events(chunk(start)) == [
        ToolCallStart(0, "call_a", "f"),
        ToolCallStart(1, "call_b", "g"),
        ToolCallArgsDelta(1, "{"),
    ]
    more = {"tool_calls": [{"index": 1, "function": {"arguments": '"x": 1}'}}]}
    assert chunk_events(chunk(more)) == [ToolCallArgsDelta(1, '"x": 1}')]


def test_refusal_and_reasoning() -> None:
    assert chunk_events(chunk({"refusal": "I can't"})) == [RefusalDelta("I can't")]
    assert chunk_events(chunk({"reasoning_content": "hmm"})) == [ReasoningDelta("hmm")]


def test_final_usage_chunk_has_empty_choices() -> None:
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    events = chunk_events({"id": "chatcmpl-1", "choices": [], "usage": usage})
    assert events == [Usage(input=20, cache_read=100, output=30, reasoning=10)]


def test_usage_without_details() -> None:
    assert normalize_usage({"prompt_tokens": 5, "completion_tokens": 2}) == Usage(input=5, output=2)


def test_body_forwards_client_fields_and_forces_streaming_with_usage() -> None:
    request = ChatRequest.model_validate(
        {
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "temperature": 0.2,
            "stream": False,
            "stream_options": {"include_usage": False, "future_option": True},
            "metadata": {"a": "b"},
        }
    )
    body = build_body(request, TARGET)
    assert body["model"] == "gpt-x-2026"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True, "future_option": True}
    assert body["max_completion_tokens"] == 50
    assert "max_tokens" not in body
    assert body["temperature"] == 0.2
    assert body["metadata"] == {"a": "b"}
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_provider_options_merge_only_this_providers_entry() -> None:
    request = ChatRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"a": 1},
            "provider_options": {
                "openai": {"seed": 7, "metadata": {"b": 2}, "stream": False},
                "anthropic": {"thinking": {"type": "adaptive"}},
            },
        }
    )
    body = build_body(request, TARGET)
    assert "provider_options" not in body  # never forwarded: OpenAI rejects unknown fields
    assert body["seed"] == 7
    assert body["metadata"] == {"a": 1, "b": 2}  # dicts merge one level deep
    assert "thinking" not in body  # another provider's options are dropped
    assert body["stream"] is True  # the gateway's own fields win
    ollama = build_body(request, TARGET, options_key="ollama")
    assert "seed" not in ollama
    assert "provider_options" not in ollama


def test_body_prefers_max_completion_tokens_and_falls_back_to_model_default() -> None:
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    both = ChatRequest.model_validate({**base, "max_tokens": 5, "max_completion_tokens": 7})
    assert build_body(both, TARGET)["max_completion_tokens"] == 7
    assert build_body(ChatRequest.model_validate(base), TARGET)["max_completion_tokens"] == 1000


def error(code: str | None = None, message: str = "nope") -> bytes:
    return orjson.dumps({"error": {"message": message, "type": "x", "code": code}})


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (401, error("invalid_api_key"), ErrorKind.AUTH),
        (403, error(), ErrorKind.AUTH),
        (429, error("rate_limit_exceeded"), ErrorKind.RATE_LIMITED),
        (429, error("insufficient_quota"), ErrorKind.QUOTA),
        (400, error("context_length_exceeded"), ErrorKind.CONTEXT_LENGTH),
        (400, error("invalid_value"), ErrorKind.BAD_REQUEST),
        (404, error("model_not_found"), ErrorKind.BAD_REQUEST),
        (408, b"", ErrorKind.TIMEOUT),
        (409, error(), ErrorKind.OVERLOADED),
        (500, b"<html>oops</html>", ErrorKind.OVERLOADED),
        (503, error(), ErrorKind.OVERLOADED),
    ],
)
def test_classification(status: int, body: bytes, kind: ErrorKind) -> None:
    assert classify_response(status, body, {}).kind is kind


def test_classification_keeps_message_and_retry_hints() -> None:
    exc = classify_response(
        429,
        error("rate_limit_exceeded", "slow down"),
        {"retry-after": "3", "x-should-retry": "false"},
    )
    assert exc.message == "slow down"
    assert exc.status == 429
    assert exc.retry_after == 3.0
    assert exc.should_retry is False


def test_client_message_hides_provider_details_except_for_request_errors() -> None:
    assert classify_response(401, error(message="key sk-abc"), {}).client_message.startswith(
        "The gateway"
    )
    bad = classify_response(400, error("invalid_value", "messages[0] is invalid"), {})
    assert bad.client_message == "messages[0] is invalid"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, None),
        ({"retry-after": "2.5"}, 2.5),
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),
        ({"retry-after": "-4"}, 0.0),
        ({"retry-after": "soon"}, None),
    ],
)
def test_retry_after(headers: dict[str, str], expected: float | None) -> None:
    assert parse_retry_after(headers) == expected


def test_retry_after_http_date() -> None:
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    seconds = parse_retry_after({"retry-after": when})
    assert seconds is not None
    assert 25 <= seconds <= 30
