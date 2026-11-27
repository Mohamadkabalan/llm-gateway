from typing import Any

import orjson
import pytest

from switchboard.providers.anthropic import classify_response, open_tool_turn, stream_error
from switchboard.providers.errors import ErrorKind
from switchboard.providers.thinking import MemoryThinkingSidecar, key, matches
from switchboard.schemas.chat import ChatRequest


def error(error_type: str, message: str = "x", code: str | None = None) -> bytes:
    body: dict[str, Any] = {"type": error_type, "message": message}
    if code:
        body["details"] = {"error_code": code}
    return orjson.dumps({"type": "error", "error": body, "request_id": "req_1"})


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (400, error("invalid_request_error", "messages: field required"), ErrorKind.BAD_REQUEST),
        (
            400,
            error("invalid_request_error", "prompt is too long: 1200000 tokens"),
            ErrorKind.CONTEXT_LENGTH,
        ),
        # Your own spend limit is a 400 on Anthropic, not a 429 (errors docs, 2026-11-27).
        (
            400,
            error("invalid_request_error", "You have reached your specified API usage limits."),
            ErrorKind.QUOTA,
        ),
        (401, error("authentication_error"), ErrorKind.AUTH),
        (402, error("billing_error"), ErrorKind.QUOTA),
        (403, error("permission_error"), ErrorKind.AUTH),
        (404, error("not_found_error"), ErrorKind.BAD_REQUEST),
        (413, error("request_too_large"), ErrorKind.BAD_REQUEST),
        (429, error("rate_limit_error"), ErrorKind.RATE_LIMITED),
        (429, error("rate_limit_error", code="enforced_spend_limit_reached"), ErrorKind.QUOTA),
        (429, error("rate_limit_error", code="contract_spend_limit_reached"), ErrorKind.QUOTA),
        (429, error("rate_limit_error", code="ramp_limit_exceeded"), ErrorKind.RATE_LIMITED),
        (500, error("api_error"), ErrorKind.OVERLOADED),
        (503, error("overloaded_error"), ErrorKind.OVERLOADED),
        (504, error("timeout_error"), ErrorKind.OVERLOADED),
        (529, error("overloaded_error"), ErrorKind.OVERLOADED),
        (502, b"<html>bad gateway</html>", ErrorKind.OVERLOADED),
    ],
)
def test_http_error_classification(status: int, body: bytes, kind: ErrorKind) -> None:
    assert classify_response(status, body, {}).kind is kind


def test_retry_hints_are_kept() -> None:
    exc = classify_response(
        529, error("overloaded_error"), {"retry-after": "2", "x-should-retry": "true"}
    )
    assert exc.retry_after == 2.0
    assert exc.should_retry is True
    assert exc.message == "overloaded_error: x"


@pytest.mark.parametrize(
    ("event", "kind"),
    [
        ({"type": "overloaded_error", "message": "Overloaded"}, ErrorKind.OVERLOADED),
        ({"type": "api_error"}, ErrorKind.OVERLOADED),
        ({"type": "rate_limit_error"}, ErrorKind.RATE_LIMITED),
        (
            {"type": "rate_limit_error", "details": {"error_code": "enforced_spend_limit_reached"}},
            ErrorKind.QUOTA,
        ),
        ({"type": "invalid_request_error"}, ErrorKind.BAD_REQUEST),
        ({"type": "something_new"}, ErrorKind.OVERLOADED),
    ],
)
def test_in_stream_error_classification(event: dict[str, Any], kind: ErrorKind) -> None:
    assert stream_error(event).kind is kind


def chat(messages: list[dict[str, Any]]) -> ChatRequest:
    return ChatRequest.model_validate({"model": "m", "messages": messages})


CALL: dict[str, Any] = {
    "id": "toolu_1",
    "type": "function",
    "function": {"name": "f", "arguments": '{"a": 1}'},
}


def test_open_tool_turn_is_the_last_assistant_turn_followed_by_tool_results() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "Calling.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"},
    ]
    assert open_tool_turn(chat(messages)) == (1, "Calling.", [("toolu_1", "f", {"a": 1})])


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "hi"}],
        # a new user message after the tool results closes the loop
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"},
            {"role": "user", "content": "thanks"},
        ],
        # no tool calls on the last assistant turn
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
    ],
)
def test_no_open_tool_turn(messages: list[dict[str, Any]]) -> None:
    assert open_tool_turn(chat(messages)) is None


STORED: list[dict[str, Any]] = [
    {"type": "thinking", "thinking": "", "signature": "sig"},
    {"type": "text", "text": "Calling."},
    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
]


def test_matches_requires_the_same_text_and_tool_calls() -> None:
    assert matches(STORED, "Calling.", [("toolu_1", "f", {"a": 1})])
    assert matches(STORED, " Calling. ", [("toolu_1", "f", {"a": 1})])  # whitespace is not content
    assert not matches(STORED, "Something else", [("toolu_1", "f", {"a": 1})])
    assert not matches(STORED, "Calling.", [("toolu_1", "f", {"a": 2})])
    assert not matches(STORED, "Calling.", [("toolu_1", "f", {"a": 1}), ("toolu_2", "f", {})])


def test_keys_are_scoped_and_order_independent() -> None:
    assert key("team-a", ["b", "a"]) == key("team-a", ["a", "b"])
    assert key("team-a", ["a"]) != key("team-b", ["a"])


async def test_memory_sidecar_round_trip_ttl_and_eviction() -> None:
    sidecar = MemoryThinkingSidecar(ttl_seconds=60, max_entries=2)
    sidecar.store("s", ["1"], STORED)
    assert await sidecar.load("s", ["1"]) == STORED
    assert await sidecar.load("other", ["1"]) is None
    sidecar.store("s", ["2"], STORED)
    sidecar.store("s", ["3"], STORED)  # evicts the oldest entry
    assert await sidecar.load("s", ["1"]) is None
    assert await sidecar.load("s", ["3"]) == STORED

    expired = MemoryThinkingSidecar(ttl_seconds=-1)
    expired.store("s", ["1"], STORED)
    assert await expired.load("s", ["1"]) is None
