from typing import Any

import orjson
import pytest

from switchboard.schemas.encode import DONE_FRAME, Aggregator, ChunkEncoder, error_frame
from switchboard.schemas.events import (
    Finish,
    FinishReason,
    MessageStart,
    ReasoningDelta,
    RefusalDelta,
    StreamEvent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStart,
    Usage,
)

USAGE = Usage(input=80, cache_read=20, output=12, reasoning=5)
EVENTS: list[StreamEvent] = [
    MessageStart(upstream_id="up-1", served_model="served-model-2026"),
    TextDelta("Hel"),
    TextDelta('lo "world"\n'),
    ReasoningDelta("hidden"),
    Finish(FinishReason.STOP),
    USAGE,
]


def frames(data: bytes) -> list[Any]:
    """Parse SSE frames back into JSON (``[DONE]`` kept as a string)."""
    out: list[Any] = []
    for block in data.split(b"\n\n"):
        if not block:
            continue
        assert block.startswith(b"data: ")
        payload = block.removeprefix(b"data: ")
        out.append("[DONE]" if payload == b"[DONE]" else orjson.loads(payload))
    return out


def encode(events: list[StreamEvent], *, include_usage: bool) -> list[Any]:
    encoder = ChunkEncoder(
        completion_id="chatcmpl-x",
        created=1700000000,
        model="fallback",
        include_usage=include_usage,
    )
    return frames(b"".join(encoder.encode(e) for e in events) + encoder.end())


def test_stream_chunks_are_openai_shaped() -> None:
    chunks = encode(EVENTS, include_usage=True)
    assert chunks[0] == {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "served-model-2026",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    texts = [c["choices"][0]["delta"].get("content") for c in chunks[1:3]]
    assert texts == ["Hel", 'lo "world"\n']  # JSON-escaped correctly
    assert chunks[3]["choices"][0] == {
        "index": 0,
        "delta": {},
        "logprobs": None,
        "finish_reason": "stop",
    }
    assert chunks[4]["choices"] == []
    assert chunks[4]["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "total_tokens": 112,
        "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    assert chunks[-1] == "[DONE]"
    assert len(chunks) == 6  # reasoning is not forwarded


def test_usage_chunk_only_when_requested() -> None:
    chunks = encode(EVENTS, include_usage=False)
    assert all("usage" not in c for c in chunks if isinstance(c, dict))
    assert chunks[-1] == "[DONE]"


def test_tool_call_and_refusal_deltas() -> None:
    chunks = encode(
        [
            MessageStart(None, None),
            ToolCallStart(0, "call_1", "lookup"),
            ToolCallArgsDelta(0, '{"q": '),
            ToolCallArgsDelta(0, '"x"}'),
            RefusalDelta("no"),
            Finish(FinishReason.TOOL_CALLS),
        ],
        include_usage=False,
    )
    assert chunks[0]["model"] == "fallback"  # no served model: keep the configured one
    deltas = [c["choices"][0]["delta"] for c in chunks[1:5]]
    assert deltas[0] == {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": ""},
            }
        ]
    }
    assert deltas[1] == {"tool_calls": [{"index": 0, "function": {"arguments": '{"q": '}}]}
    assert deltas[3] == {"refusal": "no"}
    assert chunks[5]["choices"][0]["finish_reason"] == "tool_calls"


def test_encoder_remembers_usage_and_finish_reason() -> None:
    encoder = ChunkEncoder(completion_id="c", created=0, model="m", include_usage=False)
    for event in EVENTS:
        encoder.encode(event)
    assert encoder.usage == USAGE
    assert encoder.finish_reason is FinishReason.STOP


def test_aggregator_builds_a_chat_completion() -> None:
    aggregator = Aggregator(completion_id="chatcmpl-x", created=1, model="fallback")
    for event in EVENTS:
        aggregator.add(event)
    result = aggregator.result()
    assert result["object"] == "chat.completion"
    assert result["model"] == "served-model-2026"
    assert result["choices"][0]["message"] == {
        "role": "assistant",
        "content": 'Hello "world"\n',
        "refusal": None,
    }
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 112


def test_aggregator_tool_calls_in_index_order() -> None:
    aggregator = Aggregator(completion_id="c", created=1, model="m")
    events: list[StreamEvent] = [
        ToolCallStart(1, "b", "second"),
        ToolCallStart(0, "a", "first"),
        ToolCallArgsDelta(0, "{}"),
        ToolCallArgsDelta(1, '{"x":'),
        ToolCallArgsDelta(1, "1}"),
        ToolCallArgsDelta(5, "orphan"),  # no start: ignored rather than crashing
        Finish(FinishReason.TOOL_CALLS),
    ]
    for event in events:
        aggregator.add(event)
    message = aggregator.result()["choices"][0]["message"]
    assert message["content"] is None
    assert [(c["id"], c["function"]["arguments"]) for c in message["tool_calls"]] == [
        ("a", "{}"),
        ("b", '{"x":1}'),
    ]
    assert "usage" not in aggregator.result()


@pytest.mark.parametrize("code", [None, "upstream_error"])
def test_error_frame_is_what_the_openai_sdk_expects(code: str | None) -> None:
    (body,) = frames(error_frame("boom", "server_error", code))
    assert body == {
        "error": {"message": "boom", "type": "server_error", "param": None, "code": code}
    }


def test_done_frame() -> None:
    assert DONE_FRAME == b"data: [DONE]\n\n"
