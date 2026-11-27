"""OpenAI wire format for the fake provider, pre-serialized where it matters.

A benchmark run sends tens of thousands of chunks per second, so a stream's frames
are byte concatenations of a per-stream prefix and pre-encoded words. The fake must
never be the bottleneck; the direct baseline run checks that it isn't.
"""

import base64
import hashlib
import random
import struct
from typing import Any

import orjson

from fakeprovider.behavior import Behavior
from fakeprovider.plan import (
    REFUSAL_TEXT,
    TOOL_ARGUMENTS,
    WORDS,
    Completion,
    Scenario,
    StreamPlan,
    resolve,
)

_WORD_JSON = tuple(orjson.dumps(word + " ") for word in WORDS)
_OPEN_SUFFIX = b',"logprobs":null,"finish_reason":null}]}\n\n'
DONE = b"data: [DONE]\n\n"
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    429: "rate_limit_exceeded",
}


class StreamFrames:
    """The frames of one ``chat.completion.chunk`` stream."""

    def __init__(self, completion_id: str, created: int, model: str) -> None:
        self._head: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }
        self._prefix = b"".join(
            (
                b'data: {"id":',
                orjson.dumps(completion_id),
                b',"object":"chat.completion.chunk","created":',
                str(created).encode(),
                b',"model":',
                orjson.dumps(model),
                b',"choices":[{"index":0,"delta":',
            )
        )

    def role(self) -> bytes:
        return self._prefix + b'{"role":"assistant","content":""}' + _OPEN_SUFFIX

    def word(self, index: int) -> bytes:
        word = _WORD_JSON[index % len(_WORD_JSON)]
        return self._prefix + b'{"content":' + word + b"}" + _OPEN_SUFFIX

    def refusal(self, text: str) -> bytes:
        return self._prefix + b'{"refusal":' + orjson.dumps(text) + b"}" + _OPEN_SUFFIX

    def tool_start(self, index: int, call_id: str, name: str) -> bytes:
        call: dict[str, Any] = {"index": index, "id": call_id, "type": "function"}
        call["function"] = {"name": name, "arguments": ""}
        return self._prefix + orjson.dumps({"tool_calls": [call]}) + _OPEN_SUFFIX

    def tool_args(self, index: int, fragment: str) -> bytes:
        call = {"index": index, "function": {"arguments": fragment}}
        return self._prefix + orjson.dumps({"tool_calls": [call]}) + _OPEN_SUFFIX

    def finish(self, reason: str) -> bytes:
        return b"".join(
            (self._prefix, b'{},"logprobs":null,"finish_reason":"', reason.encode(), b'"}]}\n\n')
        )

    def usage(self, prompt_tokens: int, completion_tokens: int) -> bytes:
        chunk = {**self._head, "choices": [], "usage": usage(prompt_tokens, completion_tokens)}
        return b"data: " + orjson.dumps(chunk) + b"\n\n"


class OpenAIDialect:
    path = "/v1/chat/completions"
    id_prefix = "chatcmpl-fake-"

    def validate(self, payload: dict[str, Any], headers: dict[str, str]) -> str | None:
        return None

    def error_body(self, status: int, message: str, code: str | None) -> bytes:
        return error_body(message, _ERROR_TYPES.get(status, "server_error"), code)

    def resolve_scenario(self, behavior: Behavior, payload: dict[str, Any]) -> Scenario:
        messages = payload.get("messages") or [{}]
        return resolve(
            behavior,
            after_tool_result=messages[-1].get("role") == "tool",
            has_tools=bool(payload.get("tools")),
        )

    def tool_names(self, payload: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            str((tool.get("function") or {}).get("name"))
            for tool in payload.get("tools") or ()
            if isinstance(tool, dict) and (tool.get("function") or {}).get("name")
        )

    def include_usage(self, payload: dict[str, Any]) -> bool:
        return bool((payload.get("stream_options") or {}).get("include_usage"))

    def stream_plan(self, behavior: Behavior, c: Completion) -> StreamPlan:
        frames = StreamFrames(c.completion_id, c.created, c.model)
        pieces: list[bytes] = []
        for k, name in enumerate(c.tools_called):
            pieces.append(frames.tool_start(k, c.call_id(k), name))
            pieces += [frames.tool_args(k, fragment) for fragment in TOOL_ARGUMENTS]
        if c.tools_called:
            finish = "tool_calls"
        elif c.scenario == "refusal":
            pieces, finish = [frames.refusal(REFUSAL_TEXT)], "stop"
        else:
            pieces = [frames.word(i) for i in range(behavior.chunks)]
            finish = "length" if c.scenario == "max_tokens" else "stop"
        tail = (b"" if pieces else frames.role()) + frames.finish(finish)
        if c.include_usage:
            tail += frames.usage(c.prompt_tokens, completion_tokens(behavior, c))
        return StreamPlan(
            head=b"",
            pieces=pieces,
            tail=tail + DONE,
            error=error_frame("fake mid-stream error"),
            first_prefix=frames.role(),
        )

    def response_body(self, behavior: Behavior, c: Completion) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": None, "refusal": None}
        finish = "stop"
        if c.tools_called:
            message["tool_calls"] = [
                {
                    "id": c.call_id(k),
                    "type": "function",
                    "function": {"name": name, "arguments": "".join(TOOL_ARGUMENTS)},
                }
                for k, name in enumerate(c.tools_called)
            ]
            finish = "tool_calls"
        elif c.scenario == "refusal":
            message["refusal"] = REFUSAL_TEXT
        else:
            message["content"] = text_for(behavior.chunks)
            finish = "length" if c.scenario == "max_tokens" else "stop"
        choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
        return {
            "id": c.completion_id,
            "object": "chat.completion",
            "created": c.created,
            "model": c.model,
            "choices": [choice],
            "usage": usage(c.prompt_tokens, completion_tokens(behavior, c)),
        }


def completion_tokens(behavior: Behavior, c: Completion) -> int:
    if c.tools_called:
        return len(TOOL_ARGUMENTS) * len(c.tools_called)
    if c.scenario == "refusal":
        return 6
    return behavior.chunks


def usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def text_for(chunks: int) -> str:
    return "".join(WORDS[i % len(WORDS)] + " " for i in range(chunks))


def error_body(message: str, error_type: str, code: str | None) -> bytes:
    error = {"message": message, "type": error_type, "param": None, "code": code}
    return orjson.dumps({"error": error})


def error_frame(message: str) -> bytes:
    return b"data: " + error_body(message, "server_error", None) + b"\n\n"


def embedding(text: str, dimensions: int) -> list[float]:
    """Deterministic unit vector derived from the text (same text → same vector)."""
    rng = random.Random(hashlib.sha256(text.encode()).digest())  # noqa: S311 (not crypto)
    vector = [rng.gauss(0, 1) for _ in range(dimensions)]
    norm = sum(x * x for x in vector) ** 0.5 or 1.0
    return [x / norm for x in vector]


def encode_embedding(vector: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
    return vector
