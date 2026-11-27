"""Normalized events → OpenAI Chat Completions responses.

``ChunkEncoder`` writes ``chat.completion.chunk`` SSE frames for streaming clients and
``Aggregator`` builds one ``chat.completion`` object for non-streaming clients. Both
use the gateway's own completion id, so responses look the same whichever provider
served them.

The encoder is on the per-chunk hot path. Each text delta costs one ``orjson.dumps``
of the text plus a byte concatenation with a prefix built once per stream.
"""

from typing import Any

import orjson

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

DONE_FRAME = b"data: [DONE]\n\n"
_OPEN_CHOICE_SUFFIX = b',"logprobs":null,"finish_reason":null}]}\n\n'


def openai_usage(usage: Usage) -> dict[str, Any]:
    prompt = usage.prompt_total
    return {
        "prompt_tokens": prompt,
        "completion_tokens": usage.output,
        "total_tokens": prompt + usage.output,
        "prompt_tokens_details": {"cached_tokens": usage.cache_read},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning},
    }


def error_frame(message: str, error_type: str, code: str | None = None) -> bytes:
    """An in-stream error. The OpenAI SDK raises ``APIError`` when it reads one."""
    body = {"error": {"message": message, "type": error_type, "param": None, "code": code}}
    return b"data: " + orjson.dumps(body) + b"\n\n"


class ChunkEncoder:
    def __init__(
        self, *, completion_id: str, created: int, model: str, include_usage: bool
    ) -> None:
        self._id = completion_id
        self._created = created
        self._model = model
        self._include_usage = include_usage
        self._prefix = self._build_prefix()
        self.usage: Usage | None = None
        self.finish_reason: FinishReason | None = None

    def _build_prefix(self) -> bytes:
        return b"".join(
            (
                b'data: {"id":',
                orjson.dumps(self._id),
                b',"object":"chat.completion.chunk","created":',
                str(self._created).encode(),
                b',"model":',
                orjson.dumps(self._model),
                b',"choices":[{"index":0,"delta":',
            )
        )

    def _delta(self, delta: bytes) -> bytes:
        return self._prefix + delta + _OPEN_CHOICE_SUFFIX

    def encode(self, event: StreamEvent) -> bytes:  # noqa: PLR0911 (one return per event type)
        """Return the SSE bytes for one event (empty if the event is not forwarded)."""
        match event:
            case TextDelta(text=text):
                return self._delta(b'{"content":' + orjson.dumps(text) + b"}")
            case ToolCallArgsDelta(tool_index=index, fragment=fragment):
                call = {"index": index, "function": {"arguments": fragment}}
                return self._delta(orjson.dumps({"tool_calls": [call]}))
            case ToolCallStart(tool_index=index, call_id=call_id, name=name):
                call = {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
                return self._delta(orjson.dumps({"tool_calls": [call]}))
            case RefusalDelta(text=text):
                return self._delta(b'{"refusal":' + orjson.dumps(text) + b"}")
            case MessageStart(served_model=served_model):
                if served_model and served_model != self._model:
                    self._model = served_model
                    self._prefix = self._build_prefix()
                return self._delta(b'{"role":"assistant","content":""}')
            case Finish(reason=reason):
                self.finish_reason = reason
                return b"".join(
                    (
                        self._prefix,
                        b'{},"logprobs":null,"finish_reason":"',
                        reason.value.encode(),
                        b'"}]}\n\n',
                    )
                )
            case Usage():
                self.usage = event
                return b""
            case ReasoningDelta():
                return b""

    def end(self) -> bytes:
        """The usage chunk (only if the client asked for it), then ``[DONE]``."""
        if not (self._include_usage and self.usage is not None):
            return DONE_FRAME
        chunk = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [],
            "usage": openai_usage(self.usage),
        }
        return b"data: " + orjson.dumps(chunk) + b"\n\n" + DONE_FRAME


class Aggregator:
    """Collects a whole stream into one ``chat.completion`` response."""

    def __init__(self, *, completion_id: str, created: int, model: str) -> None:
        self._id = completion_id
        self._created = created
        self.model = model
        self._text: list[str] = []
        self._refusal: list[str] = []
        self._tool_calls: dict[int, tuple[str, str, list[str]]] = {}
        self.finish_reason: FinishReason | None = None
        self.usage: Usage | None = None

    def add(self, event: StreamEvent) -> None:
        match event:
            case TextDelta(text=text):
                self._text.append(text)
            case ToolCallArgsDelta(tool_index=index, fragment=fragment):
                if index in self._tool_calls:
                    self._tool_calls[index][2].append(fragment)
            case ToolCallStart(tool_index=index, call_id=call_id, name=name):
                self._tool_calls[index] = (call_id, name, [])
            case RefusalDelta(text=text):
                self._refusal.append(text)
            case MessageStart(served_model=served_model):
                self.model = served_model or self.model
            case Finish(reason=reason):
                self.finish_reason = reason
            case Usage():
                self.usage = event
            case ReasoningDelta():
                pass

    def result(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self._text) if self._text else None,
            "refusal": "".join(self._refusal) if self._refusal else None,
        }
        if self._tool_calls:
            message["tool_calls"] = [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "".join(fragments)},
                }
                for _, (call_id, name, fragments) in sorted(self._tool_calls.items())
            ]
        finish = self.finish_reason or FinishReason.STOP
        body: dict[str, Any] = {
            "id": self._id,
            "object": "chat.completion",
            "created": self._created,
            "model": self.model,
            "choices": [
                {"index": 0, "message": message, "logprobs": None, "finish_reason": finish.value}
            ],
        }
        if self.usage is not None:
            body["usage"] = openai_usage(self.usage)
        return body
