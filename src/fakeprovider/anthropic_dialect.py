"""Anthropic Messages wire format for the fake provider (``POST /v1/messages``).

Event shapes follow Anthropic's streaming docs (checked 2026-11-27, API version
2026-10-21): ``message_start``, then per block ``content_block_start``,
``content_block_delta``... and ``content_block_stop``, then ``message_delta`` with the
stop reason and cumulative usage, then ``message_stop``. Thinking blocks use
``display: "omitted"``, the default on current models: one empty ``thinking_delta``
and a ``signature_delta``. Like the real API, the fake rejects sampling parameters,
a trailing assistant message (prefill) and a missing ``max_tokens``.
"""

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

SAMPLING_REMOVED_IN = "2026-10-21"
_WORD_JSON = tuple(orjson.dumps(word + " ") for word in WORDS)
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    504: "timeout_error",
    529: "overloaded_error",
}
_STOP_REASONS = {"max_tokens": "max_tokens", "refusal": "refusal"}
_TEXT_DELTA_PREFIX = (
    b"event: content_block_delta\ndata: "
    b'{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":'
)


def frame(event: str, data: dict[str, Any]) -> bytes:
    return b"event: " + event.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"


def _block_start(index: int, block: dict[str, Any]) -> bytes:
    data = {"type": "content_block_start", "index": index, "content_block": block}
    return frame("content_block_start", data)


def _delta(index: int, delta: dict[str, Any]) -> bytes:
    return frame(
        "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
    )


def _block_stop(index: int) -> bytes:
    return frame("content_block_stop", {"type": "content_block_stop", "index": index})


class AnthropicDialect:
    path = "/v1/messages"
    id_prefix = "msg_fake_"

    def validate(self, payload: dict[str, Any], headers: dict[str, str]) -> str | None:
        version = headers.get("anthropic-version", "")
        if version >= SAMPLING_REMOVED_IN:
            for name in ("temperature", "top_p", "top_k"):
                if name in payload:
                    return f"{name}: Extra inputs are not permitted for this API version"
        if not isinstance(payload.get("max_tokens"), int):
            return "max_tokens: Field required"
        messages = payload.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            return "This model does not support assistant message prefill."
        return None

    def error_body(self, status: int, message: str, code: str | None) -> bytes:
        error: dict[str, Any] = {"type": _ERROR_TYPES.get(status, "api_error"), "message": message}
        if code:
            error["details"] = {"error_code": code}
        return orjson.dumps({"type": "error", "error": error, "request_id": "req_fake"})

    def resolve_scenario(self, behavior: Behavior, payload: dict[str, Any]) -> Scenario:
        messages = payload.get("messages") or [{}]
        last = messages[-1]
        content = last.get("content") if isinstance(last.get("content"), list) else []
        after_tool_result = last.get("role") == "user" and any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        )
        return resolve(
            behavior, after_tool_result=after_tool_result, has_tools=bool(payload.get("tools"))
        )

    def tool_names(self, payload: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(tool["name"])
            for tool in payload.get("tools") or ()
            if isinstance(tool, dict) and tool.get("name")
        )

    def include_usage(self, payload: dict[str, Any]) -> bool:
        return True

    def stream_plan(self, behavior: Behavior, c: Completion) -> StreamPlan:
        start = {
            "type": "message_start",
            "message": {
                "id": c.completion_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": c.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": c.prompt_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 1,
                },
            },
        }
        head = frame("message_start", start) + frame("ping", {"type": "ping"})
        pieces: list[bytes] = []
        index = 0
        if c.scenario == "thinking_tool_call":
            pieces.append(
                _block_start(index, {"type": "thinking", "thinking": "", "signature": ""})
                + _delta(index, {"type": "thinking_delta", "thinking": ""})
                + _delta(index, {"type": "signature_delta", "signature": f"sig-{c.completion_id}"})
                + _block_stop(index)
            )
            index += 1
            pieces.append(
                _block_start(index, {"type": "text", "text": ""})
                + _delta(index, {"type": "text_delta", "text": "Let me check."})
                + _block_stop(index)
            )
            index += 1
        for k, name in enumerate(c.tools_called):
            block = {"type": "tool_use", "id": c.call_id(k), "name": name, "input": {}}
            pieces.append(
                _block_start(index, block)
                + _delta(index, {"type": "input_json_delta", "partial_json": ""})
            )
            pieces += [
                _delta(index, {"type": "input_json_delta", "partial_json": f})
                for f in TOOL_ARGUMENTS
            ]
            pieces[-1] += _block_stop(index)
            index += 1
        if not c.tools_called and c.scenario != "refusal":
            text_prefix = _TEXT_DELTA_PREFIX
            pieces = [
                text_prefix + _WORD_JSON[i % len(_WORD_JSON)] + b"}}\n\n"
                for i in range(behavior.chunks)
            ]
            if pieces:
                pieces[0] = _block_start(0, {"type": "text", "text": ""}) + pieces[0]
                pieces[-1] += _block_stop(0)

        stop_reason = "tool_use" if c.tools_called else _STOP_REASONS.get(c.scenario, "end_turn")
        delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
        if c.scenario == "refusal":
            delta["stop_details"] = {
                "type": "refusal",
                "category": "general_harms",
                "explanation": REFUSAL_TEXT,
            }
        usage = {"output_tokens": _output_tokens(behavior, c)}
        tail = frame("message_delta", {"type": "message_delta", "delta": delta, "usage": usage})
        tail += frame("message_stop", {"type": "message_stop"})
        error = frame(
            "error",
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "fake mid-stream error"},
            },
        )
        return StreamPlan(head=head, pieces=pieces, tail=tail, error=error)

    def response_body(self, behavior: Behavior, c: Completion) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "tool_use",
                "id": c.call_id(k),
                "name": name,
                "input": orjson.loads("".join(TOOL_ARGUMENTS)),
            }
            for k, name in enumerate(c.tools_called)
        ]
        if not content and c.scenario != "refusal":
            content = [
                {
                    "type": "text",
                    "text": "".join(WORDS[i % len(WORDS)] + " " for i in range(behavior.chunks)),
                }
            ]
        stop_reason = "tool_use" if c.tools_called else _STOP_REASONS.get(c.scenario, "end_turn")
        return {
            "id": c.completion_id,
            "type": "message",
            "role": "assistant",
            "model": c.model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": c.prompt_tokens,
                "output_tokens": _output_tokens(behavior, c),
            },
        }


def _output_tokens(behavior: Behavior, c: Completion) -> int:
    if c.tools_called:
        return len(TOOL_ARGUMENTS) * len(c.tools_called)
    return 6 if c.scenario == "refusal" else behavior.chunks
