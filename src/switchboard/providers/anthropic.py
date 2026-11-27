"""Anthropic Messages adapter (PLAN.md §3.3, §3.4).

Requests are translated by ``anthropic_request``; this module sends them, turns the
Messages stream back into normalized events, classifies errors, and runs the thinking
sidecar for tool loops. Stream event shapes were checked against Anthropic's streaming
docs on 2026-11-27; unknown event types are ignored, as Anthropic's versioning policy
asks.
"""

import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import orjson
import structlog

from switchboard.providers.anthropic_request import Passback, build_request
from switchboard.providers.base import CallContext, CallTimings, EmbeddingResult
from switchboard.providers.errors import (
    ErrorKind,
    ProviderError,
    parse_retry_after,
    parse_should_retry,
)
from switchboard.providers.http import UpstreamClient, UpstreamResponse
from switchboard.providers.sse import SSEDecoder
from switchboard.providers.thinking import PASSBACK, ThinkingSidecar, matches
from switchboard.routing.config import ProviderConfig
from switchboard.routing.table import ModelTarget
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.embeddings import EmbeddingRequest
from switchboard.schemas.events import (
    CONTENT_EVENTS,
    Finish,
    FinishReason,
    MessageStart,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStart,
    Usage,
)

DEFAULT_API_VERSION = "2026-10-21"
_ERROR_BODY_LIMIT = 64 * 1024
_STOP_REASONS = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "pause_turn": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
    "refusal": FinishReason.CONTENT_FILTER,
}
_QUOTA_CODES = frozenset({"enforced_spend_limit_reached", "contract_spend_limit_reached"})
_SPEND_LIMIT_400 = "You have reached your specified API usage limits"

_log = structlog.get_logger("switchboard.providers.anthropic")


class AnthropicAdapter:
    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        *,
        api_key: str | None,
        sidecar: ThinkingSidecar,
    ) -> None:
        self.name = name
        self.api_version = config.api_version or DEFAULT_API_VERSION
        self.unsupported_params = config.unsupported_params
        self.thinking_passback = config.thinking_passback
        self._sidecar = sidecar
        headers = {"content-type": "application/json", "anthropic-version": self.api_version}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = UpstreamClient(str(config.base_url), config.pool, headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(
        self, request: EmbeddingRequest, target: ModelTarget, ctx: CallContext
    ) -> EmbeddingResult:
        raise ProviderError(
            ErrorKind.INVALID_REQUEST, "Anthropic has no embeddings API.", param="model"
        )

    async def stream_chat(
        self, request: ChatRequest, target: ModelTarget, ctx: CallContext
    ) -> AsyncGenerator[StreamEvent]:
        passback = await self._passback(request, ctx) if self.thinking_passback else None
        translation = build_request(
            request,
            target,
            api_version=self.api_version,
            unsupported_params=self.unsupported_params,
            passback=passback,
        )
        ctx.dropped_params.extend(translation.dropped)
        ctx.timings.upstream_start = time.perf_counter()
        async with self._client.stream(
            "/v1/messages",
            body=orjson.dumps(translation.body),
            headers={"accept": "text/event-stream", "x-request-id": ctx.request_id},
            timeouts=ctx.timeouts,
        ) as response:
            if response.status >= 400:
                raise classify_response(
                    response.status, await response.read_capped(_ERROR_BODY_LIMIT), response.headers
                )
            stream = AnthropicStream()
            async for event in _events(response, stream, ctx.timings):
                yield event
        if self.thinking_passback and (saved := stream.passback_content()) is not None:
            tool_ids, content = saved
            self._sidecar.store(ctx.scope, tool_ids, content)

    async def _passback(self, request: ChatRequest, ctx: CallContext) -> Passback | None:
        """The stored thinking for the open tool turn, if the client's echo matches."""
        turn = open_tool_turn(request)
        if turn is None:
            return None
        index, text, calls = turn
        stored = await self._sidecar.load(ctx.scope, [call_id for call_id, _, _ in calls])
        if stored is None:
            PASSBACK.labels(result="miss").inc()
            return None
        if not matches(stored, text, calls):
            PASSBACK.labels(result="mismatch").inc()
            return None
        PASSBACK.labels(result="hit").inc()
        return Passback(message_index=index, content=stored)


def open_tool_turn(request: ChatRequest) -> tuple[int, str, list[tuple[str, str, Any]]] | None:
    """The last assistant turn, if it called tools and only tool results follow it:
    ``(message index, its text, [(call id, name, parsed arguments)])``."""
    messages = request.messages
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "tool":
            continue
        if message.role != "assistant" or not message.tool_calls or index == len(messages) - 1:
            return None
        calls = []
        for call in message.tool_calls:
            function = call.get("function") or {}
            try:
                arguments = orjson.loads(function.get("arguments") or "{}")
            except orjson.JSONDecodeError:
                return None
            calls.append((str(call.get("id")), str(function.get("name")), arguments))
        text = (
            message.content
            if isinstance(message.content, str)
            else "".join(
                str(part.get("text", ""))
                for part in message.content or ()
                if part.get("type") == "text"
            )
        )
        return index, text, calls
    return None


async def _events(
    response: UpstreamResponse, stream: AnthropicStream, timings: CallTimings
) -> AsyncGenerator[StreamEvent]:
    decoder = SSEDecoder()
    async for raw in response.iter_chunks():
        read_at = timings.last_read = time.perf_counter()
        for sse in decoder.feed(raw):
            try:
                data = orjson.loads(sse.data)
            except orjson.JSONDecodeError as exc:
                raise ProviderError(ErrorKind.PROTOCOL, f"invalid JSON in stream: {exc}") from exc
            if not isinstance(data, dict):
                raise ProviderError(ErrorKind.PROTOCOL, "stream event is not a JSON object")
            for event in stream.handle(data):
                if timings.first_content_read is None and isinstance(event, CONTENT_EVENTS):
                    timings.first_content_read = read_at
                yield event
    decoder.close()
    if not stream.finished:
        raise ProviderError(ErrorKind.PROTOCOL, "upstream stream ended before message_stop")


class AnthropicStream:
    """Turns Messages stream events into normalized events, one event at a time.

    It also keeps each content block whole (thinking with its signature, tool input as
    parsed JSON), so a tool-calling turn can be stored for the thinking sidecar.
    """

    def __init__(self) -> None:
        self.finished = False
        self.refusal_category: str | None = None
        self._tool_index: dict[int, int] = {}  # Anthropic block index → OpenAI tool index
        self._blocks: dict[int, dict[str, Any]] = {}
        self._json_parts: dict[int, list[str]] = {}
        self._usage: dict[str, Any] = {}
        self._stop_reason: str | None = None

    def handle(self, data: Mapping[str, Any]) -> list[StreamEvent]:  # noqa: PLR0911 (one branch per event type)
        kind = data.get("type")
        if kind == "content_block_delta":
            return self._delta(int(data.get("index", 0)), data.get("delta") or {})
        if kind == "content_block_start":
            return self._start(int(data.get("index", 0)), data.get("content_block") or {})
        if kind == "content_block_stop":
            self._stop(int(data.get("index", 0)))
            return []
        if kind == "message_start":
            message = data.get("message") or {}
            self._merge_usage(message.get("usage"))
            return [MessageStart(upstream_id=message.get("id"), served_model=message.get("model"))]
        if kind == "message_delta":
            delta = data.get("delta") or {}
            if delta.get("stop_reason"):
                self._stop_reason = delta["stop_reason"]
            details = delta.get("stop_details") or {}
            if details.get("type") == "refusal":
                self.refusal_category = details.get("category")
                _log.info("anthropic.refusal", category=self.refusal_category)
            self._merge_usage(data.get("usage"))
            return []
        if kind == "message_stop":
            self.finished = True
            reason = _STOP_REASONS.get(self._stop_reason or "", FinishReason.STOP)
            return [Finish(reason), self.usage()]
        if kind == "error":
            raise stream_error(data.get("error") or {})
        return []  # ping, and event types added after this was written

    def _start(self, index: int, block: Mapping[str, Any]) -> list[StreamEvent]:
        kind = block.get("type")
        if kind == "text":
            self._blocks[index] = {"type": "text", "text": block.get("text") or ""}
            return [TextDelta(block["text"])] if block.get("text") else []
        if kind == "tool_use":
            tool_index = len(self._tool_index)
            self._tool_index[index] = tool_index
            self._blocks[index] = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
            }
            self._json_parts[index] = []
            events: list[StreamEvent] = [
                ToolCallStart(tool_index, str(block.get("id")), str(block.get("name")))
            ]
            if initial := block.get("input"):
                fragment = orjson.dumps(initial).decode()
                self._json_parts[index].append(fragment)
                events.append(ToolCallArgsDelta(tool_index, fragment))
            return events
        if kind == "thinking":
            self._blocks[index] = {
                "type": "thinking",
                "thinking": block.get("thinking") or "",
                "signature": block.get("signature") or "",
            }
            return []
        # redacted_thinking, server tool blocks and anything newer arrive whole: kept
        # verbatim for pass-back, not shown to the client.
        self._blocks[index] = dict(block)
        return []

    def _delta(self, index: int, delta: Mapping[str, Any]) -> list[StreamEvent]:
        kind = delta.get("type")
        block = self._blocks.get(index)
        if kind == "text_delta":
            text = str(delta.get("text") or "")
            if block is not None:
                block["text"] = block.get("text", "") + text
            return [TextDelta(text)] if text else []
        if kind == "input_json_delta":
            fragment = str(delta.get("partial_json") or "")
            if not fragment or index not in self._tool_index:
                return []
            self._json_parts[index].append(fragment)
            return [ToolCallArgsDelta(self._tool_index[index], fragment)]
        if kind == "thinking_delta":
            text = str(delta.get("thinking") or "")
            if block is not None:
                block["thinking"] = block.get("thinking", "") + text
            return [ReasoningDelta(text)] if text else []
        if kind == "signature_delta" and block is not None:
            block["signature"] = block.get("signature", "") + str(delta.get("signature") or "")
        elif kind == "citations_delta" and block is not None:
            block.setdefault("citations", []).append(delta.get("citation"))
        return []

    def _stop(self, index: int) -> None:
        if index in self._json_parts:
            raw = "".join(self._json_parts.pop(index)) or "{}"
            try:
                self._blocks[index]["input"] = orjson.loads(raw)
            except orjson.JSONDecodeError:
                self._blocks[index]["input"] = None  # not storable; see passback_content

    def _merge_usage(self, usage: Mapping[str, Any] | None) -> None:
        """``message_delta`` usage is cumulative; keep the latest non-null value."""
        for key, value in (usage or {}).items():
            if value is not None:
                self._usage[key] = value

    def usage(self) -> Usage:
        u = self._usage
        breakdown = u.get("cache_creation") or {}
        write_5m = breakdown.get("ephemeral_5m_input_tokens")
        write_1h = breakdown.get("ephemeral_1h_input_tokens")
        if write_5m is None and write_1h is None:
            write_5m, write_1h = u.get("cache_creation_input_tokens") or 0, 0
        return Usage(
            input=u.get("input_tokens") or 0,  # excludes cache reads and writes
            cache_read=u.get("cache_read_input_tokens") or 0,
            cache_write_5m=write_5m or 0,
            cache_write_1h=write_1h or 0,
            output=u.get("output_tokens") or 0,  # includes thinking
            reasoning=(u.get("output_tokens_details") or {}).get("thinking_tokens") or 0,
        )

    def passback_content(self) -> tuple[list[str], list[dict[str, Any]]] | None:
        """``(tool_use ids, content blocks)`` when this turn called tools and thought;
        None when there is nothing worth storing or a block could not be kept intact."""
        if self._stop_reason != "tool_use":
            return None
        blocks = [self._blocks[i] for i in sorted(self._blocks)]
        if not any(b.get("type") in ("thinking", "redacted_thinking") for b in blocks):
            return None
        content: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("type") == "text" and not block.get("text"):
                continue  # the API rejects empty text blocks
            if block.get("type") == "tool_use" and not isinstance(block.get("input"), dict):
                return None
            content.append(block)
        ids = [str(b["id"]) for b in content if b.get("type") == "tool_use"]
        return (ids, content) if ids else None


def classify_response(status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError:
    """Classify an Anthropic HTTP error (PLAN.md §2.2; Anthropic error docs)."""
    error = _error_object(body)
    message = str(error.get("message") or f"upstream returned HTTP {status}")
    code = str((error.get("details") or {}).get("error_code") or "")
    error_type = str(error.get("type") or "")
    if status in (401, 403):
        kind = ErrorKind.AUTH
    elif status == 402 or code in _QUOTA_CODES or message.startswith(_SPEND_LIMIT_400):
        kind = ErrorKind.QUOTA  # the gateway's own account is out of credit or capped
    elif status == 429:
        kind = ErrorKind.RATE_LIMITED
    elif status == 408:
        kind = ErrorKind.TIMEOUT
    elif status == 409 or status >= 500:
        kind = ErrorKind.OVERLOADED
    elif "prompt is too long" in message.lower():
        kind = ErrorKind.CONTEXT_LENGTH
    else:
        kind = ErrorKind.BAD_REQUEST
    return ProviderError(
        kind,
        f"{error_type}: {message}" if error_type else message,
        status=status,
        retry_after=parse_retry_after(headers),
        should_retry=parse_should_retry(headers.get("x-should-retry")),
    )


def stream_error(error: Mapping[str, Any]) -> ProviderError:
    """An ``event: error`` inside a 200 stream: ``details.error_code`` wins, then type."""
    code = str((error.get("details") or {}).get("error_code") or "")
    error_type = str(error.get("type") or "")
    if code in _QUOTA_CODES:
        kind = ErrorKind.QUOTA
    elif error_type == "rate_limit_error":
        kind = ErrorKind.RATE_LIMITED
    elif error_type == "invalid_request_error":
        kind = ErrorKind.BAD_REQUEST
    else:  # overloaded_error, api_error, timeout_error and unknown types
        kind = ErrorKind.OVERLOADED
    return ProviderError(kind, f"in-stream {error_type or 'error'}: {error.get('message')}")


def _error_object(body: bytes) -> dict[str, Any]:
    try:
        parsed = orjson.loads(body)
    except orjson.JSONDecodeError:
        return {}
    error = parsed.get("error") if isinstance(parsed, dict) else None
    return error if isinstance(error, dict) else {}
