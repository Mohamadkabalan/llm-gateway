"""OpenAI Chat Completions and Embeddings adapter (PLAN.md §3.5).

Near pass-through: the client's request is forwarded with its own fields, after
setting the upstream model name, forcing ``stream: true`` with usage, and replacing
the deprecated ``max_tokens`` with ``max_completion_tokens``. Subclasses adjust the
body for OpenAI-compatible servers with fewer features (``providers.ollama``).
"""

import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any

import orjson

from switchboard.providers.base import CallContext, CallTimings, EmbeddingResult
from switchboard.providers.errors import (
    ErrorKind,
    ProviderError,
    parse_retry_after,
    parse_should_retry,
)
from switchboard.providers.http import UpstreamClient, UpstreamResponse
from switchboard.providers.sse import SSEDecoder
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
    RefusalDelta,
    StreamEvent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStart,
    Usage,
)

_ERROR_BODY_LIMIT = 64 * 1024
# 2,048 inputs x 3,072 float dimensions as JSON is about 120 MB; base64 is a third of it.
_EMBEDDING_BODY_LIMIT = 256 * 1024 * 1024
_DONE = b"[DONE]"
_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}
# 429 error codes that mean "out of money", not "slow down" (PLAN.md §2.2).
_QUOTA_CODES = frozenset({"insufficient_quota", "billing_hard_limit_reached", "billing_not_active"})


class OpenAIAdapter:
    base_path = ""  # appended to the provider's base_url

    def __init__(self, name: str, config: ProviderConfig, *, api_key: str | None) -> None:
        self.name = name
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self.unsupported_params = config.unsupported_params
        base_url = str(config.base_url).rstrip("/") + self.base_path
        self._client = UpstreamClient(base_url, config.pool, headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    def prepare_chat(
        self, request: ChatRequest, target: ModelTarget, ctx: CallContext
    ) -> dict[str, Any]:
        return build_body(request, target)

    def prepare_embedding(
        self, request: EmbeddingRequest, target: ModelTarget, ctx: CallContext
    ) -> dict[str, Any]:
        body = request.to_upstream_dict()
        body["model"] = target.upstream_model
        return body

    async def embed(
        self, request: EmbeddingRequest, target: ModelTarget, ctx: CallContext
    ) -> EmbeddingResult:
        body = orjson.dumps(self.prepare_embedding(request, target, ctx))
        ctx.timings.upstream_start = time.perf_counter()
        status, headers, data = await self._client.request(
            "/embeddings",
            body=body,
            headers={"accept": "application/json", "x-request-id": ctx.request_id},
            timeouts=ctx.timeouts,
            max_bytes=_EMBEDDING_BODY_LIMIT,
        )
        if status >= 400:
            raise classify_response(status, data[:_ERROR_BODY_LIMIT], headers)
        try:
            parsed = orjson.loads(data)
        except orjson.JSONDecodeError as exc:
            raise ProviderError(ErrorKind.PROTOCOL, f"invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("data"), list):
            raise ProviderError(ErrorKind.PROTOCOL, "embeddings response has no 'data' list")
        usage = parsed.get("usage") or {}
        return EmbeddingResult(body=parsed, prompt_tokens=int(usage.get("prompt_tokens") or 0))

    async def stream_chat(
        self, request: ChatRequest, target: ModelTarget, ctx: CallContext
    ) -> AsyncGenerator[StreamEvent]:
        body = orjson.dumps(self.prepare_chat(request, target, ctx))
        ctx.timings.upstream_start = time.perf_counter()
        async with self._client.stream(
            "/chat/completions",
            body=body,
            headers={"accept": "text/event-stream", "x-request-id": ctx.request_id},
            timeouts=ctx.timeouts,
        ) as response:
            if response.status >= 400:
                raise classify_response(
                    response.status, await response.read_capped(_ERROR_BODY_LIMIT), response.headers
                )
            async for event in _stream_events(response, ctx.timings):
                yield event


def build_body(request: ChatRequest, target: ModelTarget) -> dict[str, Any]:
    body = request.to_upstream_dict()
    body["model"] = target.upstream_model
    body["stream"] = True
    body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    max_tokens = body.pop("max_tokens", None)
    if body.get("max_completion_tokens") is None:
        limit = max_tokens or target.defaults.max_tokens
        if limit is not None:
            body["max_completion_tokens"] = limit
    return body


async def _stream_events(
    response: UpstreamResponse, timings: CallTimings
) -> AsyncIterator[StreamEvent]:
    decoder = SSEDecoder()
    started = False
    done = False
    async for raw in response.iter_chunks():
        read_at = time.perf_counter()
        for sse in decoder.feed(raw):
            if done:
                continue  # keep reading to EOF so the connection goes back to the pool
            if sse.data == _DONE:
                done = True
                continue
            try:
                chunk = orjson.loads(sse.data)
            except orjson.JSONDecodeError as exc:
                raise ProviderError(ErrorKind.PROTOCOL, f"invalid JSON in stream: {exc}") from exc
            if not isinstance(chunk, dict):
                raise ProviderError(ErrorKind.PROTOCOL, "stream chunk is not a JSON object")
            if chunk.get("error") is not None:
                raise _stream_error(chunk["error"])
            if not started:
                started = True
                yield MessageStart(upstream_id=chunk.get("id"), served_model=chunk.get("model"))
            for event in chunk_events(chunk):
                if timings.first_content_read is None and isinstance(event, CONTENT_EVENTS):
                    timings.first_content_read = read_at
                yield event
    decoder.close()
    if not done:
        raise ProviderError(ErrorKind.PROTOCOL, "upstream stream ended before [DONE]")


def chunk_events(chunk: Mapping[str, Any]) -> list[StreamEvent]:
    """Map one ``chat.completion.chunk`` to normalized events (choice 0 only)."""
    events: list[StreamEvent] = []
    choices = chunk.get("choices")
    if choices:
        choice = choices[0]
        delta = choice.get("delta") or {}
        if text := delta.get("content"):
            events.append(TextDelta(text))
        if refusal := delta.get("refusal"):
            events.append(RefusalDelta(refusal))
        # Some OpenAI-compatible servers (vLLM, DeepSeek) stream reasoning separately.
        if reasoning := delta.get("reasoning_content") or delta.get("reasoning"):
            events.append(ReasoningDelta(reasoning))
        for call in delta.get("tool_calls") or ():
            index = call.get("index", 0)
            function = call.get("function") or {}
            if call.get("id") or function.get("name"):
                events.append(
                    ToolCallStart(
                        index, call.get("id") or f"call_{index}", function.get("name", "")
                    )
                )
            if arguments := function.get("arguments"):
                events.append(ToolCallArgsDelta(index, arguments))
        if finish := choice.get("finish_reason"):
            events.append(Finish(_FINISH_REASONS.get(finish, FinishReason.STOP)))
    if usage := chunk.get("usage"):
        events.append(normalize_usage(usage))
    return events


def normalize_usage(usage: Mapping[str, Any]) -> Usage:
    """OpenAI's ``prompt_tokens`` includes cached tokens; split them out."""
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return Usage(
        input=max(prompt - cached, 0),
        cache_read=cached,
        output=usage.get("completion_tokens") or 0,
        reasoning=reasoning,
    )


def classify_response(status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError:
    """Classify an HTTP error response (PLAN.md §2.2)."""
    error = _error_object(body)
    message = str(error.get("message") or f"upstream returned HTTP {status}")
    code = str(error.get("code") or error.get("type") or "")
    should_retry = parse_should_retry(headers.get("x-should-retry"))
    retry_after = parse_retry_after(headers)

    if status in (401, 403):
        kind = ErrorKind.AUTH
    elif status == 429:
        kind = ErrorKind.QUOTA if code in _QUOTA_CODES else ErrorKind.RATE_LIMITED
    elif status == 408:
        kind = ErrorKind.TIMEOUT
    elif status == 409 or status >= 500:
        kind = ErrorKind.OVERLOADED
    elif code == "context_length_exceeded":
        kind = ErrorKind.CONTEXT_LENGTH
    else:
        kind = ErrorKind.BAD_REQUEST
    return ProviderError(
        kind, message, status=status, retry_after=retry_after, should_retry=should_retry
    )


def _error_object(body: bytes) -> dict[str, Any]:
    try:
        parsed = orjson.loads(body)
    except orjson.JSONDecodeError:
        return {}
    error = parsed.get("error") if isinstance(parsed, dict) else None
    return error if isinstance(error, dict) else {}


def _stream_error(error: object) -> ProviderError:
    message = error.get("message") if isinstance(error, dict) else None
    return ProviderError(ErrorKind.OVERLOADED, f"in-stream error: {message or error}")
