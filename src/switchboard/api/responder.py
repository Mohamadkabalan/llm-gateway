"""Serving one chat completion: commit point, streaming, aggregation and disconnects.

The upstream call runs inside the response object rather than in the endpoint. That
gives the response two properties PLAN.md §2.1 needs:

* **Commit point.** The status line is held back until the first content event
  arrives, so an upstream failure before it is still returned as a normal HTTP error.
  (Phase 3 retries and falls back at exactly this point.) After it, a failure can only
  be reported as an in-stream error event.
* **Disconnects.** A watcher task listens for ``http.disconnect`` from the moment the
  response starts, including the wait for the first token. If the client goes away,
  the upstream call is cancelled, so the provider stops generating billable output.
  (uvicorn silently drops writes to a closed connection, so without the watcher a
  stream would run to completion for nobody.)
"""

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, MutableMapping

import orjson
from starlette.responses import Response
from starlette.types import Message, Receive, Scope, Send

from switchboard.api.errors import error_body
from switchboard.lifecycle import Lifecycle
from switchboard.observability.metrics import (
    INFLIGHT,
    OVERHEAD_FIRST_CHUNK,
    OVERHEAD_PRE_UPSTREAM,
    REQUEST_DURATION,
    REQUESTS,
    TTFT,
)
from switchboard.providers.base import CallContext, ProviderAdapter
from switchboard.providers.errors import CLIENT_FAULT_KINDS, ErrorKind, ProviderError
from switchboard.routing.table import RoutePlan
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.encode import Aggregator, ChunkEncoder, error_frame
from switchboard.schemas.events import FinishReason, StreamEvent, Usage, is_content

_SSE_HEADERS = (
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
    (b"x-accel-buffering", b"no"),  # stop nginx-style proxies from buffering the stream
)
_JSON = b"application/json"
_ERROR_DETAIL_LIMIT = 500
_CLIENT_CLOSED = 499  # nginx's convention for "client closed the request"


class ChatResponder(Response):
    def __init__(
        self,
        *,
        chat: ChatRequest,
        plan: RoutePlan,
        adapter: ProviderAdapter,
        ctx: CallContext,
        lifecycle: Lifecycle,
        received_at: float,
        log_fields: MutableMapping[str, object],
    ) -> None:
        # Starlette's Response.__init__ renders a body up front; this response has none
        # until the upstream call produces it, so only the attributes FastAPI reads are set.
        self.background = None
        self.status_code = 200
        self._chat = chat
        self._plan = plan
        self._target = plan.targets[0]
        self._adapter = adapter
        self._ctx = ctx
        self._lifecycle = lifecycle
        self._received_at = received_at
        self._log_fields = log_fields
        self._completion_id = f"chatcmpl-{uuid.uuid7().hex}"
        self._created = int(time.time())

        self._status = _CLIENT_CLOSED
        self._outcome = "success"
        self._started = False  # status line sent
        self._finished = False  # final body frame sent
        self._first_token_at: float | None = None
        self._usage: Usage | None = None
        self._finish: FinishReason | None = None
        self._error: ProviderError | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        with self._lifecycle.track(), INFLIGHT.track_inprogress():
            work = asyncio.create_task(self._respond(send))
            watcher = asyncio.create_task(_wait_for_disconnect(receive))
            completed = False
            try:
                await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
            finally:
                completed = work.done()
                if not completed:
                    disconnected = watcher.done() and not watcher.cancelled()
                    self._outcome = "client_disconnect" if disconnected else "cancelled"
                work.cancel()
                watcher.cancel()
                await asyncio.wait((work, watcher))
                self._record()
        if completed and not work.cancelled() and (exc := work.exception()) is not None:
            raise exc

    async def _respond(self, send: Send) -> None:
        timeouts = self._ctx.timeouts
        try:
            async with (
                asyncio.timeout(timeouts.total),
                contextlib.aclosing(
                    self._adapter.stream_chat(self._chat, self._target, self._ctx)
                ) as events,
            ):
                head = await self._until_first_content(events, timeouts.first_token)
                if self._chat.stream:
                    await self._stream(send, head, events)
                else:
                    await self._aggregate(send, head, events)
        except ProviderError as exc:
            await self._fail(send, exc)
        except TimeoutError:
            message = f"request exceeded the total timeout of {timeouts.total:g}s"
            await self._fail(send, ProviderError(ErrorKind.TIMEOUT, message))

    async def _until_first_content(
        self, events: AsyncIterator[StreamEvent], first_token: float
    ) -> list[StreamEvent]:
        """Buffer events up to and including the first content event (the commit point)."""
        head: list[StreamEvent] = []
        try:
            async with asyncio.timeout(first_token):
                async for event in events:
                    head.append(event)
                    if is_content(event):
                        break
        except TimeoutError:
            message = f"no first token within {first_token:g}s"
            raise ProviderError(ErrorKind.TIMEOUT, message) from None
        return head

    async def _stream(
        self, send: Send, head: list[StreamEvent], events: AsyncIterator[StreamEvent]
    ) -> None:
        encoder = ChunkEncoder(
            completion_id=self._completion_id,
            created=self._created,
            model=self._target.upstream_model,
            include_usage=self._chat.include_usage,
        )
        try:
            await send(self._start(200, _SSE_HEADERS))
            await send(_body(b"".join(encoder.encode(event) for event in head)))
            if any(is_content(event) for event in head):
                self._first_token_at = time.perf_counter()
            async for event in events:
                if frame := encoder.encode(event):
                    await send(_body(frame))
            await send(_body(encoder.end(), more=False))
            self._finished = True
        finally:
            self._usage, self._finish = encoder.usage, encoder.finish_reason

    async def _aggregate(
        self, send: Send, head: list[StreamEvent], events: AsyncIterator[StreamEvent]
    ) -> None:
        aggregator = Aggregator(
            completion_id=self._completion_id,
            created=self._created,
            model=self._target.upstream_model,
        )
        for event in head:
            aggregator.add(event)
        async for event in events:
            aggregator.add(event)
        self._usage, self._finish = aggregator.usage, aggregator.finish_reason
        body = orjson.dumps(aggregator.result())
        await send(self._start(200, ((b"content-type", _JSON),), len(body)))
        await send(_body(body, more=False))
        self._finished = True

    async def _fail(self, send: Send, exc: ProviderError) -> None:
        self._error = exc
        self._outcome = "client_error" if exc.kind in CLIENT_FAULT_KINDS else "upstream_error"
        client = exc.client_error
        if not self._started:
            headers = [(b"content-type", _JSON)]
            if exc.kind is ErrorKind.RATE_LIMITED and exc.retry_after is not None:
                headers.append((b"retry-after", str(max(1, round(exc.retry_after))).encode()))
            message, error_type = exc.client_message, client.error_type
            body = orjson.dumps(error_body(message, error_type, param=exc.param, code=client.code))
            await send(self._start(client.status, headers, len(body)))
            await send(_body(body, more=False))
        elif not self._finished:
            frame = error_frame(exc.client_message, client.error_type, client.code)
            await send(_body(frame, more=False))
        self._finished = True

    def _start(
        self,
        status: int,
        headers: tuple[tuple[bytes, bytes], ...] | list[tuple[bytes, bytes]],
        content_length: int | None = None,
    ) -> Message:
        self._started = True
        self._status = status
        raw = [
            *headers,
            (b"x-switchboard-provider", self._target.provider.encode()),
            (b"x-switchboard-model", self._target.name.encode()),
            (b"x-switchboard-attempts", str(self._ctx.attempt).encode()),
        ]
        if self._ctx.dropped_params:
            dropped = ",".join(sorted(set(self._ctx.dropped_params)))
            raw.append((b"x-switchboard-dropped-params", dropped.encode()))
        if content_length is not None:
            raw.append((b"content-length", str(content_length).encode()))
        return {"type": "http.response.start", "status": status, "headers": raw}

    def _record(self) -> None:
        """Metrics and the request's log fields. Runs once, however the request ended."""
        now = time.perf_counter()
        target = self._target
        labels = (self._plan.route_label, target.provider, target.name)
        REQUESTS.labels(*labels, str(self._status), self._outcome).inc()
        REQUEST_DURATION.labels(*labels, str(self._chat.stream).lower()).observe(
            now - self._received_at
        )
        timings = self._ctx.timings
        if timings.upstream_start is not None:
            OVERHEAD_PRE_UPSTREAM.observe(timings.upstream_start - self._received_at)
        if self._first_token_at is not None:
            TTFT.labels(*labels).observe(self._first_token_at - self._received_at)
            if timings.first_content_read is not None:
                OVERHEAD_FIRST_CHUNK.observe(self._first_token_at - timings.first_content_read)

        fields = self._log_fields
        fields.update(
            status=self._status,
            alias=self._plan.alias,
            provider=target.provider,
            model=target.name,
            stream=self._chat.stream,
            outcome=self._outcome,
        )
        if self._first_token_at is not None:
            fields["ttft_ms"] = round((self._first_token_at - self._received_at) * 1000, 2)
        if self._usage is not None:
            fields["prompt_tokens"] = self._usage.prompt_total
            fields["completion_tokens"] = self._usage.output
        if self._finish is not None:
            fields["finish_reason"] = self._finish.value
        if self._error is not None:
            fields["error_kind"] = self._error.kind.value
            fields["error_detail"] = self._error.message[:_ERROR_DETAIL_LIMIT]
            if self._error.status is not None:
                fields["upstream_status"] = self._error.status


def _body(data: bytes, *, more: bool = True) -> Message:
    return {"type": "http.response.body", "body": data, "more_body": more}


async def _wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass
