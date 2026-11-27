"""Serving one chat completion: failover, commit point, streaming and disconnects.

The upstream work runs inside the response object rather than in the endpoint. That
gives the response the properties PLAN.md §2.1 needs:

* **Failover before the commit point.** The executor walks the route's candidates
  (retries, fallbacks, hedging) until one produces its first content event. Nothing is
  sent before that, so a failure up to then is invisible to the client, and a total
  failure is still a normal HTTP error.
* **Keep-alive.** If a streaming request has no first token after ``keepalive_after``,
  the status line and SSE comments (``: ping``) go out every ``keepalive_after``, so
  proxies do not time out a slow-thinking model. Failover still works; only a total
  failure is then reported as an in-stream error event, and the routing headers
  (provider, model, attempts) are not sent, because they are not known yet.
* **After the commit point**, a failure can only be an in-stream error event. The idle
  timeout (``timeouts.idle``) bounds the gap between upstream reads.
* **Disconnects.** A watcher task listens for ``http.disconnect`` from the moment the
  response starts. If the client goes away, the work is cancelled, which closes every
  upstream connection, so providers stop generating billable output. (uvicorn
  silently drops writes to a closed connection, so without the watcher a stream would
  run to completion for nobody.)
"""

import asyncio
import time
import uuid
from collections.abc import Mapping, MutableMapping

import orjson
from starlette.responses import Response
from starlette.types import Message, Receive, Scope, Send

from switchboard.accounting.ledger import RequestLedger, Served
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
from switchboard.providers.base import ProviderAdapter
from switchboard.providers.errors import CLIENT_FAULT_KINDS, ErrorKind, ProviderError
from switchboard.reliability.executor import Committed, Executor, ExhaustedError, OpenStream, Run
from switchboard.reliability.idle import idle_guard
from switchboard.routing.table import ModelTarget, RoutePlan
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.encode import Aggregator, ChunkEncoder, error_frame
from switchboard.schemas.events import FinishReason, Usage, is_content

_SSE_HEADERS = (
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
    (b"x-accel-buffering", b"no"),  # stop nginx-style proxies from buffering the stream
)
_JSON = b"application/json"
_PING = b": ping\n\n"
_ERROR_DETAIL_LIMIT = 500
_CLIENT_CLOSED = 499  # nginx's convention for "client closed the request"


class ChatResponder(Response):
    def __init__(
        self,
        *,
        chat: ChatRequest,
        plan: RoutePlan,
        executor: Executor,
        adapters: Mapping[str, ProviderAdapter],
        request_id: str,
        scope: str,
        lifecycle: Lifecycle,
        received_at: float,
        log_fields: MutableMapping[str, object],
        ledger: RequestLedger,
    ) -> None:
        # Starlette's Response.__init__ renders a body up front; this response has none
        # until the upstream call produces it, so only the attributes FastAPI reads are set.
        self.background = None
        self.status_code = 200
        self._chat = chat
        self._plan = plan
        self._executor = executor
        self._adapters = adapters
        self._lifecycle = lifecycle
        self._received_at = received_at
        self._log_fields = log_fields
        self._ledger = ledger
        self._output_chars = 0
        self._completion_id = f"chatcmpl-{uuid.uuid7().hex}"
        self._created = int(time.time())
        self._run = Run(plan=plan, request_id=request_id, scope=scope, deadline=0.0)

        self._status = _CLIENT_CLOSED
        self._outcome = "success"
        self._started = False  # status line sent
        self._finished = False  # final body frame sent
        self._committed: Committed[OpenStream] | None = None
        self._pinger: asyncio.Task[None] | None = None
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
        total = self._plan.primary.timeouts.total
        self._run.deadline = asyncio.get_running_loop().time() + total
        try:
            async with asyncio.timeout_at(self._run.deadline):
                committed = self._committed = await self._open(send)
                try:
                    if self._chat.stream:
                        await self._stream(send, committed)
                    else:
                        await self._aggregate(send, committed)
                except ProviderError as exc:
                    self._executor.post_commit_failure(committed, exc)
                    raise
                finally:
                    await committed.value.aclose()
        except ExhaustedError as exc:
            await self._fail(send, exc.error)
        except ProviderError as exc:
            await self._fail(send, exc)
        except TimeoutError:
            message = f"request exceeded the total timeout of {total:g}s"
            await self._fail(send, ProviderError(ErrorKind.TIMEOUT, message))

    async def _open(self, send: Send) -> Committed[OpenStream]:
        """Run the executor; for streams, send keep-alive pings while it works."""
        opening = self._executor.open_chat(self._run, self._chat, self._adapters)
        if not self._chat.stream:
            return await opening
        loop = asyncio.get_running_loop()
        timer = loop.call_later(self._plan.keepalive_after, self._start_pinger, send)
        try:
            return await opening
        finally:
            timer.cancel()
            if (pinger := self._pinger) is not None:
                self._pinger = None
                pinger.cancel()
                await asyncio.wait({pinger})
                if not pinger.cancelled():
                    pinger.exception()  # retrieved: a failed ping must not log as unhandled

    def _start_pinger(self, send: Send) -> None:
        self._pinger = asyncio.create_task(self._ping(send))

    async def _ping(self, send: Send) -> None:
        if not self._started:
            await send(self._start(200, _SSE_HEADERS, routed=False))
        while True:
            await send(_body(_PING))
            await asyncio.sleep(self._plan.keepalive_after)

    async def _stream(self, send: Send, committed: Committed[OpenStream]) -> None:
        stream = committed.value
        encoder = ChunkEncoder(
            completion_id=self._completion_id,
            created=self._created,
            model=committed.target.upstream_model,
            include_usage=self._chat.include_usage,
        )
        try:
            if not self._started:
                await send(self._start(200, _SSE_HEADERS))
            await send(_body(b"".join(encoder.encode(event) for event in stream.head)))
            if any(is_content(event) for event in stream.head):
                self._first_token_at = time.perf_counter()
            idle = committed.target.timeouts.idle
            async with idle_guard(idle, committed.ctx.timings) as watchdog:
                async for event in stream.events:
                    if frame := encoder.encode(event):
                        watchdog.sending = True
                        await send(_body(frame))
                        watchdog.sent()
            await send(_body(encoder.end(), more=False))
            self._finished = True
        finally:
            self._usage, self._finish = encoder.usage, encoder.finish_reason
            self._output_chars = encoder.output_chars

    async def _aggregate(self, send: Send, committed: Committed[OpenStream]) -> None:
        stream = committed.value
        aggregator = Aggregator(
            completion_id=self._completion_id,
            created=self._created,
            model=committed.target.upstream_model,
        )
        for event in stream.head:
            aggregator.add(event)
        try:
            async with idle_guard(committed.target.timeouts.idle, committed.ctx.timings):
                async for event in stream.events:
                    aggregator.add(event)
        finally:
            self._usage, self._finish = aggregator.usage, aggregator.finish_reason
            self._output_chars = aggregator.output_chars
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

    def _served_by(self) -> ModelTarget:
        """The committed target, else the last one tried, else the primary."""
        if self._committed is not None:
            return self._committed.target
        return self._run.last_target or self._plan.primary

    def _start(
        self,
        status: int,
        headers: tuple[tuple[bytes, bytes], ...] | list[tuple[bytes, bytes]],
        content_length: int | None = None,
        *,
        routed: bool = True,
    ) -> Message:
        """The status line. Set ``_started`` before the send, so a keep-alive pinger and
        the main path never both send it."""
        self._started = True
        self._status = status
        raw = list(headers)
        if routed:
            target = self._served_by()
            raw += [
                (b"x-switchboard-provider", target.provider.encode()),
                (b"x-switchboard-model", target.name.encode()),
                (b"x-switchboard-attempts", str(self._run.upstream_calls).encode()),
            ]
            committed = self._committed
            if committed is not None and committed.target is not self._plan.primary:
                raw.append((b"x-switchboard-fallback", b"true"))
            dropped = committed.ctx.dropped_params if committed is not None else ()
            if dropped:
                raw.append(
                    (b"x-switchboard-dropped-params", ",".join(sorted(set(dropped))).encode())
                )
            raw += [(k.encode(), v.encode()) for k, v in self._ledger.rate_headers().items()]
        if content_length is not None:
            raw.append((b"content-length", str(content_length).encode()))
        return {"type": "http.response.start", "status": status, "headers": raw}

    def _record(self) -> None:
        """Metrics and the request's log fields. Runs once, however the request ended."""
        now = time.perf_counter()
        target = self._served_by()
        labels = (self._plan.route_label, target.provider, target.name)
        REQUESTS.labels(*labels, str(self._status), self._outcome).inc()
        REQUEST_DURATION.labels(*labels, str(self._chat.stream).lower()).observe(
            now - self._received_at
        )
        first = self._run.first_ctx
        if first is not None and first.timings.upstream_start is not None:
            OVERHEAD_PRE_UPSTREAM.observe(first.timings.upstream_start - self._received_at)
        overhead_ms: float | None = None
        if first is not None and first.timings.upstream_start is not None:
            overhead_ms = (first.timings.upstream_start - self._received_at) * 1000
        if self._first_token_at is not None:
            TTFT.labels(*labels).observe(self._first_token_at - self._received_at)
            committed = self._committed
            read_at = committed.ctx.timings.first_content_read if committed else None
            if read_at is not None:
                OVERHEAD_FIRST_CHUNK.observe(self._first_token_at - read_at)
                overhead_ms = (overhead_ms or 0.0) + (self._first_token_at - read_at) * 1000
        cost = self._ledger.finish(self._served(overhead_ms))
        self._log_fields.update(self._log_summary(target))
        if cost.usd:
            self._log_fields["cost_usd"] = str(cost.usd)
            if cost.estimated:
                self._log_fields["cost_estimated"] = True

    def _served(self, overhead_ms: float | None) -> Served:
        run, committed = self._run, self._committed
        target = committed.target if committed is not None else run.last_target
        by_name = {t.name: t for t in self._plan.candidates}
        losers = [
            by_name[a.model]
            for a in run.attempts
            if run.hedged and a.outcome == "cancelled" and a.model in by_name
        ]
        return Served(
            target=target,
            outcome=self._outcome,
            http_status=self._status,
            usage=self._usage,
            error_kind=self._error.kind.value if self._error is not None else None,
            fallback=committed is not None and committed.target is not self._plan.primary,
            attempts=[
                {"model": a.model, "outcome": a.outcome, "http_status": a.upstream_status}
                for a in run.attempts
            ],
            hedge_losers=losers,
            output_chars=self._output_chars,
            ttft_ms=(self._first_token_at - self._received_at) * 1000
            if self._first_token_at is not None
            else None,
            overhead_ms=round(overhead_ms, 3) if overhead_ms is not None else None,
        )

    def _log_summary(self, target: ModelTarget) -> dict[str, object]:
        plan, run = self._plan, self._run
        fields: dict[str, object] = {
            "status": self._status,
            "alias": plan.alias,
            "provider": target.provider,
            "model": target.name,
            "stream": self._chat.stream,
            "outcome": self._outcome,
            "attempts": run.upstream_calls,
        }
        optional: dict[str, object | None] = {
            "rule": plan.rule,
            "route": plan.route if plan.route != plan.alias else None,
            "fallback": True if target is not plan.primary and self._committed else None,
            "hedged": True if run.hedged else None,
            "attempt_log": [str(a) for a in run.attempts]
            if len(run.attempts) > 1 or self._error is not None
            else None,
            "ttft_ms": round((self._first_token_at - self._received_at) * 1000, 2)
            if self._first_token_at is not None
            else None,
            "finish_reason": self._finish.value if self._finish is not None else None,
        }
        fields.update({k: v for k, v in optional.items() if v is not None})
        if self._usage is not None:
            fields["prompt_tokens"] = self._usage.prompt_total
            fields["completion_tokens"] = self._usage.output
        if self._error is not None:
            fields["error_kind"] = self._error.kind.value
            fields["error_detail"] = self._error.message[:_ERROR_DETAIL_LIMIT]
            if self._error.status is not None:
                fields["upstream_status"] = self._error.status
        return fields


def _body(data: bytes, *, more: bool = True) -> Message:
    return {"type": "http.response.body", "body": data, "more_body": more}


async def _wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass
