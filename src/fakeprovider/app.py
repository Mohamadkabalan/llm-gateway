"""The fake provider's ASGI app.

Plain ASGI with a small route table, so the per-chunk path is a sleep and one
``send``. Endpoints:

* ``POST /v1/chat/completions`` (OpenAI) and ``POST /v1/messages`` (Anthropic), both
  streaming and not; ``POST /v1/embeddings``, ``GET /v1/models``, ``GET /health``
* ``GET /_control/stats``, ``POST /_control/reset``, ``PUT|DELETE /_control/behavior``

Stats and runtime behavior live in the worker process. With more than one worker they
would differ per worker, so the control API refuses to run then.
"""

import asyncio
import itertools
import logging
import math
import os
import random
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import orjson
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send

from fakeprovider import openai_dialect as oai
from fakeprovider.anthropic_dialect import AnthropicDialect
from fakeprovider.behavior import Behavior, Fault, header_overrides, load_profile
from fakeprovider.plan import Completion, Dialect, StreamPlan

_SSE_HEADERS = [
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
]
_JSON_HEADERS = [(b"content-type", b"application/json")]
_RECENT_REQUESTS = 100


class SimulatedDisconnect(Exception):  # noqa: N818 (it is not an error condition)
    """Raised mid-response; uvicorn then closes the connection without finishing it."""


class _QuietSimulatedDisconnect(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and isinstance(record.exc_info[1], SimulatedDisconnect))


@dataclass
class Stats:
    requests: Counter[str] = field(default_factory=Counter)
    faults: Counter[str] = field(default_factory=Counter)
    streams_completed: int = 0
    client_disconnects: int = 0
    recent: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_RECENT_REQUESTS))

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": dict(self.requests),
            "faults": dict(self.faults),
            "streams_completed": self.streams_completed,
            "client_disconnects": self.client_disconnects,
            "recent": list(self.recent),
        }


@dataclass
class _Request:
    scope: Scope
    receive: Receive
    send: Send
    body: bytes = b""

    @property
    def headers(self) -> list[tuple[bytes, bytes]]:
        return list(self.scope.get("headers", ()))


Handler = Callable[[_Request], Awaitable[None]]
_OPENAI = oai.OpenAIDialect()
_ANTHROPIC = AnthropicDialect()


class FakeProvider:
    def __init__(
        self, behavior: Behavior | None = None, *, workers: int = 1, seed: int | None = None
    ):
        self.profile = behavior or Behavior()
        self.runtime: dict[str, Any] = {}
        self.stats = Stats()
        self._workers = workers
        self._rng = random.Random(seed)  # noqa: S311 (fault injection, not crypto)
        self._ids = itertools.count(1)
        self._routes: dict[tuple[str, str], Handler] = {
            ("POST", "/v1/chat/completions"): self._chat,
            ("POST", "/v1/messages"): self._messages,
            ("POST", "/v1/embeddings"): self._embeddings,
            ("GET", "/v1/models"): self._models,
            ("GET", "/health"): self._health,
            ("GET", "/_control/stats"): self._control_stats,
            ("POST", "/_control/reset"): self._control_reset,
            ("PUT", "/_control/behavior"): self._control_set_behavior,
            ("DELETE", "/_control/behavior"): self._control_clear_behavior,
        }
        error_log = logging.getLogger("uvicorn.error")
        if not any(isinstance(f, _QuietSimulatedDisconnect) for f in error_log.filters):
            error_log.addFilter(_QuietSimulatedDisconnect())

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await _lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        handler = self._routes.get((scope["method"], scope["path"]))
        request = _Request(scope, receive, send)
        if handler is None:
            await _json(send, 404, oai.error_body("Not found", "not_found_error", None))
            return
        if scope["method"] in ("POST", "PUT"):
            request.body = await _read_body(receive)
        if not scope["path"].startswith("/_control/"):
            self.stats.requests[scope["path"]] += 1
        await handler(request)

    # --- chat completions and messages -----------------------------------------------

    async def _chat(self, request: _Request) -> None:
        await self._complete(request, _OPENAI)

    async def _messages(self, request: _Request) -> None:
        await self._complete(request, _ANTHROPIC)

    async def _complete(self, request: _Request, dialect: Dialect) -> None:
        headers = _plain(request.headers)
        try:
            payload = orjson.loads(request.body)
            if not isinstance(payload, dict):
                raise TypeError("the body must be a JSON object")
            behavior = self._behavior(request, payload)
        except (orjson.JSONDecodeError, ValidationError, AttributeError, TypeError) as exc:
            await _json(request.send, 400, dialect.error_body(400, str(exc), None))
            return
        self.stats.recent.append({"path": dialect.path, "body": payload, "headers": headers})
        if (problem := dialect.validate(payload, headers)) is not None:
            await _json(request.send, 400, dialect.error_body(400, problem, None))
            return

        fault = behavior.fault if self._rng.random() < behavior.fault_rate else Fault.NONE
        if fault is not Fault.NONE:
            self.stats.faults[fault.value] += 1
        if fault is Fault.STATUS:
            await self._status_fault(request.send, dialect, behavior)
            return
        if fault is Fault.HANG:
            await _wait_for_disconnect(request.receive)
            self.stats.client_disconnects += 1
            return

        completion = Completion(
            completion_id=f"{dialect.id_prefix}{next(self._ids)}",
            created=int(time.time()),
            model=behavior.model or str(payload.get("model") or "fake-chat"),
            prompt_tokens=max(1, math.ceil(len(request.body) / 4)),
            scenario=dialect.resolve_scenario(behavior, payload),
            tool_names=dialect.tool_names(payload),
            include_usage=dialect.include_usage(payload),
        )
        if payload.get("stream"):
            plan = dialect.stream_plan(behavior, completion)
            await self._watch(request, self._play(request.send, plan, behavior, fault))
        else:
            pieces = len(dialect.stream_plan(behavior, completion).pieces)
            await asyncio.sleep(
                (behavior.ttft_ms + max(pieces - 1, 0) * behavior.interval_ms) / 1000
            )
            body = orjson.dumps(dialect.response_body(behavior, completion))
            await _json(request.send, 200, body)

    def _behavior(self, request: _Request, payload: dict[str, Any]) -> Behavior:
        behavior = self.profile.merged(self.runtime).merged(header_overrides(request.headers))
        return behavior.merged(payload.get("fake") or {})

    async def _status_fault(self, send: Send, dialect: Dialect, behavior: Behavior) -> None:
        headers = list(_JSON_HEADERS)
        if behavior.retry_after is not None:
            headers.append((b"retry-after", f"{behavior.retry_after:g}".encode()))
        body = dialect.error_body(behavior.status, f"fake {behavior.status}", behavior.error_code)
        await _json(send, behavior.status, body, headers)

    async def _play(self, send: Send, plan: StreamPlan, behavior: Behavior, fault: Fault) -> None:
        """Send a stream: head now, content at first-token + i x interval, then the tail.
        Times are absolute, so per-chunk scheduling jitter does not accumulate."""
        loop = asyncio.get_running_loop()
        await send({"type": "http.response.start", "status": 200, "headers": _SSE_HEADERS})
        if plan.head:
            await send(_body(plan.head))
        first_at = loop.time() + behavior.ttft_ms / 1000
        interval = behavior.interval_ms / 1000
        faulty = fault in (Fault.MID_STREAM_ERROR, Fault.DISCONNECT)
        for index, piece in enumerate(plan.pieces):
            await _sleep_until(loop, first_at + index * interval)
            if faulty and index == behavior.fault_after:
                await self._mid_stream_fault(send, fault, plan)
                return
            await send(_body(plan.first_prefix + piece if index == 0 else piece))
        if faulty:  # fault_after is at or past the last piece
            await _sleep_until(loop, first_at + len(plan.pieces) * interval)
            await self._mid_stream_fault(send, fault, plan)
            return
        if not plan.pieces:
            await _sleep_until(loop, first_at)
        await send(_body(plan.tail, more=False))
        self.stats.streams_completed += 1

    async def _mid_stream_fault(self, send: Send, fault: Fault, plan: StreamPlan) -> None:
        if fault is Fault.DISCONNECT:
            raise SimulatedDisconnect
        await send(_body(plan.error, more=False))

    async def _watch(self, request: _Request, work: Awaitable[None]) -> None:
        """Run ``work`` until it finishes or the client disconnects, then cancel it."""
        task = asyncio.ensure_future(work)
        watcher = asyncio.ensure_future(_wait_for_disconnect(request.receive))
        try:
            await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not task.done():
                self.stats.client_disconnects += 1
            task.cancel()
            watcher.cancel()
            await asyncio.wait((task, watcher))
        if not task.cancelled() and (exc := task.exception()) is not None:
            raise exc

    # --- embeddings and models -------------------------------------------------------

    async def _embeddings(self, request: _Request) -> None:
        try:
            payload = orjson.loads(request.body)
            inputs = payload["input"]
            texts = [inputs] if isinstance(inputs, str) else list(inputs)
            if not all(isinstance(text, str) for text in texts):
                raise TypeError("input must be a string or a list of strings")
            dimensions = int(payload.get("dimensions") or 256)
        except (orjson.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            await _json(request.send, 400, oai.error_body(str(exc), "invalid_request_error", None))
            return
        self.stats.recent.append(
            {"path": "/v1/embeddings", "body": payload, "headers": _plain(request.headers)}
        )
        encoding = str(payload.get("encoding_format") or "float")
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": oai.encode_embedding(oai.embedding(text, dimensions), encoding),
            }
            for i, text in enumerate(texts)
        ]
        tokens = sum(max(1, math.ceil(len(text) / 4)) for text in texts)
        body = {
            "object": "list",
            "data": data,
            "model": payload.get("model") or "fake-embed",
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
        await _json(request.send, 200, orjson.dumps(body))

    async def _models(self, request: _Request) -> None:
        models = [
            {"id": name, "object": "model", "created": 0, "owned_by": "fakeprovider"}
            for name in ("fake-chat", "fake-embed")
        ]
        await _json(request.send, 200, orjson.dumps({"object": "list", "data": models}))

    async def _health(self, request: _Request) -> None:
        await _json(request.send, 200, b'{"status":"ok"}')

    # --- control API -----------------------------------------------------------------

    async def _control_stats(self, request: _Request) -> None:
        if await self._refuse_multi_worker(request.send):
            return
        await _json(request.send, 200, orjson.dumps(self.stats.as_dict()))

    async def _control_reset(self, request: _Request) -> None:
        if await self._refuse_multi_worker(request.send):
            return
        self.stats = Stats()
        await _json(request.send, 200, b'{"status":"reset"}')

    async def _control_set_behavior(self, request: _Request) -> None:
        if await self._refuse_multi_worker(request.send):
            return
        try:
            overrides = orjson.loads(request.body)
            self.profile.merged(overrides)  # validate before accepting
        except (orjson.JSONDecodeError, ValidationError, AttributeError, TypeError) as exc:
            await _json(request.send, 400, oai.error_body(str(exc), "invalid_request_error", None))
            return
        self.runtime = dict(overrides)
        await _json(request.send, 200, orjson.dumps(self.profile.merged(self.runtime).model_dump()))

    async def _control_clear_behavior(self, request: _Request) -> None:
        if await self._refuse_multi_worker(request.send):
            return
        self.runtime = {}
        await _json(request.send, 200, orjson.dumps(self.profile.model_dump()))

    async def _refuse_multi_worker(self, send: Send) -> bool:
        if self._workers == 1:
            return False
        message = "the control API needs --workers 1 (state is per worker process)"
        await _json(send, 409, oai.error_body(message, "conflict_error", None))
        return True


def app_from_env() -> FakeProvider:
    """Factory for ``uvicorn --factory`` (each worker process calls it)."""
    seed = os.environ.get("FAKE_SEED")
    return FakeProvider(
        load_profile(os.environ.get("FAKE_PROFILE") or None),
        workers=int(os.environ.get("FAKE_WORKERS", "1")),
        seed=int(seed) if seed else None,
    )


def _plain(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    return {name.decode(): value.decode("latin-1") for name, value in headers}


async def _sleep_until(loop: asyncio.AbstractEventLoop, when: float) -> None:
    delay = when - loop.time()
    if delay > 0:
        await asyncio.sleep(delay)


async def _read_body(receive: Receive) -> bytes:
    parts: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        parts.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(parts)


async def _wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


def _body(data: bytes, *, more: bool = True) -> Message:
    return {"type": "http.response.body", "body": data, "more_body": more}


async def _json(
    send: Send, status: int, body: bytes, headers: list[tuple[bytes, bytes]] | None = None
) -> None:
    raw = [*(headers or _JSON_HEADERS), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send(_body(body, more=False))


async def _lifespan(receive: Receive, send: Send) -> None:
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return
