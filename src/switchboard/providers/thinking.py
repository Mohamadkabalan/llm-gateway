"""The thinking sidecar: carry Anthropic thinking blocks across OpenAI-format tool loops.

Within a tool-use turn, Anthropic expects the assistant turn's thinking blocks back
unchanged with the tool results. OpenAI-format clients have nowhere to keep them, so a
naive translation drops them and the model loses its reasoning mid-loop (PLAN.md §3.4,
ADR-0009). The gateway therefore:

1. stores the full content array of a streamed response that called tools and
   contains thinking, keyed by (scope, the tool_use ids), for an hour;
2. when a later request's last assistant turn has those tool-call ids, and the text
   and tool calls the client echoed match what was stored, sends the stored array
   verbatim instead of the translated one.

Any doubt (no entry, a mismatch, Redis unavailable) falls back to sending the turn
without thinking blocks, which the API accepts; only the reasoning is lost. Outcomes
are counted in ``switchboard_thinking_passback_total``.
"""

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Protocol

import orjson
import structlog
from prometheus_client import Counter
from redis.asyncio import Redis
from redis.exceptions import RedisError

TTL_SECONDS = 3600
_KEY_PREFIX = "sb:thinking:"

PASSBACK = Counter(
    "switchboard_thinking_passback",
    "Thinking-block pass-back for Anthropic tool loops. hit: stored blocks were sent; "
    "miss: nothing stored; mismatch: the client's history differed from what was stored; "
    "stored: a response's blocks were saved; error: the store was unavailable.",
    ["result"],
)

_log = structlog.get_logger("switchboard.thinking")

ContentBlocks = list[dict[str, Any]]


class ThinkingSidecar(Protocol):
    async def load(self, scope: str, tool_use_ids: Sequence[str]) -> ContentBlocks | None: ...

    def store(self, scope: str, tool_use_ids: Sequence[str], content: ContentBlocks) -> None:
        """Save in the background; never delays or fails the response."""
        ...


class MemoryThinkingSidecar:
    """In-process store for when Redis is not configured.

    Correct for a single gateway replica; with several, a follow-up request that lands
    on another replica is a miss (the turn is sent without thinking, which still works).
    """

    def __init__(self, *, ttl_seconds: int = TTL_SECONDS, max_entries: int = 10_000) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: OrderedDict[str, tuple[float, bytes]] = OrderedDict()

    async def load(self, scope: str, tool_use_ids: Sequence[str]) -> ContentBlocks | None:
        entry = self._entries.get(key(scope, tool_use_ids))
        if entry is None or entry[0] < time.monotonic():
            return None
        blocks = orjson.loads(entry[1])
        return blocks if isinstance(blocks, list) else None

    def store(self, scope: str, tool_use_ids: Sequence[str], content: ContentBlocks) -> None:
        entry_key = key(scope, tool_use_ids)
        self._entries[entry_key] = (time.monotonic() + self._ttl, orjson.dumps(content))
        self._entries.move_to_end(entry_key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)
        PASSBACK.labels(result="stored").inc()


class RedisThinkingSidecar:
    def __init__(self, redis: Redis, *, ttl_seconds: int = TTL_SECONDS) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._pending: set[asyncio.Task[None]] = set()

    async def load(self, scope: str, tool_use_ids: Sequence[str]) -> ContentBlocks | None:
        try:
            raw = await self._redis.get(key(scope, tool_use_ids))
        except RedisError as exc:
            PASSBACK.labels(result="error").inc()
            _log.warning("thinking.load_failed", error=f"{type(exc).__name__}: {exc}")
            return None
        if raw is None:
            return None
        blocks = orjson.loads(raw)
        return blocks if isinstance(blocks, list) else None

    def store(self, scope: str, tool_use_ids: Sequence[str], content: ContentBlocks) -> None:
        task = asyncio.create_task(self._store(key(scope, tool_use_ids), orjson.dumps(content)))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _store(self, redis_key: str, value: bytes) -> None:
        try:
            await self._redis.set(redis_key, value, ex=self._ttl)
        except RedisError as exc:
            PASSBACK.labels(result="error").inc()
            _log.warning("thinking.store_failed", error=f"{type(exc).__name__}: {exc}")
        else:
            PASSBACK.labels(result="stored").inc()


def key(scope: str, tool_use_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\0".join([scope, *sorted(tool_use_ids)]).encode()).hexdigest()
    return _KEY_PREFIX + digest[:40]


def matches(stored: ContentBlocks, text: str, tool_calls: Sequence[tuple[str, str, Any]]) -> bool:
    """Does the client's echo of a turn (its text and ``(id, name, input)`` tool calls)
    agree with the stored content? Only then is it safe to swap the stored blocks in."""
    stored_text = "".join(b.get("text", "") for b in stored if b.get("type") == "text")
    stored_calls = [
        (b.get("id"), b.get("name"), b.get("input")) for b in stored if b.get("type") == "tool_use"
    ]
    return stored_text.strip() == text.strip() and stored_calls == list(tool_calls)
