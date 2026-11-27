"""Bearer token → ``KeyContext``, with an in-process cache and push invalidation.

* **Cache.** Lookups are cached for ``ttl`` (30 s) per key hash, unknown keys for
  ``negative_ttl`` (5 s) so a flood of bad keys cannot hammer Postgres. Concurrent
  misses for the same key share one lookup.
* **Revocation within a second.** Changes are published on the Redis channel
  ``sb:chan:invalidate``; every replica listens and drops the affected entries at once.
  If the subscription drops, the whole cache is cleared on reconnect, because messages
  may have been missed. Without Redis, a change takes effect within ``ttl``.
* **Stale if error.** If the store fails (Postgres down), an entry that has expired but
  was never invalidated is still used for up to ``stale_ttl``, so a database blip does
  not log every client out. Revocations still apply: they remove the entry.
"""

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import orjson
import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from switchboard.auth.context import KeyContext
from switchboard.auth.keys import hash_secret, plausible
from switchboard.auth.store import KeyStore
from switchboard.observability.metrics import AUTH_FAILURES, DEGRADED

_log = structlog.get_logger("switchboard.auth")

INVALIDATE_CHANNEL = "sb:chan:invalidate"
_RECONNECT_DELAY = 1.0


class AuthError(Exception):
    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


@dataclass(slots=True)
class _Entry:
    context: KeyContext | None
    fresh_until: float
    stale_until: float


class KeyResolver:
    def __init__(
        self,
        store: KeyStore,
        pepper: bytes,
        *,
        ttl: float = 30.0,
        negative_ttl: float = 5.0,
        stale_ttl: float = 300.0,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._pepper = pepper
        self._ttl = ttl
        self._negative_ttl = negative_ttl
        self._stale_ttl = stale_ttl
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()
        self._inflight: dict[bytes, asyncio.Future[KeyContext | None]] = {}

    async def resolve(self, token: str) -> KeyContext:
        if not plausible(token):
            raise _invalid("malformed")
        key_hash = hash_secret(self._pepper, token)
        context = await self._lookup(key_hash)
        if context is None:
            raise _invalid("unknown")
        if context.revoked:
            AUTH_FAILURES.labels("revoked").inc()
            raise AuthError(401, "This API key has been revoked.", "invalid_api_key")
        if context.expires_at is not None and context.expires_at <= datetime.now(UTC):
            AUTH_FAILURES.labels("expired").inc()
            raise AuthError(401, "This API key has expired.", "expired_api_key")
        return context

    async def _lookup(self, key_hash: bytes) -> KeyContext | None:
        now = self._clock()
        entry = self._entries.get(key_hash)
        if entry is not None and now < entry.fresh_until:
            self._entries.move_to_end(key_hash)
            return entry.context
        pending = self._inflight.get(key_hash)
        if pending is not None:
            return await asyncio.shield(pending)
        future: asyncio.Future[KeyContext | None] = asyncio.get_running_loop().create_future()
        self._inflight[key_hash] = future
        try:
            try:
                context = await self._store.lookup(key_hash)
            except Exception as exc:  # the store is down: serve a stale entry if there is one
                if entry is not None and now < entry.stale_until:
                    DEGRADED.labels("auth").inc()
                    _log.warning("auth.store_unavailable", detail=str(exc), served="stale")
                    future.set_result(entry.context)
                    return entry.context
                _log.error("auth.store_unavailable", detail=str(exc))
                raise AuthError(
                    503, "Authentication is temporarily unavailable.", "auth_unavailable"
                ) from exc
            self._remember(key_hash, context, now)
            future.set_result(context)
            return context
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                future.exception()  # retrieved here; waiters re-raise it
            raise
        finally:
            del self._inflight[key_hash]

    def _remember(self, key_hash: bytes, context: KeyContext | None, now: float) -> None:
        ttl = self._ttl if context is not None else self._negative_ttl
        self._entries[key_hash] = _Entry(context, now + ttl, now + ttl + self._stale_ttl)
        self._entries.move_to_end(key_hash)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    # --- invalidation ------------------------------------------------------------------

    def invalidate_key(self, key_hash: bytes) -> None:
        self._entries.pop(key_hash, None)

    def invalidate_team(self, team_id: UUID) -> None:
        for key_hash, entry in list(self._entries.items()):
            if entry.context is not None and entry.context.team_id == team_id:
                del self._entries[key_hash]

    def clear(self) -> None:
        self._entries.clear()

    def apply(self, message: bytes) -> None:
        """Apply one invalidation message (see ``invalidation_message``)."""
        try:
            data = orjson.loads(message)
        except orjson.JSONDecodeError:
            self.clear()  # unreadable: assume the worst
            return
        if isinstance(data, dict) and isinstance(data.get("key_hash"), str):
            self.invalidate_key(bytes.fromhex(data["key_hash"]))
        elif isinstance(data, dict) and isinstance(data.get("team_id"), str):
            self.invalidate_team(UUID(data["team_id"]))
        else:
            self.clear()

    async def listen(self, redis: Redis) -> None:
        """Follow the invalidation channel until cancelled, reconnecting as needed."""
        first = True
        while True:
            try:
                async with redis.pubsub() as pubsub:
                    await pubsub.subscribe(INVALIDATE_CHANNEL)
                    if not first:
                        self.clear()  # messages may have been missed while disconnected
                        _log.info("auth.invalidation_resubscribed")
                    first = False
                    async for message in pubsub.listen():
                        if message.get("type") == "message":
                            self.apply(message["data"])
            except (RedisError, OSError) as exc:
                _log.warning("auth.invalidation_disconnected", detail=str(exc))
                await asyncio.sleep(_RECONNECT_DELAY)


def invalidation_message(*, key_hash: bytes | None = None, team_id: UUID | None = None) -> bytes:
    if key_hash is not None:
        return orjson.dumps({"key_hash": key_hash.hex()})
    if team_id is not None:
        return orjson.dumps({"team_id": str(team_id)})
    return orjson.dumps({"all": True})


async def publish_invalidation(
    redis: Redis | None, *, key_hash: bytes | None = None, team_id: UUID | None = None
) -> bool:
    """Tell every replica to drop cached state for a key or team. False if Redis is
    unavailable (the change then takes effect when cache entries expire)."""
    if redis is None:
        return False
    with contextlib.suppress(RedisError, OSError):
        message = invalidation_message(key_hash=key_hash, team_id=team_id)
        await redis.publish(INVALIDATE_CHANNEL, message)
        return True
    return False


def _invalid(reason: str) -> AuthError:
    AUTH_FAILURES.labels(reason).inc()
    return AuthError(401, "Incorrect API key provided.", "invalid_api_key")
