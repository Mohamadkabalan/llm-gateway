"""Circuit breakers, one per ``provider/model`` (PLAN.md §5, ADR-0004).

States::

    closed ──(failure rate ≥ threshold over the window, with ≥ min_requests)──► open
    open ──(cooldown elapsed; the next request)──► half_open
    half_open ──(every probe succeeds)──► closed
    half_open ──(a probe fails; cooldown doubles, capped)──► open

Outcomes are counted in a ring of ``BREAKER_SLOTS`` buckets covering the window, so
it slides in steps of a tenth of its length. In half-open, at most
``half_open_probes`` requests are let through as probes; the rest are refused until
the probes report. A probe round that never reports (the
probing replica died, the client hung up) is restarted after one cooldown.

Two stores implement the same state machine:

* ``RedisBreakerStore``: Lua scripts, so every replica sees one breaker and the
  half-open probe count is exact across replicas.
* ``MemoryBreakerStore``: per process; used without Redis and whenever Redis fails.

``Breakers`` is what the executor calls. With Redis it keeps a local view of each
breaker, updated from every store reply: a closed or still-open breaker is decided
locally with no round trip; only an expired open breaker and half-open probing go to
Redis. Outcomes are written in the background and never delay a response.
"""

import asyncio
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from switchboard.observability.metrics import BREAKER_STATE, BREAKER_TRANSITIONS
from switchboard.routing.config import BREAKER_SLOTS, BreakerPolicy
from switchboard.routing.table import ModelTarget

_log = structlog.get_logger("switchboard.reliability.breaker")

KEY_PREFIX = "sb:cb:"
_REDIS_RETRY_AFTER = 5.0  # after a Redis error, use the in-process store this long
_SHUTDOWN_WAIT = 2.0
_STATE_VALUE = {"closed": 0, "half_open": 1, "open": 2}


class BreakerState(StrEnum):
    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"


class Admission(StrEnum):
    ALLOW = "allow"
    PROBE = "probe"  # allowed as one of the half-open probes; report its outcome
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Verdict:
    state: BreakerState
    open_until: float = 0.0  # epoch seconds; meaningful while open
    transitioned: bool = False
    admission: Admission = Admission.ALLOW  # for admit() replies


def redis_key(name: str) -> str:
    # The braces make the whole name a Redis Cluster hash tag.
    return f"{KEY_PREFIX}{{{name}}}"


def idle_ttl(policy: BreakerPolicy) -> float:
    """A closed breaker with no traffic for this long has nothing left to count."""
    return policy.window + policy.bucket_seconds


# --- in-process store ------------------------------------------------------------------


@dataclass(slots=True)
class _Breaker:
    state: BreakerState = BreakerState.CLOSED
    open_until: float = 0.0
    cooldown: float | None = None
    probes: int = 0
    probes_ok: int = 0
    probe_deadline: float = 0.0
    expires: float = math.inf  # mirrors the Redis key's TTL
    bucket_ids: list[int] = field(default_factory=list)
    ok: list[int] = field(default_factory=list)
    fail: list[int] = field(default_factory=list)


class MemoryBreakerStore:
    """The state machine in Python. Kept line-for-line equivalent to the Lua scripts
    below; one scenario table tests both (``tests/breaker_cases.py``)."""

    def __init__(self) -> None:
        self._breakers: dict[str, _Breaker] = {}

    def admit(self, name: str, policy: BreakerPolicy, now: float) -> Verdict:
        b = self._get(name, now)
        if b.state is BreakerState.CLOSED:
            return Verdict(BreakerState.CLOSED)
        cooldown = b.cooldown or policy.cooldown
        transitioned = False
        if b.state is BreakerState.OPEN:
            if now < b.open_until:
                return Verdict(BreakerState.OPEN, b.open_until, admission=Admission.DENY)
            b.state, b.probes, b.probes_ok = BreakerState.HALF_OPEN, 0, 0
            b.probe_deadline = now + cooldown
            b.expires = now + math.ceil(cooldown) + idle_ttl(policy)
            transitioned = True
        elif now >= b.probe_deadline:  # a probe round that never reported back
            b.probes, b.probes_ok = 0, 0
            b.probe_deadline = now + cooldown
            b.expires = now + math.ceil(cooldown) + idle_ttl(policy)
        b.probes += 1
        admission = Admission.PROBE if b.probes <= policy.half_open_probes else Admission.DENY
        return Verdict(BreakerState.HALF_OPEN, transitioned=transitioned, admission=admission)

    def record(
        self, name: str, policy: BreakerPolicy, now: float, *, ok: bool, probe: bool
    ) -> Verdict:
        b = self._get(name, now)
        if b.state is BreakerState.OPEN:
            return Verdict(BreakerState.OPEN, b.open_until)
        if b.state is BreakerState.HALF_OPEN:
            return self._record_half_open(name, b, policy, now, ok=ok, probe=probe)
        n = BREAKER_SLOTS
        if len(b.bucket_ids) != n:
            b.bucket_ids, b.ok, b.fail = [-1] * n, [0] * n, [0] * n
        bucket = math.floor(now / policy.bucket_seconds)
        slot = bucket % n
        if b.bucket_ids[slot] != bucket:
            b.bucket_ids[slot], b.ok[slot], b.fail[slot] = bucket, 0, 0
        if ok:
            b.ok[slot] += 1
        else:
            b.fail[slot] += 1
        b.expires = now + idle_ttl(policy)
        total = fails = 0
        for i in range(n):
            if 0 <= bucket - b.bucket_ids[i] < n:
                total += b.ok[i] + b.fail[i]
                fails += b.fail[i]
        if total >= policy.min_requests and fails >= policy.failure_rate * total:
            return self._open(name, policy, now, policy.cooldown)
        return Verdict(BreakerState.CLOSED)

    def state(self, name: str, now: float) -> BreakerState:
        return self._get(name, now).state

    def _record_half_open(
        self, name: str, b: _Breaker, policy: BreakerPolicy, now: float, *, ok: bool, probe: bool
    ) -> Verdict:
        if not probe:
            return Verdict(BreakerState.HALF_OPEN)  # admitted before it opened
        if not ok:
            cooldown = min((b.cooldown or policy.cooldown) * 2, policy.max_cooldown)
            return self._open(name, policy, now, cooldown)
        b.probes_ok += 1
        if b.probes_ok >= policy.half_open_probes:
            del self._breakers[name]
            return Verdict(BreakerState.CLOSED, transitioned=True)
        return Verdict(BreakerState.HALF_OPEN)

    def _open(self, name: str, policy: BreakerPolicy, now: float, cooldown: float) -> Verdict:
        until = now + cooldown
        self._breakers[name] = _Breaker(
            state=BreakerState.OPEN,
            open_until=until,
            cooldown=cooldown,
            expires=now + math.ceil(cooldown) + idle_ttl(policy),
        )
        return Verdict(BreakerState.OPEN, until, transitioned=True)

    def _get(self, name: str, now: float) -> _Breaker:
        b = self._breakers.get(name)
        if b is None or now >= b.expires:
            b = self._breakers[name] = _Breaker()
        return b


# --- Redis store -----------------------------------------------------------------------

# Hash fields: st state, ou open_until, cd cooldown, pr probes issued, pk probes ok,
# pd probe-round deadline, and per ring slot i: b<i> bucket id, o<i> ok, f<i> fail.
# Numbers go back to Python as strings: Lua numbers become integers in Redis replies.

_ADMIT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local max_probes = tonumber(ARGV[2])
local base_cooldown = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local v = redis.call('HMGET', key, 'st', 'ou', 'cd', 'pd')
local state = v[1] or 'closed'
if state == 'closed' then return {'allow', 'closed', '0', 0} end
local cooldown = tonumber(v[3] or base_cooldown)
local transitioned = 0
if state == 'open' then
  if now < tonumber(v[2]) then return {'deny', 'open', v[2], 0} end
  redis.call('HSET', key, 'st', 'half_open', 'pr', 0, 'pk', 0, 'pd', tostring(now + cooldown))
  redis.call('EXPIRE', key, math.ceil(cooldown) + ttl)
  transitioned = 1
elseif now >= tonumber(v[4] or 0) then
  redis.call('HSET', key, 'pr', 0, 'pk', 0, 'pd', tostring(now + cooldown))
  redis.call('EXPIRE', key, math.ceil(cooldown) + ttl)
end
if redis.call('HINCRBY', key, 'pr', 1) <= max_probes then
  return {'probe', 'half_open', '0', transitioned}
end
return {'deny', 'half_open', '0', transitioned}
"""

_RECORD = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local ok = ARGV[2] == '1'
local probe = ARGV[3] == '1'
local n = tonumber(ARGV[4])
local bucket_seconds = tonumber(ARGV[5])
local min_requests = tonumber(ARGV[6])
local failure_rate = tonumber(ARGV[7])
local base_cooldown = tonumber(ARGV[8])
local max_cooldown = tonumber(ARGV[9])
local max_probes = tonumber(ARGV[10])
local ttl = tonumber(ARGV[11])

local function open(cooldown)
  local open_until = tostring(now + cooldown)
  redis.call('DEL', key)
  redis.call('HSET', key, 'st', 'open', 'ou', open_until, 'cd', tostring(cooldown))
  redis.call('EXPIRE', key, math.ceil(cooldown) + ttl)
  return {'open', open_until, 1}
end

local state = redis.call('HGET', key, 'st') or 'closed'
if state == 'open' then return {'open', redis.call('HGET', key, 'ou'), 0} end
if state == 'half_open' then
  if not probe then return {'half_open', '0', 0} end
  if not ok then
    local cooldown = tonumber(redis.call('HGET', key, 'cd') or base_cooldown)
    return open(math.min(cooldown * 2, max_cooldown))
  end
  if redis.call('HINCRBY', key, 'pk', 1) >= max_probes then
    redis.call('DEL', key)
    return {'closed', '0', 1}
  end
  return {'half_open', '0', 0}
end

local bucket = math.floor(now / bucket_seconds)
local slot = bucket % n
local fb, fo, ff = 'b' .. slot, 'o' .. slot, 'f' .. slot
if tonumber(redis.call('HGET', key, fb) or -1) ~= bucket then
  redis.call('HSET', key, fb, bucket, fo, 0, ff, 0)
end
if ok then redis.call('HINCRBY', key, fo, 1) else redis.call('HINCRBY', key, ff, 1) end
redis.call('EXPIRE', key, ttl)
local total, fails = 0, 0
for i = 0, n - 1 do
  local v = redis.call('HMGET', key, 'b' .. i, 'o' .. i, 'f' .. i)
  local b = tonumber(v[1])
  if b and bucket - b >= 0 and bucket - b < n then
    total = total + tonumber(v[2]) + tonumber(v[3])
    fails = fails + tonumber(v[3])
  end
end
if total >= min_requests and fails >= failure_rate * total then
  return open(base_cooldown)
end
return {'closed', '0', 0}
"""


class RedisBreakerStore:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._admit = redis.register_script(_ADMIT)
        self._record = redis.register_script(_RECORD)

    async def admit(self, name: str, policy: BreakerPolicy, now: float) -> Verdict:
        reply = await self._admit(
            keys=[redis_key(name)],
            args=[repr(now), policy.half_open_probes, repr(policy.cooldown), _ttl(policy)],
        )
        admission, state, open_until, transitioned = _decode(reply)
        return Verdict(
            BreakerState(state), float(open_until), bool(int(transitioned)), Admission(admission)
        )

    async def record(
        self, name: str, policy: BreakerPolicy, now: float, *, ok: bool, probe: bool
    ) -> Verdict:
        reply = await self._record(
            keys=[redis_key(name)],
            args=[
                repr(now),
                int(ok),
                int(probe),
                BREAKER_SLOTS,
                repr(policy.bucket_seconds),
                policy.min_requests,
                repr(policy.failure_rate),
                repr(policy.cooldown),
                repr(policy.max_cooldown),
                policy.half_open_probes,
                _ttl(policy),
            ],
        )
        state, open_until, transitioned = _decode(reply)
        return Verdict(BreakerState(state), float(open_until), bool(int(transitioned)))

    async def state(self, name: str) -> BreakerState:
        raw: bytes | str | None = await self._redis.hget(redis_key(name), "st")
        if raw is None:
            return BreakerState.CLOSED
        return BreakerState(raw.decode() if isinstance(raw, bytes) else raw)


def _ttl(policy: BreakerPolicy) -> int:
    return math.ceil(idle_ttl(policy))


def _decode(reply: object) -> list[str]:
    assert isinstance(reply, list)  # noqa: S101 (the scripts always return a list)
    return [part.decode() if isinstance(part, bytes) else str(part) for part in reply]


# --- the facade the executor uses ------------------------------------------------------


class Breakers:
    def __init__(
        self,
        redis: Redis | None = None,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.memory = MemoryBreakerStore()
        self._redis = RedisBreakerStore(redis) if redis is not None else None
        self._clock = clock
        self._monotonic = monotonic
        self._redis_down_until = -math.inf
        # Redis mode: the last state each breaker was seen in, and until when it is open.
        self._view: dict[str, tuple[BreakerState, float]] = {}
        self._pending: set[asyncio.Task[None]] = set()

    def track(self, names: Iterable[str]) -> None:
        """Export a state series for every configured model, so dashboards see zeros."""
        for name in names:
            BREAKER_STATE.labels(name).set(_STATE_VALUE[self._local_state(name)])

    async def admit(self, target: ModelTarget) -> Admission:
        now = self._clock()
        if not self._redis_ok():
            return self._observe(target.name, self.memory.admit(target.name, target.breaker, now))
        state, open_until = self._view.get(target.name, (BreakerState.CLOSED, 0.0))
        if state is BreakerState.CLOSED:
            return Admission.ALLOW
        if state is BreakerState.OPEN and now < open_until:
            return Admission.DENY
        assert self._redis is not None  # noqa: S101 (narrowing: _redis_ok checked it)
        try:
            verdict = await self._redis.admit(target.name, target.breaker, now)
        except (RedisError, OSError) as exc:
            self._redis_failed(exc)
            verdict = self.memory.admit(target.name, target.breaker, now)
        return self._observe(target.name, verdict)

    def record(self, target: ModelTarget, *, ok: bool, probe: bool) -> None:
        """Report an attempt's outcome. Never blocks: Redis writes run in the background."""
        now = self._clock()
        if not self._redis_ok():
            verdict = self.memory.record(target.name, target.breaker, now, ok=ok, probe=probe)
            self._observe(target.name, verdict)
            return
        task = asyncio.create_task(self._record_redis(target, now, ok=ok, probe=probe))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def aclose(self) -> None:
        """Let background writes finish (bounded) at shutdown."""
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=_SHUTDOWN_WAIT)

    def _local_state(self, name: str) -> BreakerState:
        if self._redis_ok():
            return self._view.get(name, (BreakerState.CLOSED, 0.0))[0]
        return self.memory.state(name, self._clock())

    async def _record_redis(
        self, target: ModelTarget, now: float, *, ok: bool, probe: bool
    ) -> None:
        assert self._redis is not None  # noqa: S101
        try:
            verdict = await self._redis.record(target.name, target.breaker, now, ok=ok, probe=probe)
        except (RedisError, OSError) as exc:
            self._redis_failed(exc)
            verdict = self.memory.record(target.name, target.breaker, now, ok=ok, probe=probe)
        self._observe(target.name, verdict)

    def _observe(self, name: str, verdict: Verdict) -> Admission:
        previous = self._view.get(name)
        self._view[name] = (verdict.state, verdict.open_until)
        if previous is None or previous[0] is not verdict.state:  # the gauge, on change only
            BREAKER_STATE.labels(name).set(_STATE_VALUE[verdict.state])
        if verdict.transitioned:
            BREAKER_TRANSITIONS.labels(name, verdict.state.value).inc()
            log = _log.warning if verdict.state is BreakerState.OPEN else _log.info
            log("breaker.transition", model=name, state=verdict.state.value)
        return verdict.admission

    def _redis_ok(self) -> bool:
        return self._redis is not None and self._monotonic() >= self._redis_down_until

    def _redis_failed(self, exc: Exception) -> None:
        if self._monotonic() >= self._redis_down_until:
            _log.warning(
                "breaker.redis_unavailable",
                error=f"{type(exc).__name__}: {exc}",
                fallback="in-process breakers",
                retry_in_s=_REDIS_RETRY_AFTER,
            )
        self._redis_down_until = self._monotonic() + _REDIS_RETRY_AFTER
