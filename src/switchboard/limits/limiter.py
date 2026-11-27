"""Admission (rate limits and budget reservation) and settlement (PLAN.md §7, ADR-0005).

``Limiter.admit`` runs before any upstream call: one Redis round trip, and none at all
for keys whose key and team have no limits. ``Limiter.settle`` runs when the request
ends, in the background. Money is integer nano-USD in Redis.

Redis failure policy (PLAN.md §7): rate limits fail open; budgets fail open unless
``budget_on_redis_failure`` is ``closed``. Either way the degraded path is counted in
``switchboard_degraded_total{subsystem}`` and logged.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from switchboard.accounting.pricing import NANO, Cost
from switchboard.auth.context import KeyContext, Limits
from switchboard.limits.scripts import ADMIT, SETTLE
from switchboard.observability.metrics import (
    BUDGET_BLOCKS,
    BUDGET_UTILIZATION,
    DEGRADED,
    RATELIMIT_REJECTIONS,
)

_log = structlog.get_logger("switchboard.limits")

Scope = Literal["key", "team"]
Kind = Literal["rpm", "tpm"]
FailurePolicy = Literal["open", "closed"]

_BUCKET_TTL_MS = 121_000  # an emptied (or indebted) bucket is full again after 2 min
_SPEND_TTL_S = 40 * 86_400  # a month's counters outlive the month
_ALERT_TTL_S = 40 * 86_400


# The ledger's spend for one scope in one month, to rebuild a missing counter.
type SpendSource = Callable[[Scope, UUID, datetime, datetime], Awaitable[Decimal]]


@dataclass(frozen=True, slots=True)
class Bucket:
    scope: Scope
    kind: Kind
    key: str
    capacity: int
    cost: int


@dataclass(frozen=True, slots=True)
class Budget:
    scope: Scope
    scope_id: UUID
    spend_key: str
    resv_key: str
    alert_key: str
    hard_nano: int  # -1: no hard limit
    soft_nano: int  # -1: no soft limit
    limits: Limits


@dataclass(slots=True)
class RateInfo:
    """What the ``x-ratelimit-*`` headers report: the key's buckets if it has its own
    limits, else the team's."""

    limit_requests: int | None = None
    remaining_requests: int | None = None
    limit_tokens: int | None = None
    remaining_tokens: int | None = None

    def headers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for kind, limit, remaining in (
            ("requests", self.limit_requests, self.remaining_requests),
            ("tokens", self.limit_tokens, self.remaining_tokens),
        ):
            if limit is None or remaining is None:
                continue
            left = max(remaining, 0)
            out[f"x-ratelimit-limit-{kind}"] = str(limit)
            out[f"x-ratelimit-remaining-{kind}"] = str(left)
            out[f"x-ratelimit-reset-{kind}"] = _duration((limit - left) * 60.0 / limit)
        return out


@dataclass(slots=True)
class Admitted:
    member: str
    period: str
    buckets: list[Bucket]
    budgets: list[Budget]
    estimate: Cost
    tokens_charged: int
    rate: RateInfo = field(default_factory=RateInfo)
    degraded: bool = False  # admitted without checks (Redis unavailable)
    hold_ms: int = 900_000


class RateLimitedError(Exception):
    def __init__(self, bucket: Bucket, retry_after: float, rate: RateInfo) -> None:
        super().__init__(f"{bucket.scope} {bucket.kind} limit")
        self.bucket = bucket
        self.retry_after = retry_after
        self.rate = rate


class BudgetExceededError(Exception):
    def __init__(self, budget: Budget) -> None:
        super().__init__(f"{budget.scope} budget")
        self.budget = budget


class LimiterUnavailableError(Exception):
    """Budgets could not be checked and the policy is to fail closed."""


@dataclass(frozen=True, slots=True)
class Settled:
    budget: Budget
    spent_nano: int
    alert: bool


type AlertHook = Callable[[Settled, KeyContext, str], None]


class Limiter:
    def __init__(
        self,
        redis: Redis,
        *,
        spend_source: SpendSource | None = None,
        budget_on_failure: FailurePolicy = "open",
        reservation_ttl_s: float = 900.0,
        on_alert: AlertHook | None = None,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        """``now_ms`` overrides Redis ``TIME`` (tests only)."""
        self._redis = redis
        self._admit = redis.register_script(ADMIT)
        self._settle = redis.register_script(SETTLE)
        self._spend_source = spend_source
        self._policy = budget_on_failure
        self._reservation_ms = int(reservation_ttl_s * 1000)
        self._on_alert = on_alert
        self._now_ms = now_ms
        self._pending: set[asyncio.Task[None]] = set()

    async def admit(
        self,
        key: KeyContext,
        *,
        request_id: str,
        tokens: int,
        cost: Cost,
        hold_s: float | None = None,
        now: datetime | None = None,
    ) -> Admitted | None:
        """Check and charge every limit, or raise. None: nothing to check for this key.
        ``hold_s`` is how long the budget reservation outlives a lost settlement."""
        if not key.limited:
            return None
        period = (now or datetime.now(UTC)).strftime("%Y%m")
        buckets = _buckets(key, tokens)
        budgets = _budgets(key, period, cost.nano)
        admitted = Admitted(request_id, period, buckets, budgets, cost, tokens)
        admitted.hold_ms = int(hold_s * 1000) if hold_s is not None else self._reservation_ms
        try:
            reply = await self._run_admit(admitted)
            if reply[0] == "rebuild":
                await self._rebuild(key, admitted, [int(i) for i in reply[1:]])
                reply = await self._run_admit(admitted)
        except (RedisError, OSError) as exc:
            return self._degraded(admitted, exc)
        verdict = reply[0]
        if verdict == "budget":
            budget = budgets[int(reply[1]) - 1]
            BUDGET_BLOCKS.labels(budget.scope).inc()
            raise BudgetExceededError(budget)
        if verdict == "rate":
            bucket = buckets[int(reply[1]) - 1]
            RATELIMIT_REJECTIONS.labels(bucket.scope, bucket.kind).inc()
            levels = [int(v) for v in reply[3:]]
            raise RateLimitedError(bucket, int(reply[2]) / 1000, _rate_info(buckets, levels))
        if verdict != "ok":  # "rebuild" twice: the counter vanished again
            return self._degraded(admitted, RuntimeError(f"unexpected admission reply {verdict}"))
        admitted.rate = _rate_info(buckets, [int(v) for v in reply[2:]])
        return admitted

    def settle(
        self, admitted: Admitted | None, key: KeyContext, *, tokens: int | None, cost: Cost
    ) -> None:
        """Turn the reservation into spend; never blocks the response."""
        if admitted is None or admitted.degraded:
            return
        task = asyncio.create_task(self._settle_now(admitted, key, tokens=tokens, cost=cost))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def aclose(self) -> None:
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=2.0)

    async def settle_now(
        self, admitted: Admitted, key: KeyContext, *, tokens: int | None, cost: Cost
    ) -> list[Settled]:
        """Settle and wait (tests and the shutdown path); ``settle`` is the usual call."""
        args: list[str | int] = [len(_tpm(admitted)), len(admitted.budgets), admitted.member]
        args.append(_ALERT_TTL_S)
        actual_tokens = admitted.tokens_charged if tokens is None else tokens
        for bucket in _tpm(admitted):
            charged = min(bucket.cost, bucket.capacity)
            args += [bucket.capacity, actual_tokens - charged]
        for budget in admitted.budgets:
            args += [cost.nano, admitted.estimate.nano, budget.soft_nano]
        keys = [b.key for b in _tpm(admitted)]
        for budget in admitted.budgets:
            keys += [budget.spend_key, budget.resv_key, budget.alert_key]
        reply = await self._settle(keys=keys, args=args)
        results = []
        for j, budget in enumerate(admitted.budgets):
            spent, fired = int(reply[2 * j]), bool(int(reply[2 * j + 1]))
            results.append(Settled(budget, spent, fired))
            if budget.scope == "team" and budget.hard_nano > 0 and spent >= 0:
                BUDGET_UTILIZATION.labels(key.team_name).set(spent / budget.hard_nano)
            if fired and self._on_alert is not None:
                self._on_alert(results[-1], key, admitted.period)
        return results

    async def _settle_now(
        self, admitted: Admitted, key: KeyContext, *, tokens: int | None, cost: Cost
    ) -> None:
        try:
            await self.settle_now(admitted, key, tokens=tokens, cost=cost)
        except (RedisError, OSError) as exc:
            # The reservation expires on its own (reservation_ttl_s); the spend is in the
            # ledger and is rebuilt from it if the counter is ever lost.
            DEGRADED.labels("settlement").inc()
            _log.warning("limits.settle_failed", detail=str(exc))

    async def _run_admit(self, admitted: Admitted) -> list[str]:
        now = "" if self._now_ms is None else str(self._now_ms())
        args: list[str | int] = [
            now,
            len(admitted.buckets),
            len(admitted.budgets),
            admitted.member,
            admitted.hold_ms,
            _SPEND_TTL_S,
            _BUCKET_TTL_MS,
        ]
        for bucket in admitted.buckets:
            args += [bucket.capacity, bucket.cost]
        for budget in admitted.budgets:
            args += [budget.hard_nano, admitted.estimate.nano]
        keys = [b.key for b in admitted.buckets]
        for budget in admitted.budgets:
            keys += [budget.spend_key, budget.resv_key]
        reply = await self._admit(keys=keys, args=args)
        return [part.decode() if isinstance(part, bytes) else str(part) for part in reply]

    async def _rebuild(self, key: KeyContext, admitted: Admitted, missing: list[int]) -> None:
        """Seed missing counters from the ledger. ``HSETNX`` lets concurrent rebuilds race
        safely: the first value wins."""
        start, end = month_bounds(admitted.period)
        async with self._redis.pipeline(transaction=False) as pipe:
            for index in missing:
                budget = admitted.budgets[index - 1]
                spent = Decimal(0)
                if self._spend_source is not None:
                    spent = await self._spend_source(budget.scope, budget.scope_id, start, end)
                pipe.hsetnx(budget.spend_key, "spent", int(spent * NANO))
                pipe.hsetnx(budget.spend_key, "reserved", 0)
                pipe.expire(budget.spend_key, _SPEND_TTL_S)
            await pipe.execute()
        _log.info("limits.counters_rebuilt", key=key.key_prefix, count=len(missing))

    def _degraded(self, admitted: Admitted, exc: Exception) -> Admitted:
        has_budget = bool(admitted.budgets)
        DEGRADED.labels("limits").inc()
        _log.warning(
            "limits.unavailable",
            detail=f"{type(exc).__name__}: {exc}",
            budgets="blocked" if has_budget and self._policy == "closed" else "not enforced",
        )
        if has_budget and self._policy == "closed":
            raise LimiterUnavailableError(str(exc)) from exc
        admitted.degraded = True
        return admitted


def _tpm(admitted: Admitted) -> list[Bucket]:
    return [b for b in admitted.buckets if b.kind == "tpm"]


def _tag(key: KeyContext) -> str:
    return f"sb:{{t:{key.team_id}}}"


def _scopes(key: KeyContext) -> list[tuple[Scope, Limits, UUID, str]]:
    return [
        ("key", key.key_limits, key.key_id, f"key:{key.key_id}"),
        ("team", key.team_limits, key.team_id, "team"),
    ]


def _buckets(key: KeyContext, tokens: int) -> list[Bucket]:
    out: list[Bucket] = []
    for scope, limits, _, suffix in _scopes(key):
        if limits.rpm:
            out.append(Bucket(scope, "rpm", f"{_tag(key)}:rl:{suffix}:rpm", limits.rpm, 1))
        if limits.tpm:
            cost = max(tokens, 1)
            out.append(Bucket(scope, "tpm", f"{_tag(key)}:rl:{suffix}:tpm", limits.tpm, cost))
    return out


def _budgets(key: KeyContext, period: str, estimate_nano: int) -> list[Budget]:
    out: list[Budget] = []
    for scope, limits, scope_id, suffix in _scopes(key):
        if not limits.has_budget:
            continue
        out.append(
            Budget(
                scope=scope,
                scope_id=scope_id,
                spend_key=f"{_tag(key)}:spend:{suffix}:{period}",
                resv_key=f"{_tag(key)}:resv:{suffix}:{period}",
                alert_key=f"{_tag(key)}:alert:{suffix}:{period}:soft",
                hard_nano=_nano(limits.hard_usd),
                soft_nano=_nano(limits.soft_usd),
                limits=limits,
            )
        )
    return out


def month_bounds(period: str) -> tuple[datetime, datetime]:
    """``"202611"`` → (2026-11-01, 2026-12-01), both UTC midnight."""
    start = datetime.strptime(period, "%Y%m").replace(tzinfo=UTC)
    if start.month == 12:
        return start, start.replace(year=start.year + 1, month=1)
    return start, start.replace(month=start.month + 1)


def _nano(usd: Decimal | None) -> int:
    return -1 if usd is None else int(usd * NANO)


def _rate_info(buckets: list[Bucket], levels: list[int]) -> RateInfo:
    info = RateInfo()
    has_key = any(b.scope == "key" for b in buckets)
    for bucket, level in zip(buckets, levels, strict=False):
        if has_key and bucket.scope != "key":
            continue
        if bucket.kind == "rpm":
            info.limit_requests, info.remaining_requests = bucket.capacity, level
        else:
            info.limit_tokens, info.remaining_tokens = bucket.capacity, level
    return info


def _duration(seconds: float) -> str:
    """OpenAI's reset format: ``20ms``, ``1.5s``, ``6m0s``."""
    if seconds < 1:
        return f"{max(math.ceil(seconds * 1000), 0)}ms"
    if seconds < 60:
        return f"{seconds:.3g}s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m{rest}s"
