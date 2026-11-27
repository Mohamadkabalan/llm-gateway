"""Admission and settlement: the real Lua scripts on fakeredis, with a test clock.

The same scripts run against real Redis in ``tests/integration``.
"""

import asyncio
import random
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import fakeredis
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from redis.exceptions import ConnectionError as RedisConnectionError

from switchboard.accounting.pricing import NANO, Cost
from switchboard.auth.context import KeyContext, Limits
from switchboard.limits.limiter import (
    Admitted,
    BudgetExceededError,
    Limiter,
    LimiterUnavailableError,
    RateInfo,
    RateLimitedError,
    Settled,
)
from tests.support import key_context

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.ms = 1_764_000_000_000

    def __call__(self) -> int:
        return self.ms

    def advance(self, seconds: float) -> None:
        self.ms += int(seconds * 1000)


def nano(n: int) -> Cost:
    return Cost(Decimal(n) / NANO)


def usd(n: int) -> Decimal:
    return Decimal(n) / NANO


def redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


def limiter(r: fakeredis.FakeAsyncRedis, clock: Clock, **kwargs: Any) -> Limiter:
    return Limiter(r, now_ms=clock, **kwargs)


_ids = iter(range(10**9))


async def admit(
    lim: Limiter, key: KeyContext, *, tokens: int = 10, cost: int = 0, hold_s: float = 900
) -> Admitted:
    admitted = await lim.admit(
        key, request_id=f"r{next(_ids)}", tokens=tokens, cost=nano(cost), hold_s=hold_s, now=NOW
    )
    assert admitted is not None
    return admitted


async def test_keys_without_limits_skip_redis() -> None:
    r = redis()
    lim = limiter(r, Clock())
    assert await lim.admit(key_context(), request_id="x", tokens=10, cost=nano(1)) is None
    assert await r.keys("*") == []


async def test_rpm_bucket_refills_over_a_minute() -> None:
    clock = Clock()
    lim = limiter(redis(), clock)
    key = key_context(key_limits=Limits(rpm=2))
    first = await admit(lim, key)
    assert first.rate.limit_requests == 2
    assert first.rate.remaining_requests == 1
    await admit(lim, key)
    with pytest.raises(RateLimitedError) as caught:
        await admit(lim, key)
    assert caught.value.bucket.kind == "rpm"
    assert caught.value.retry_after == 30  # one token at 2 per minute
    clock.advance(30)
    await admit(lim, key)


async def test_a_rejection_takes_nothing_from_any_bucket() -> None:
    lim = limiter(redis(), Clock())
    key = key_context(key_limits=Limits(rpm=10, tpm=100))
    await admit(lim, key, tokens=150)  # more than the bucket holds: needs it full
    with pytest.raises(RateLimitedError) as caught:
        await admit(lim, key, tokens=10)
    assert caught.value.bucket.kind == "tpm"
    assert caught.value.rate.remaining_requests == 9  # the RPM bucket was not charged


async def test_team_buckets_are_shared_by_its_keys() -> None:
    lim = limiter(redis(), Clock())
    team = Limits(rpm=2)
    one = key_context("sb_live_" + "a" * 43, team_limits=team)
    two = key_context("sb_live_" + "b" * 43, team_limits=team)
    await admit(lim, one)
    await admit(lim, two)
    with pytest.raises(RateLimitedError) as caught:
        await admit(lim, one)
    assert caught.value.bucket.scope == "team"


async def test_settlement_corrects_the_tpm_bucket() -> None:
    clock = Clock()
    lim = limiter(redis(), clock)
    key = key_context(key_limits=Limits(tpm=1000))
    admitted = await admit(lim, key, tokens=600)
    await lim.settle_now(admitted, key, tokens=100, cost=nano(0))  # 500 tokens back
    after = await admit(lim, key, tokens=1)
    assert after.rate.remaining_tokens == 899


async def test_hard_budget_reserves_then_settles() -> None:
    lim = limiter(redis(), Clock())
    key = key_context(key_limits=Limits(hard_usd=usd(1_000_000)))
    first = await admit(lim, key, cost=600_000)
    with pytest.raises(BudgetExceededError) as caught:  # 600k reserved + 600k > 1M
        await admit(lim, key, cost=600_000)
    assert caught.value.budget.scope == "key"
    await lim.settle_now(first, key, tokens=None, cost=nano(100_000))
    await admit(lim, key, cost=600_000)  # 100k spent + 600k fits


async def test_an_unsettled_reservation_is_released_after_its_hold() -> None:
    clock = Clock()
    lim = limiter(redis(), clock)
    key = key_context(key_limits=Limits(hard_usd=usd(1_000)))
    await admit(lim, key, cost=900, hold_s=10)  # the replica "dies": never settled
    with pytest.raises(BudgetExceededError):
        await admit(lim, key, cost=900)
    clock.advance(11)
    await admit(lim, key, cost=900)


async def test_soft_alert_fires_once_per_period() -> None:
    fired: list[Settled] = []
    lim = limiter(redis(), Clock(), on_alert=lambda s, k, p: fired.append(s))
    key = key_context(key_limits=Limits(soft_usd=usd(150)))
    for _ in range(3):
        admitted = await admit(lim, key, cost=100)
        await lim.settle_now(admitted, key, tokens=None, cost=nano(100))
    assert len(fired) == 1
    assert fired[0].spent_nano == 200  # crossed on the second request


async def test_missing_counters_are_rebuilt_from_the_ledger_once() -> None:
    calls: list[tuple[str, UUID]] = []

    async def ledger(scope: str, scope_id: UUID, start: datetime, end: datetime) -> Decimal:
        calls.append((scope, scope_id))
        assert (start.month, end.month) == (11, 12)
        return usd(900)

    lim = limiter(redis(), Clock(), spend_source=ledger)
    key = key_context(key_limits=Limits(hard_usd=usd(1_000)))
    with pytest.raises(BudgetExceededError):
        await admit(lim, key, cost=200)  # 900 already spent this month
    await admit(lim, key, cost=100)
    assert calls == [("key", key.key_id)]


class BrokenScript:
    async def __call__(self, **_: Any) -> Any:
        raise RedisConnectionError("connection refused")


class BrokenRedis:
    def register_script(self, _source: str) -> BrokenScript:
        return BrokenScript()


@pytest.mark.parametrize(
    ("policy", "limits", "outcome"),
    [
        ("open", Limits(hard_usd=Decimal(1)), "degraded"),
        ("closed", Limits(hard_usd=Decimal(1)), "refused"),
        ("closed", Limits(rpm=5), "degraded"),  # rate limits always fail open
    ],
)
async def test_redis_failure_policy(policy: str, limits: Limits, outcome: str) -> None:
    lim = Limiter(BrokenRedis(), budget_on_failure=policy)  # type: ignore[arg-type]
    key = key_context(key_limits=limits)
    if outcome == "refused":
        with pytest.raises(LimiterUnavailableError):
            await lim.admit(key, request_id="x", tokens=1, cost=nano(1))
        return
    admitted = await lim.admit(key, request_id="x", tokens=1, cost=nano(1))
    assert admitted is not None
    assert admitted.degraded


@pytest.mark.parametrize(
    ("remaining", "reset"),
    [(59, "1s"), (30, "30s"), (0, "1m0s")],
)
def test_rate_limit_headers(remaining: int, reset: str) -> None:
    headers = RateInfo(limit_requests=60, remaining_requests=remaining).headers()
    assert headers == {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": str(remaining),
        "x-ratelimit-reset-requests": reset,
    }


# --- properties (ADR-0005) -------------------------------------------------------------

requests_strategy = st.lists(
    st.tuples(st.integers(min_value=1, max_value=500), st.integers(min_value=0, max_value=900)),
    min_size=1,
    max_size=20,
)


@settings(max_examples=30, deadline=None)
@given(requests=requests_strategy, seed=st.integers(min_value=0, max_value=2**32))
def test_budget_overshoot_is_bounded_and_no_reservation_leaks(
    requests: list[tuple[int, int]], seed: int
) -> None:
    """Admissions and settlements interleave at random (requests in flight at once).

    * Spend can exceed the hard limit only by what in-flight requests cost beyond their
      estimates: spent <= limit + sum(max(0, actual - estimate)) over admitted requests.
    * When every estimate is high enough, the limit is never exceeded.
    * Once everything has settled, nothing is left reserved.
    """
    asyncio.run(_simulate(requests, random.Random(seed)))


async def _simulate(requests: list[tuple[int, int]], rnd: random.Random) -> None:
    r = redis()
    lim = limiter(r, Clock())
    hard = 2_000
    key = key_context(key_limits=Limits(hard_usd=usd(hard)))
    queue = list(requests)
    in_flight: list[tuple[Admitted, int, int]] = []
    admitted: list[tuple[int, int]] = []
    first: Admitted | None = None
    while queue or in_flight:
        if queue and (not in_flight or rnd.random() < 0.6):
            estimate, actual = queue.pop(0)
            try:
                a = await admit(lim, key, cost=estimate)
            except BudgetExceededError:
                continue
            first = first or a
            in_flight.append((a, estimate, actual))
            admitted.append((estimate, actual))
        else:
            a, _, actual = in_flight.pop(rnd.randrange(len(in_flight)))
            await lim.settle_now(a, key, tokens=None, cost=nano(actual))
    assert first is not None  # the first request always fits (estimate <= 500 < 2000)
    budget = first.budgets[0]  # every request shares the key's one budget
    values = await r.hmget(budget.spend_key, ["spent", "reserved"])
    spent, reserved = (int(v or 0) for v in values)
    assert reserved == 0
    assert await r.zcard(budget.resv_key) == 0  # an emptied sorted set is gone: 0
    overrun = sum(max(0, actual - estimate) for estimate, actual in admitted)
    assert spent <= hard + overrun
    if all(actual <= estimate for estimate, actual in admitted):
        assert spent <= hard
