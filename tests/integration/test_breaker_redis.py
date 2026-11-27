"""The Redis circuit-breaker store against the compose stack's Redis.

The same scenario table as the in-process store (``tests/breaker_cases.py``) runs here
against the Lua scripts, which is what keeps the two implementations equivalent. Each
test uses its own breaker name and deletes it afterwards.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis

from switchboard.reliability.breaker import (
    Admission,
    Breakers,
    BreakerState,
    RedisBreakerStore,
    redis_key,
)
from switchboard.routing.config import Capabilities, ModelDefaults, ModelKind, Timeouts
from switchboard.routing.table import ModelTarget
from tests.breaker_cases import BASE, CASES, POLICY, policy_for

pytestmark = pytest.mark.integration


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    url = os.environ.get("SB_REDIS_URL")
    if not url:
        pytest.skip("SB_REDIS_URL is not set (run via `make test-integration`)")
    client = Redis.from_url(url)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def name(redis: Redis) -> AsyncIterator[str]:
    breaker = f"test/{uuid.uuid4().hex}"
    try:
        yield breaker
    finally:
        await redis.delete(redis_key(breaker))


@pytest.mark.parametrize("case", list(CASES))
async def test_redis_store_scenarios(redis: Redis, name: str, case: str) -> None:
    store = RedisBreakerStore(redis)
    policy = policy_for(case)
    for i, step in enumerate(CASES[case]):
        if step[0] == "admit":
            _, t, admission, state = step
            verdict = await store.admit(name, policy, BASE + t)
            assert (verdict.admission, verdict.state) == (admission, state), f"step {i}: {step}"
        else:
            _, t, ok, probe, state, transitioned = step
            verdict = await store.record(name, policy, BASE + t, ok=ok, probe=probe)
            assert (verdict.state, verdict.transitioned) == (state, transitioned), f"step {i}"


async def test_concurrent_admits_in_half_open_get_exactly_the_probe_limit(
    redis: Redis, name: str
) -> None:
    store = RedisBreakerStore(redis)
    for t in range(4):
        await store.record(name, POLICY, BASE + t, ok=False, probe=False)  # open until +13
    verdicts = await asyncio.gather(*(store.admit(name, POLICY, BASE + 13) for _ in range(50)))
    admissions = [v.admission for v in verdicts]
    assert admissions.count(Admission.PROBE) == POLICY.half_open_probes
    assert admissions.count(Admission.DENY) == 50 - POLICY.half_open_probes


def target(name: str) -> ModelTarget:
    return ModelTarget(
        name=name,
        provider="test",
        upstream_model="m",
        kind=ModelKind.CHAT,
        capabilities=Capabilities(),
        defaults=ModelDefaults(),
        timeouts=Timeouts(),
        breaker=POLICY,
    )


class Clock:
    def __init__(self) -> None:
        self.now = BASE

    def __call__(self) -> float:
        return self.now


async def test_replicas_share_one_breaker(redis: Redis, name: str) -> None:
    """Two gateway replicas (two Breakers) on one Redis: what one opens, the other
    learns from its next reported outcome, and half-open probing is shared."""
    clock = Clock()
    one, two = Breakers(redis, clock=clock), Breakers(redis, clock=clock)
    t = target(name)
    for _ in range(4):
        one.record(t, ok=False, probe=False)
        await one.aclose()  # wait for the background writes, in order
    assert await one.admit(t) is Admission.DENY  # decided locally, no round trip
    assert await RedisBreakerStore(redis).state(name) is BreakerState.OPEN

    assert await two.admit(t) is Admission.ALLOW  # not seen yet: its view is closed
    two.record(t, ok=True, probe=False)
    await two.aclose()
    assert await two.admit(t) is Admission.DENY  # the store's reply updated its view

    clock.now += POLICY.cooldown
    admissions = [await one.admit(t), await two.admit(t), await one.admit(t)]
    assert admissions.count(Admission.PROBE) == POLICY.half_open_probes
