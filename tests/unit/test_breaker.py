"""Circuit breakers: the shared scenario table on the in-process store, a property test
for half-open probing, and the facade's Redis-failure fallback."""

import asyncio
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from redis.exceptions import ConnectionError as RedisConnectionError

from switchboard.reliability.breaker import (
    Admission,
    Breakers,
    BreakerState,
    MemoryBreakerStore,
)
from switchboard.routing.config import (
    BreakerPolicy,
    Capabilities,
    ModelDefaults,
    ModelKind,
    Timeouts,
)
from switchboard.routing.table import ModelTarget
from tests.breaker_cases import BASE, CASES, POLICY, Step, policy_for


def run_case(store: MemoryBreakerStore, steps: list[Step], policy: BreakerPolicy) -> None:
    for i, step in enumerate(steps):
        if step[0] == "admit":
            _, t, admission, state = step
            verdict = store.admit("p/m", policy, BASE + t)
            assert (verdict.admission, verdict.state) == (admission, state), f"step {i}: {step}"
        else:
            _, t, ok, probe, state, transitioned = step
            verdict = store.record("p/m", policy, BASE + t, ok=ok, probe=probe)
            assert (verdict.state, verdict.transitioned) == (state, transitioned), f"step {i}"


@pytest.mark.parametrize("name", list(CASES))
def test_memory_store_scenarios(name: str) -> None:
    run_case(MemoryBreakerStore(), CASES[name], policy_for(name))


def test_an_idle_open_breaker_expires_like_the_redis_key() -> None:
    store = MemoryBreakerStore()
    for t in range(4):
        store.record("p/m", POLICY, BASE + t, ok=False, probe=False)
    assert store.state("p/m", BASE + 4) is BreakerState.OPEN
    # Redis keeps an open breaker for cooldown + window + one bucket: 10 + 30 + 3 s.
    assert store.state("p/m", BASE + 3 + 50) is BreakerState.CLOSED


@given(
    probes=st.integers(min_value=1, max_value=5),
    ops=st.lists(st.sampled_from(["admit", "ok", "fail"]), min_size=1, max_size=60),
)
def test_half_open_never_admits_more_than_the_probe_limit(probes: int, ops: list[str]) -> None:
    """However admits and probe outcomes interleave within one round, at most
    ``half_open_probes`` requests get through, and a failed probe stops them all."""
    policy = BreakerPolicy(min_requests=1, cooldown=10, half_open_probes=probes)
    store = MemoryBreakerStore()
    store.record("p/m", policy, BASE, ok=False, probe=False)  # open until BASE + 10
    now = BASE + 10  # every op happens at the end of the cooldown: one round
    admitted = 0
    outstanding = 0
    reopened = False
    for op in ops:
        if op == "admit":
            verdict = store.admit("p/m", policy, now)
            if verdict.admission is Admission.PROBE:
                admitted += 1
                outstanding += 1
            assert verdict.admission is not Admission.ALLOW  # half-open never just allows
            if reopened:
                assert verdict.admission is Admission.DENY
        elif outstanding:
            outstanding -= 1
            verdict = store.record("p/m", policy, now, ok=op == "ok", probe=True)
            if op == "fail" and verdict.transitioned:
                reopened = True
            if verdict.state is BreakerState.CLOSED:
                return  # every probe succeeded; the round is over
        assert admitted <= probes


# --- the facade ------------------------------------------------------------------------


def target(name: str = "p/m", policy: BreakerPolicy = POLICY) -> ModelTarget:
    return ModelTarget(
        name=name,
        provider="p",
        upstream_model="m",
        kind=ModelKind.CHAT,
        capabilities=Capabilities(),
        defaults=ModelDefaults(),
        timeouts=Timeouts(),
        breaker=policy,
    )


class Clock:
    def __init__(self, now: float = BASE) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_breakers_without_redis_use_the_in_process_store() -> None:
    clock = Clock()
    breakers = Breakers(clock=clock)
    t = target()
    for _ in range(4):
        assert await breakers.admit(t) is Admission.ALLOW
        breakers.record(t, ok=False, probe=False)
    assert await breakers.admit(t) is Admission.DENY
    clock.now += 10
    assert await breakers.admit(t) is Admission.PROBE


class BrokenScript:
    async def __call__(self, **_: Any) -> Any:
        raise RedisConnectionError("connection refused")


class BrokenRedis:
    """A Redis client whose every call fails, as when Redis is down."""

    def register_script(self, _source: str) -> BrokenScript:
        return BrokenScript()


async def test_redis_failure_falls_back_to_in_process_breakers() -> None:
    clock = Clock()
    breakers = Breakers(BrokenRedis(), clock=clock)  # type: ignore[arg-type]
    t = target()
    for _ in range(4):
        breakers.record(t, ok=False, probe=False)  # the first write fails over to memory
        await asyncio.sleep(0)
    await breakers.aclose()
    # Redis is now considered down for a while; the in-process breaker decides.
    assert await breakers.admit(t) is Admission.DENY
