"""The failure-classification table (PLAN.md §2.2), backoff and the ratio budgets."""

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.reliability.backoff import full_jitter, retry_delay
from switchboard.reliability.budget import RatioBudget
from switchboard.reliability.policy import (
    DISPOSITIONS,
    Fallback,
    may_fall_back,
    may_retry,
    stops_the_walk,
)
from switchboard.routing.config import Capabilities, ModelDefaults, ModelKind, RetryPolicy, Timeouts
from switchboard.routing.table import ModelTarget

# One row per kind: (retry same target, fall back, counts toward the breaker).
TABLE = {
    ErrorKind.CONNECT: (True, Fallback.YES, True),
    ErrorKind.TIMEOUT: (True, Fallback.YES, True),
    ErrorKind.RATE_LIMITED: (True, Fallback.YES, True),
    ErrorKind.QUOTA: (False, Fallback.YES, True),
    ErrorKind.OVERLOADED: (True, Fallback.YES, True),
    ErrorKind.PROTOCOL: (True, Fallback.YES, True),
    ErrorKind.AUTH: (False, Fallback.YES, True),
    ErrorKind.CONTEXT_LENGTH: (False, Fallback.LARGER_CONTEXT, False),
    ErrorKind.BAD_REQUEST: (False, Fallback.NO, False),
    ErrorKind.INVALID_REQUEST: (False, Fallback.YES, False),
    ErrorKind.CIRCUIT_OPEN: (False, Fallback.YES, False),
    ErrorKind.REFUSAL: (False, Fallback.YES, False),
}


def target(name: str, max_input: int | None = None) -> ModelTarget:
    return ModelTarget(
        name=name,
        provider=name.split("/", maxsplit=1)[0],
        upstream_model=name,
        kind=ModelKind.CHAT,
        capabilities=Capabilities(max_input_tokens=max_input),
        defaults=ModelDefaults(),
        timeouts=Timeouts(),
    )


def test_every_error_kind_has_a_row() -> None:
    assert set(DISPOSITIONS) == set(ErrorKind)
    assert set(TABLE) == set(ErrorKind)


@pytest.mark.parametrize("kind", list(ErrorKind))
def test_classification_row(kind: ErrorKind) -> None:
    retry, fallback, breaker = TABLE[kind]
    row = DISPOSITIONS[kind]
    assert (row.retry, row.fallback, row.breaker) == (retry, fallback, breaker)
    error = ProviderError(kind, "x")
    assert may_retry(error) is retry
    assert stops_the_walk(error) is (fallback is Fallback.NO)


@pytest.mark.parametrize(
    ("kind", "header", "expected"),
    [
        (ErrorKind.OVERLOADED, False, False),  # x-should-retry: false wins
        (ErrorKind.BAD_REQUEST, True, True),  # x-should-retry: true wins
        (ErrorKind.OVERLOADED, None, True),
    ],
)
def test_x_should_retry_overrides_the_retry_column(
    kind: ErrorKind, header: bool | None, expected: bool
) -> None:
    assert may_retry(ProviderError(kind, "x", should_retry=header)) is expected


@pytest.mark.parametrize(
    ("failed", "candidate", "expected"),
    [(100, 200, True), (100, 100, False), (100, 50, False), (None, 200, False), (100, None, False)],
)
def test_context_length_falls_back_only_to_larger_models(
    failed: int | None, candidate: int | None, expected: bool
) -> None:
    error = ProviderError(ErrorKind.CONTEXT_LENGTH, "too long")
    assert may_fall_back(error, target("a/x", failed), target("b/y", candidate)) is expected


def test_bad_request_never_falls_back() -> None:
    error = ProviderError(ErrorKind.BAD_REQUEST, "no")
    assert not may_fall_back(error, target("a/x"), target("b/y"))


# --- backoff ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=20), st.integers(min_value=0, max_value=2**32))
def test_full_jitter_stays_within_bounds(retry_index: int, seed: int) -> None:
    policy = RetryPolicy(base_delay=0.25, max_delay=4.0)
    delay = full_jitter(retry_index, policy, random.Random(seed))
    assert 0 <= delay <= min(4.0, 0.25 * 2**retry_index)


def test_full_jitter_cap_grows_then_stops_at_max_delay() -> None:
    policy = RetryPolicy(base_delay=0.25, max_delay=1.0)
    rng = random.Random(1)
    maxima = [max(full_jitter(i, policy, rng) for _ in range(2000)) for i in range(4)]
    assert maxima[0] <= 0.25 < maxima[1] <= 0.5 < maxima[2] <= 1.0
    assert maxima[3] <= 1.0


def test_retry_after_within_the_cap_is_honoured() -> None:
    error = ProviderError(ErrorKind.RATE_LIMITED, "slow down", retry_after=3.0)
    delay = retry_delay(error, 0, RetryPolicy(honor_retry_after_up_to=10), random.Random(0))
    assert delay == 3.0


def test_retry_after_past_the_cap_means_do_not_retry() -> None:
    error = ProviderError(ErrorKind.RATE_LIMITED, "slow down", retry_after=30.0)
    assert retry_delay(error, 0, RetryPolicy(honor_retry_after_up_to=10), random.Random(0)) is None


# --- ratio budgets ---------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_budget_allows_the_minimum_then_the_ratio() -> None:
    clock = Clock()
    budget = RatioBudget(0.2, 2, clock=clock)
    assert [budget.try_spend() for _ in range(3)] == [True, True, False]  # minimum 2
    for _ in range(20):
        budget.note_request()
    assert [budget.try_spend() for _ in range(3)] == [True, True, False]  # 20% of 20 = 4


def test_budget_window_ages_out() -> None:
    clock = Clock()
    budget = RatioBudget(0.0, 1, clock=clock)
    assert budget.try_spend()
    assert not budget.try_spend()
    clock.now += 9.5
    assert not budget.try_spend()  # still inside the 10 s window
    clock.now += 1.0
    assert budget.try_spend()


@given(st.lists(st.sampled_from(["request", "spend", "tick"]), max_size=300))
def test_budget_never_spends_past_its_allowance(ops: list[str]) -> None:
    """At every successful spend, the window's spends (this one included) fit within
    max(minimum, ratio * the window's requests)."""
    clock = Clock()
    budget = RatioBudget(0.25, 1, clock=clock)
    events: list[tuple[float, str]] = []
    for op in ops:
        if op == "tick":
            clock.now += 0.7
        elif op == "request":
            budget.note_request()
            events.append((clock.now, "request"))
        elif budget.try_spend():
            events.append((clock.now, "spend"))
            start = int(clock.now) - 9  # the ten one-second slots ending now
            recent = [kind for at, kind in events if int(at) >= start]
            assert recent.count("spend") <= max(1, int(0.25 * recent.count("request")))
