"""Retry delays: full-jitter exponential backoff that honours ``retry-after``."""

import random

from switchboard.providers.errors import ProviderError
from switchboard.routing.config import RetryPolicy


def full_jitter(retry_index: int, policy: RetryPolicy, rng: random.Random) -> float:
    """``random(0, min(max_delay, base_delay * 2**retry_index))``; index 0 is the first retry."""
    cap = min(policy.max_delay, policy.base_delay * 2**retry_index)
    return rng.uniform(0, cap)


def retry_delay(
    error: ProviderError, retry_index: int, policy: RetryPolicy, rng: random.Random
) -> float | None:
    """Seconds to wait before retrying, or None if waiting as asked is not worth it.

    A ``retry-after`` longer than ``honor_retry_after_up_to`` means the next candidate
    is a better bet than waiting, so the target is not retried.
    """
    delay = full_jitter(retry_index, policy, rng)
    if error.retry_after is not None:
        if error.retry_after > policy.honor_retry_after_up_to:
            return None
        delay = max(delay, error.retry_after)
    return delay
