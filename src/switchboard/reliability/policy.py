"""What each kind of upstream failure allows (PLAN.md §2.2).

Three independent questions per failure: may the same target be tried again, may the
next candidate be tried, and does the failure count against the target's circuit
breaker. The table below is the single source of truth; every row has a unit test.
"""

from dataclasses import dataclass
from enum import StrEnum

from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.routing.table import ModelTarget


class Fallback(StrEnum):
    NO = "no"
    YES = "yes"
    LARGER_CONTEXT = "larger_context"  # only to candidates with a larger max_input_tokens


@dataclass(frozen=True, slots=True)
class Disposition:
    retry: bool
    fallback: Fallback
    breaker: bool  # counts as the provider's failure


DISPOSITIONS: dict[ErrorKind, Disposition] = {
    ErrorKind.CONNECT: Disposition(retry=True, fallback=Fallback.YES, breaker=True),
    # Retried at most once per target (see TIMEOUT_RETRIES), then the next candidate.
    ErrorKind.TIMEOUT: Disposition(retry=True, fallback=Fallback.YES, breaker=True),
    # Retried only if retry-after is within honor_retry_after_up_to (see backoff).
    ErrorKind.RATE_LIMITED: Disposition(retry=True, fallback=Fallback.YES, breaker=True),
    ErrorKind.QUOTA: Disposition(retry=False, fallback=Fallback.YES, breaker=True),
    ErrorKind.OVERLOADED: Disposition(retry=True, fallback=Fallback.YES, breaker=True),
    ErrorKind.PROTOCOL: Disposition(retry=True, fallback=Fallback.YES, breaker=True),
    ErrorKind.AUTH: Disposition(retry=False, fallback=Fallback.YES, breaker=True),
    ErrorKind.CONTEXT_LENGTH: Disposition(
        retry=False, fallback=Fallback.LARGER_CONTEXT, breaker=False
    ),
    ErrorKind.BAD_REQUEST: Disposition(retry=False, fallback=Fallback.NO, breaker=False),
    # The adapter could not express the request for this provider and sent nothing;
    # another provider may be able to.
    ErrorKind.INVALID_REQUEST: Disposition(retry=False, fallback=Fallback.YES, breaker=False),
    # Not an attempt: the target's breaker is open, so it was skipped.
    ErrorKind.CIRCUIT_OPEN: Disposition(retry=False, fallback=Fallback.YES, breaker=False),
    # A refusal before the commit point, on a route with fallback_on_refusal.
    ErrorKind.REFUSAL: Disposition(retry=False, fallback=Fallback.YES, breaker=False),
}

TIMEOUT_RETRIES = 1  # a timed-out target is retried at most this many times


def disposition(error: ProviderError) -> Disposition:
    return DISPOSITIONS[error.kind]


def may_retry(error: ProviderError) -> bool:
    """The table's answer, unless the provider said otherwise with ``x-should-retry``."""
    if error.should_retry is not None:
        return error.should_retry
    return disposition(error).retry


def may_fall_back(error: ProviderError, failed: ModelTarget, candidate: ModelTarget) -> bool:
    rule = disposition(error).fallback
    if rule is Fallback.YES:
        return True
    if rule is Fallback.LARGER_CONTEXT:
        have = failed.capabilities.max_input_tokens
        offered = candidate.capabilities.max_input_tokens
        return have is not None and offered is not None and offered > have
    return False


def stops_the_walk(error: ProviderError) -> bool:
    """True if no later candidate may be tried at all (the request itself is wrong)."""
    return disposition(error).fallback is Fallback.NO
