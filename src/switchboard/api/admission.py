"""The part of every data-plane request between authentication and the upstream call.

``open_ledger`` starts a request's accounting once its key is known; ``recorded`` turns
any refusal after that point (bad request, unknown or forbidden model, rate limit,
budget) into an OpenAI-shaped error and a ``usage_events`` row, so every authenticated
request is in the ledger.
"""

import contextlib
import math
from collections.abc import Iterator

from fastapi import Request
from fastapi.exceptions import RequestValidationError

from switchboard.accounting.ledger import RequestLedger
from switchboard.api.errors import GatewayError
from switchboard.auth.context import KeyContext
from switchboard.limits.limiter import (
    BudgetExceededError,
    LimiterUnavailableError,
    RateLimitedError,
)
from switchboard.routing.table import RouteError
from switchboard.state import AppState


def open_ledger(
    state: AppState, request: Request, key: KeyContext, *, endpoint: str
) -> RequestLedger:
    scope_state = request.scope["state"]
    return RequestLedger(
        key=key,
        sink=state.usage,
        limiter=state.limiter,
        prices=state.config.current.loaded.pricing,
        request_id=scope_state["request_id"],
        endpoint=endpoint,
        requested_model="",
        stream=False,
        tags=key.tags,
        end_user=None,
        received_at=scope_state["received_at"],
    )


def check_model_access(key: KeyContext, model: str) -> None:
    """A model the key may not use looks exactly like one that does not exist."""
    if not key.may_use(model):
        raise RouteError(
            404,
            f"The model '{model}' does not exist or you do not have access to it.",
            code="model_not_found",
            error_type="invalid_request_error",
        )


@contextlib.contextmanager
def recorded(ledger: RequestLedger) -> Iterator[None]:
    """Record a refusal in the ledger, and turn limiter errors into HTTP errors."""
    try:
        yield
    except RateLimitedError as exc:
        ledger.reject(status="rate_limited", http_status=429, error_type="rate_limit_exceeded")
        raise rate_limited(exc) from None
    except BudgetExceededError as exc:
        ledger.reject(status="budget_blocked", http_status=429, error_type="insufficient_quota")
        raise budget_exceeded(exc) from None
    except LimiterUnavailableError:
        ledger.reject(status="unavailable", http_status=503, error_type="limits_unavailable")
        raise GatewayError(
            503,
            "Budgets cannot be checked right now; try again shortly.",
            error_type="server_error",
            code="limits_unavailable",
        ) from None
    except RouteError as exc:
        status = "unavailable" if exc.status >= 500 else "invalid_request"
        ledger.reject(status=status, http_status=exc.status, error_type=exc.code)
        raise
    except GatewayError as exc:
        status = "unavailable" if exc.status >= 500 else "invalid_request"
        ledger.reject(status=status, http_status=exc.status, error_type=exc.code)
        raise
    except RequestValidationError:
        ledger.reject(status="invalid_request", http_status=400, error_type="invalid_request")
        raise


def rate_limited(exc: RateLimitedError) -> GatewayError:
    bucket = exc.bucket
    unit = "requests" if bucket.kind == "rpm" else "tokens"
    wait = max(1, math.ceil(exc.retry_after))
    who = "this API key" if bucket.scope == "key" else "this key's team"
    headers = {"retry-after": str(wait), **exc.rate.headers()}
    return GatewayError(
        429,
        f"Rate limit reached for {who} on {unit} per minute ({bucket.kind.upper()}): limit "
        f"{bucket.capacity}. Please try again in {wait}s.",
        error_type=unit,
        code="rate_limit_exceeded",
        headers=headers,
    )


def budget_exceeded(exc: BudgetExceededError) -> GatewayError:
    budget = exc.budget
    who = "This API key" if budget.scope == "key" else "This key's team"
    limit = budget.limits.hard_usd
    return GatewayError(
        429,
        f"{who} has reached its monthly budget of ${limit:.2f}. It resets at the start of "
        "next month (UTC), or an admin can raise it.",
        error_type="insufficient_quota",
        code="insufficient_quota",
        # The OpenAI SDKs honour this and do not retry: waiting will not help.
        headers={"x-should-retry": "false"},
    )
