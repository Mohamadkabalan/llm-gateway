"""One request's accounting: admission, cost, settlement, metrics and its usage row.

Endpoints create a ``RequestLedger`` once the key is known, call ``admit`` before any
upstream work, and ``reject`` or ``finish`` exactly once at the end. ``finish``:

* prices provider-reported usage at the active pricing version, or estimates the cost
  (``cost_estimated``) when a stream ended without usage;
* settles the budget reservation and corrects the TPM buckets (in the background);
* adds tokens and cost to the metrics;
* queues the ``usage_events`` row.
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import orjson

from switchboard.accounting.pricing import Cost, PriceTable, estimate_tokens_cost
from switchboard.accounting.usage import UsageEvent, UsageSink
from switchboard.auth.context import KeyContext
from switchboard.limits.limiter import Admitted, Limiter
from switchboard.observability.metrics import COST, TOKENS
from switchboard.routing.table import ModelTarget, RoutePlan
from switchboard.schemas.events import Usage

DEFAULT_OUTPUT_ESTIMATE = 4096  # when neither the request nor the model sets max_tokens

# How a request ended, as usage_events.status.
OUTCOME_STATUS = {
    "success": "ok",
    "upstream_error": "upstream_error",
    "client_error": "invalid_request",
    "client_disconnect": "client_cancelled",
    "cancelled": "client_cancelled",
}


def output_estimate(requested: int | None, target: ModelTarget) -> int:
    """The output tokens to reserve for: the request's max_tokens, else the model's
    default, else ``DEFAULT_OUTPUT_ESTIMATE``; never past the model's maximum."""
    estimate = requested or target.defaults.max_tokens or DEFAULT_OUTPUT_ESTIMATE
    ceiling = target.capabilities.max_output_tokens
    return min(estimate, ceiling) if ceiling else estimate


@dataclass(slots=True)
class Served:
    """What ``finish`` needs to know about how the request went."""

    target: ModelTarget | None
    outcome: str
    http_status: int
    usage: Usage | None = None
    error_kind: str | None = None
    fallback: bool = False
    attempts: list[dict[str, object]] = field(default_factory=list)
    hedge_losers: list[ModelTarget] = field(default_factory=list)
    output_chars: int = 0
    ttft_ms: float | None = None
    overhead_ms: float | None = None


class RequestLedger:
    def __init__(
        self,
        *,
        key: KeyContext,
        sink: UsageSink,
        limiter: Limiter | None,
        prices: PriceTable | None,
        request_id: str,
        endpoint: str,
        requested_model: str,
        stream: bool,
        tags: frozenset[str],
        end_user: str | None,
        received_at: float,
    ) -> None:
        self.key = key
        self._sink = sink
        self._limiter = limiter
        self._prices = prices
        self._request_id = request_id
        self._endpoint = endpoint
        self._requested = requested_model
        self._stream = stream
        self._tags = sorted(tags)
        self._end_user = end_user
        self._received_at = received_at
        self._created_at = datetime.now(UTC)
        self.prompt_tokens = 0
        self.admitted: Admitted | None = None
        self.plan: RoutePlan | None = None
        self._done = False

    # --- before the upstream call ------------------------------------------------------

    def describe(
        self,
        *,
        requested_model: str,
        stream: bool = False,
        end_user: str | None = None,
        tags: frozenset[str] | None = None,
    ) -> None:
        """Fill in what the request asked for, as it becomes known."""
        self._requested = requested_model
        self._stream = stream
        self._end_user = end_user
        if tags is not None:
            self._tags = sorted(self.key.tags | tags)

    async def admit(
        self, plan: RoutePlan, prompt_tokens: int, output_tokens: int, *, hold_s: float
    ) -> None:
        """Charge rate limits and reserve budget; raises the limiter's errors. A
        reservation not settled within ``hold_s`` (a crashed replica) is released."""
        self.plan = plan
        self.prompt_tokens = prompt_tokens
        if self._limiter is None:
            return
        estimate = Cost(Decimal(0), estimated=True)
        if self._prices is not None:
            models = [t.name for t in plan.candidates]
            estimate = estimate_tokens_cost(self._prices, models, prompt_tokens, output_tokens)
        self.admitted = await self._limiter.admit(
            self.key,
            request_id=self._request_id,
            tokens=prompt_tokens + output_tokens,
            cost=estimate,
            hold_s=hold_s,
        )

    def rate_headers(self) -> dict[str, str]:
        return self.admitted.rate.headers() if self.admitted is not None else {}

    # --- at the end --------------------------------------------------------------------

    def reject(self, *, status: str, http_status: int, error_type: str | None) -> None:
        """The request was refused before any upstream call."""
        self._write(
            Served(target=None, outcome=status, http_status=http_status, error_kind=error_type),
            Cost(Decimal(0)),
            status=status,
        )

    def finish(self, served: Served) -> Cost:
        cost = self._cost(served)
        tokens = None
        if served.usage is not None:
            tokens = served.usage.prompt_total + served.usage.output
        if self._limiter is not None:
            self._limiter.settle(self.admitted, self.key, tokens=tokens, cost=cost)
        if served.usage is not None:
            _count_tokens(served.usage)
        if served.target is not None and cost.usd:
            COST.labels(self.key.team_name, served.target.provider, served.target.name).inc(
                float(cost.usd)
            )
        self._write(served, cost, status=OUTCOME_STATUS.get(served.outcome, served.outcome))
        return cost

    def _cost(self, served: Served) -> Cost:
        if self._prices is None or served.target is None:
            return Cost(Decimal(0))
        if served.usage is not None:
            value = self._prices.cost(served.target.name, served.usage)
            return Cost(value or Decimal(0))
        if served.outcome == "success" or served.output_chars:
            # Served but no usage arrived (an interrupted stream): estimate from sizes.
            output = math.ceil(served.output_chars / 4)
            value = self._prices.estimate(served.target.name, self.prompt_tokens, output)
            return Cost(value or Decimal(0), estimated=True)
        return Cost(Decimal(0))

    def _hedge_waste(self, losers: list[ModelTarget]) -> Decimal:
        """Providers may bill the input of a cancelled request: count it, estimated."""
        if self._prices is None:
            return Decimal(0)
        total = Decimal(0)
        for target in losers:
            total += self._prices.estimate(target.name, self.prompt_tokens, 0) or Decimal(0)
        return total

    def _write(self, served: Served, cost: Cost, *, status: str) -> None:
        if self._done:
            return
        self._done = True
        usage = served.usage
        plan = self.plan
        self._sink.put(
            UsageEvent(
                id=uuid.uuid7(),
                created_at=self._created_at,
                request_id=self._request_id,
                trace_id=None,
                key_id=self.key.key_id,
                team_id=self.key.team_id,
                endpoint=self._endpoint,
                requested_model=self._requested,
                route_rule=plan.rule if plan is not None else None,
                provider=served.target.provider if served.target else None,
                model=served.target.name if served.target else None,
                status=status,
                http_status=served.http_status,
                error_type=served.error_kind,
                stream=self._stream,
                fallback_used=served.fallback,
                attempts=orjson.dumps(served.attempts).decode(),
                input_tokens=usage.input if usage else None,
                cache_read_tokens=usage.cache_read if usage else None,
                cache_write_tokens=(usage.cache_write_5m + usage.cache_write_1h) if usage else None,
                output_tokens=usage.output if usage else None,
                reasoning_tokens=usage.reasoning if usage else None,
                cost_usd=cost.usd,
                cost_estimated=cost.estimated,
                hedge_waste_usd=self._hedge_waste(served.hedge_losers),
                pricing_version=self._prices.version if self._prices is not None else None,
                ttft_ms=round(served.ttft_ms) if served.ttft_ms is not None else None,
                latency_ms=round((time.perf_counter() - self._received_at) * 1000),
                gateway_overhead_ms=served.overhead_ms,
                tags=self._tags,
                end_user=self._end_user,
            )
        )


def _count_tokens(usage: Usage) -> None:
    for kind, value in (
        ("input", usage.input),
        ("cache_read", usage.cache_read),
        ("cache_write", usage.cache_write_5m + usage.cache_write_1h),
        ("output", usage.output),
        ("reasoning", usage.reasoning),
    ):
        if value:
            TOKENS.labels(kind).inc(value)
