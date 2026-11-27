"""Prometheus metrics for the data plane, served on ``SWITCHBOARD_METRICS_PORT``.

Labels stop at alias/provider/model, never key or request id, so cardinality stays
bounded (PLAN.md §10). An unknown model name from a client never becomes a label.
"""

from prometheus_client import Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
_TTFT_BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30, 60)
# The gateway's own work is sub-millisecond to a few milliseconds.
_OVERHEAD_BUCKETS = (
    0.0001,
    0.00025,
    0.0005,
    0.00075,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.025,
    0.05,
    0.1,
    0.25,
)

REQUESTS = Counter(
    "switchboard_requests",
    "Chat completion requests that reached routing, by final outcome.",
    ["alias", "provider", "model", "status", "outcome"],
)
REQUEST_DURATION = Histogram(
    "switchboard_request_duration_seconds",
    "Time from request received to response complete (whole stream for streaming).",
    ["alias", "provider", "model", "stream"],
    buckets=_LATENCY_BUCKETS,
)
TTFT = Histogram(
    "switchboard_ttft_seconds",
    "Streaming only: time from request received to the first content chunk sent.",
    ["alias", "provider", "model"],
    buckets=_TTFT_BUCKETS,
)
GATEWAY_OVERHEAD = Histogram(
    "switchboard_gateway_overhead_seconds",
    "Time spent in the gateway itself. pre_upstream: request received to upstream "
    "request sent. first_chunk_forward: first content bytes read from upstream to "
    "first chunk written to the client.",
    ["phase"],
    buckets=_OVERHEAD_BUCKETS,
)
INFLIGHT = Gauge(
    "switchboard_inflight_requests",
    "Chat completion requests currently being served (each holds an upstream stream).",
)
CONFIG_INFO = Gauge(
    "switchboard_config_info", "Always 1; the label is the active routing config hash.", ["version"]
)

OVERHEAD_PRE_UPSTREAM = GATEWAY_OVERHEAD.labels(phase="pre_upstream")
OVERHEAD_FIRST_CHUNK = GATEWAY_OVERHEAD.labels(phase="first_chunk_forward")

# --- reliability (Phase 3) -------------------------------------------------------------

ATTEMPTS = Counter(
    "switchboard_attempts",
    "Upstream attempts by outcome: success, an error kind (overloaded, timeout, ...), "
    "circuit_open (skipped without a call), refusal, or cancelled (a hedge that lost, or "
    "the client left).",
    ["provider", "model", "outcome"],
)
RETRIES = Counter(
    "switchboard_retries",
    "Retries of the same target, by the error that caused them.",
    ["provider", "model", "reason"],
)
RETRY_BUDGET_EXHAUSTED = Counter(
    "switchboard_retry_budget_exhausted",
    "Retries not made because the provider's retry budget was spent.",
    ["provider"],
)
FALLBACKS = Counter(
    "switchboard_fallbacks",
    "Moves from a failed candidate to the next one, by the error that caused them.",
    ["alias", "from_model", "to_model", "reason"],
)
HEDGES = Counter(
    "switchboard_hedges",
    "Hedged requests: won (the hedge served), lost (the primary served), skipped (the "
    "route's hedge ratio was reached).",
    ["alias", "outcome"],
)
BREAKER_STATE = Gauge(
    "switchboard_breaker_state",
    "Circuit breaker state as this replica last saw it: 0 closed, 1 half-open, 2 open.",
    ["model"],
)
BREAKER_TRANSITIONS = Counter(
    "switchboard_breaker_transitions",
    "Circuit breaker transitions caused by this replica, by the state entered.",
    ["model", "state"],
)
CONFIG_RELOADS = Counter(
    "switchboard_config_reloads",
    "Routing config reloads by trigger (watch, signal) and result (success, unchanged, error).",
    ["trigger", "result"],
)

# --- keys, limits and accounting (Phase 4) ---------------------------------------------

AUTH_FAILURES = Counter(
    "switchboard_auth_failures",
    "Rejected API keys by reason: malformed, unknown, revoked, expired.",
    ["reason"],
)
RATELIMIT_REJECTIONS = Counter(
    "switchboard_ratelimit_rejections",
    "Requests refused by a rate limit, by scope (key, team) and limit (rpm, tpm).",
    ["scope", "limit"],
)
BUDGET_BLOCKS = Counter(
    "switchboard_budget_blocks",
    "Requests refused because a hard monthly budget would be exceeded, by scope.",
    ["scope"],
)
BUDGET_ALERTS = Counter(
    "switchboard_budget_alerts",
    "Soft budget alerts fired (at most one per scope and month), by scope.",
    ["scope"],
)
BUDGET_UTILIZATION = Gauge(
    "switchboard_budget_utilization_ratio",
    "A team's spend this month over its hard budget, as of its last settled request.",
    ["team"],
)
TOKENS = Counter(
    "switchboard_tokens",
    "Tokens reported by providers, by type: input, cache_read, cache_write, output, "
    "reasoning (reasoning is part of output).",
    ["type"],
)
COST = Counter(
    "switchboard_cost_usd",
    "Spend in USD at the active pricing version, by team, provider and model.",
    ["team", "provider", "model"],
)
USAGE_QUEUE_DEPTH = Gauge(
    "switchboard_usage_queue_depth", "Usage events waiting to be written to Postgres."
)
USAGE_WRITTEN = Counter("switchboard_usage_events_written", "Usage events written to Postgres.")
USAGE_DROPPED = Counter(
    "switchboard_usage_events_dropped", "Usage events dropped because the queue was full."
)
USAGE_WRITE_ERRORS = Counter(
    "switchboard_usage_write_errors", "Failed usage batch writes (each is retried)."
)
DEGRADED = Counter(
    "switchboard_degraded",
    "Requests served with a subsystem degraded: limits (Redis unavailable at admission), "
    "settlement, auth (a stale cached key served while the store is down).",
    ["subsystem"],
)
