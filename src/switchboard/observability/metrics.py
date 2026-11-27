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
