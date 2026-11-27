# 0001: Python (not Go) for the gateway, with a measured decision gate

- **Status:** accepted
- **Date:** 2026-11-27

## Context

Switchboard sits on the request path of every LLM call, so its own overhead matters.
Reviewers will read the code and the benchmark numbers.

Go is the conventional choice for proxies. Its per-chunk CPU cost and per-connection
memory are lower, goroutines make hundreds of concurrent streams trivial, and a static
binary is simple to ship. Several newer gateways are written in Go or Rust for these
reasons, and a Go implementation would very likely post better absolute overhead numbers.

The workload, though, is I/O-bound. Upstream time to first token is hundreds of
milliseconds to seconds, and the gateway's job is to add single-digit milliseconds. Most
of the code is JSON reshaping between provider formats, accounting and policy, where
iteration speed and testability matter more than raw speed. Python also has the
ecosystem: provider wire formats are documented with Python examples, and OpenTelemetry
GenAI conventions, tokenizers and pgvector clients all exist. AI-infrastructure readers
read it fluently. All shared state lives in Redis and Postgres, so the gateway scales
horizontally by adding replicas.

## Decision

Write the gateway in Python 3.14 with FastAPI, httpx, Pydantic v2, uvloop and orjson.
Keep the hot path disciplined:

- no Pydantic validation per stream chunk
- orjson for serialization
- pooled upstream connections
- one Redis round trip for admission
- plain ASGI middleware (no `BaseHTTPMiddleware`)

## Consequences

- **Decision gate (Phase 1):** before any features exist, measure added time to first
  token for one gateway process against the fake provider. The target is p99 ≤ 25 ms at
  200 concurrent streams on the bench machine. That target is my choice, not a
  measurement. If we miss it badly, we stop and choose between optimizing the hot path
  and switching languages, while the codebase is still small.
- One process uses about one core, so throughput comes from replicas. The benchmark
  reports single-process numbers and states this plainly.
- The upstream HTTP transport is isolated (`providers/http.py`), so if measurements
  show httpx's connection pool to be the bottleneck it can be swapped without touching
  the adapters.
