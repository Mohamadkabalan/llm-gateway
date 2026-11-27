# 0011: k6 with xk6-sse for load testing, pinned and built in Docker

- **Status:** accepted
- **Date:** 2026-11-27

## Context

The headline benchmark is time to first token added by the gateway, measured on SSE
streams at hundreds of concurrent connections. k6's own HTTP client buffers whole
responses, so it cannot see the first event. A Python load generator would compete for
the same CPU and add its own event-loop jitter to the numbers.

## Decision

- Use k6 with the `xk6-sse` extension (`github.com/phymbert/xk6-sse`, the upstream
  repository, tag v0.2.0, whose module checksum Go verifies). Build the binary with xk6
  in `bench/k6/Dockerfile`, so every run uses the same load generator. k6 v2 can fetch
  extensions on demand, but then the binary would depend on when the run happened.
- Pin **k6 v2.0.0**, the version xk6-sse v0.2.0 is tested against. Built against k6
  v2.4.0, the SSE client object reached scripts without its methods (`client.on` was
  undefined); checked on 2026-11-27.
- One script (`bench/k6/chat_stream.js`) measures both paths. Only the target URL,
  model and key differ between the direct baseline and the gateway run, so client
  behaviour is identical.

## Consequences

- k6's JavaScript runtime has no `performance.now()` (checked in k6 2.4.0), so client
  timings are whole milliseconds. That is fine for a 25 ms gate; the gateway's own
  histograms provide the sub-millisecond view.
- xk6-sse opens a fresh HTTP transport per call, so every stream uses a new TCP
  connection to its target. The direct baseline pays the same cost, so it cancels out
  of the comparison, but clients that keep connections alive would see slightly less.
- The script parses JSON only until the first content event and uses string checks
  after that; otherwise k6's runtime, not the gateway, would limit the 500-stream runs.
- Revisit when xk6-sse tags a release that works with current k6, or if k6 gains native
  SSE support.
