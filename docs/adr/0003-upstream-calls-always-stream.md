# 0003: Upstream calls always stream, and the response commits on the first token

- **Status:** accepted
- **Date:** 2026-11-27

## Context

Clients ask for streaming or non-streaming responses. Supporting both towards every
provider doubles the upstream code paths, and each path needs its own timeouts, metrics
and failure handling. Fallback is also only safe before the client has seen any output
(PLAN.md §2.1).

## Decision

- Every upstream call streams. For a non-streaming client, the gateway aggregates the
  events into one `chat.completion` (`schemas.encode.Aggregator`).
- The response is served by `api.responder.ChatResponder`, which runs the upstream call
  itself. It buffers events until the first content event (text, refusal, tool call or
  finish), the **commit point**, and only then sends the status line.
  - Before the commit point, any failure is returned as a normal HTTP error in the
    OpenAI shape. This is where Phase 3 adds retries and fallback.
  - After it, a failure ends the stream with an OpenAI-shaped SSE error event, which
    the OpenAI SDK raises as `APIError`.
- A watcher task listens for `http.disconnect` for the whole request, including the wait
  for the first token, and cancels the upstream call when the client leaves. This is
  needed because uvicorn silently discards writes to a closed connection; without the
  watcher, a stream would keep running, and billing, for nobody.
- Deadlines: the first-token and total timeouts are enforced by the responder with
  `asyncio.timeout` (never across a `yield`); the transport's per-read timeout is the
  longer of `first_token` and `idle`.

## Consequences

- One upstream code path gives one set of timeout semantics, TTFT metrics for
  every request, and one place for pre-commit fallback.
- Non-streaming clients get no benefit from streaming, and the aggregation costs a
  little CPU per chunk.
- Known gap, closed in Phase 3: if `first_token` is longer than `idle`, the idle timeout
  is only enforced at `first_token` granularity, because the transport's read timeout
  cannot change mid-response. The default config sets both to 30 s.
- Keep-alive pings for slow first tokens (`keepalive_after`) are also Phase 3, with the
  executor that owns the pre-commit wait.
