# 0012: aiohttp, not httpx, for upstream HTTP

- **Status:** accepted
- **Date:** 2026-11-27

## Context

Every chunk a client receives was first read from an upstream connection, so the HTTP
client's per-chunk cost is paid tens of thousands of times a second at a few hundred
concurrent streams. The plan started with httpx and kept the transport isolated in
`providers/http.py` in case it became the bottleneck (PLAN.md §19).

Profiling the gateway under streaming load (cProfile, CPU-time clock) found two
problems with httpx 0.28.1 / httpcore 1.0.9:

- **The pool does O(connections²) work per request.** `_assign_requests_to_connections`
  runs on every request start and end. For each idle connection it builds a list of all
  connections to compare against the keep-alive limit (`connection_pool.py`, lines
  292–296), so the cost grows with the square of the number of open connections, which
  under streaming load is roughly the number of concurrent streams.
- **The read path is deep and pure Python.** Each streamed chunk passes through h11's
  parser, httpcore's state machine and the anyio stream wrapper. aiohttp parses HTTP in
  C (llhttp), and one read often carries several SSE events.

## Measurement

A client-only A/B against the same fake provider, each client in a fresh process,
interleaved so machine drift affects both equally:

```
uv run python -m bench.transport_ab --streams 20,200 --seconds 12 --rounds 2
```

Run on 2026-11-27 on the bench laptop (i7-10510U, WSL2). The machine was heavily loaded
by unrelated work (load average 27 at the start), and WSL2 reports no CPU steal time,
so the absolute CPU figures are inflated. The comparison is what matters.

| Client | Streams | Reads/s | CPU per read |
|---|---:|---:|---:|
| httpx 0.28.1 | 20 | 714, 748 | 567, 546 µs |
| aiohttp 3.14.5 | 20 | 789, 788 | 138, 158 µs |
| httpx 0.28.1 | 200 | 1091, 1190 | 451, 530 µs |
| aiohttp 3.14.5 | 200 | 2433, 2252 | 169, 175 µs |

At 20 streams both clients keep up with the fake provider, and aiohttp uses about a
third of the CPU. At 200 streams httpx sustains about half of aiohttp's read rate.

## Decision

- Upstream HTTP uses one `aiohttp.ClientSession` per provider, behind
  `providers/http.py` (`UpstreamClient`, `UpstreamResponse`). That module also maps
  aiohttp exceptions to `ProviderError` kinds, so adapters import no HTTP library.
- Proxy settings (`HTTP(S)_PROXY`, `NO_PROXY`) are resolved once per provider at
  startup. aiohttp's `trust_env` would re-read the environment on every request.
- httpx stays as a dev dependency: tests use it as a client.

## Consequences

- aiohttp sessions must be created inside a running event loop; the app lifespan does
  that.
- aiohttp keeps idle connections until `keepalive_expiry` and has no cap on their
  number, so `pool.max_keepalive` was removed from the config schema.
- aiohttp rounds timeouts above 5 s up to whole seconds (`ceil_threshold`). Our
  deadlines are 30 s and up by default, and the first-token and total deadlines are
  enforced precisely by the caller.
- Revisit if httpcore removes the quadratic pool scan and the per-chunk call depth
  shrinks; the A/B script makes that a two-minute check.
