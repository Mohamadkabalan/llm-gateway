# 0004: Shared state in Redis, and what happens when Redis is down

- **Status:** accepted (circuit breakers, Phase 3; rate limits and budgets follow the same
  rules in Phase 4)
- **Date:** 2026-11-27

## Context

The gateway runs as interchangeable replicas (PLAN.md §0). State that decides how one
request is treated must be shared, or each replica behaves differently. With per-process
breakers, N replicas would each send `min_requests` failures before stopping, and
together let N × `half_open_probes` probes reach a provider that is recovering. Phase 4's
rate limits and budgets are useless unless every replica counts the same spend.

The hot path must not pay for this. A circuit breaker is consulted for every candidate
of every request. A Redis round trip per candidate would add latency to every request,
when the breaker is closed almost all the time.

## Decision

- **Store.** Breaker state lives in one Redis hash per `provider/model`
  (`sb:cb:{provider/model}`; the braces make it a Redis Cluster hash tag). Two Lua
  scripts own every transition:
  - `admit` turns an expired open breaker half-open and hands out probe slots with
    `HINCRBY`, so the probe limit is exact across replicas.
  - `record` counts an outcome in a ring of 10 buckets across the window and opens or
    closes the breaker.

  Only core commands and Lua are used, so Valkey works too. Times come from the gateway
  (wall clock), which tolerates the few milliseconds of skew between replicas.
- **Reads are local.** Each replica keeps the last state it saw for each breaker,
  updated from every script reply. A closed, or still-open, breaker is decided with no
  round trip. Redis is asked only when an open breaker's cooldown has passed, and while
  it is half-open.
- **Writes are fire-and-forget.** Outcomes are recorded in a background task and never
  delay or fail a response.
- **Redis down.** Any Redis error moves that replica to an in-process breaker with the
  same state machine, for 5 s, after which it tries Redis again. The switch is logged once
  per outage. A breaker is a safety device, so failing *closed* (refusing traffic because
  Redis is down) would turn a Redis outage into a full outage.
- **One implementation, two stores, one test table.** `tests/breaker_cases.py` runs
  against the in-process store in unit tests and against the Lua scripts in the
  integration tests (real Redis).

## Consequences

- With traffic flowing and the breaker closed, a request costs no Redis round trip for
  breaking, and one background `EVALSHA` per attempt.
- **Lag.** A replica learns that another one opened a breaker from the reply to its own
  next recorded outcome. Until then it keeps sending requests: roughly its in-flight
  requests to that target, each of which then fails over. This is accepted, and covered
  by `test_replicas_share_one_breaker`.
- **Redis outage.** During one, breakers are per replica again, with the weaker
  guarantees described above, until Redis is back.
- **Expiry.** Breaker keys expire: an idle closed breaker after one window plus a
  bucket, an open one after its cooldown plus that. Redis therefore holds only breakers
  in use, and a stale open state cannot outlive its usefulness.
- **Revisit** if the background writes show up in the gateway's CPU profile at the
  Phase 8 benchmark rates. Batching outcomes per replica (one script call per 100 ms)
  would be the next step.
