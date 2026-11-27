# 0005: Budgets by reservation, and how far they can overshoot

- **Status:** accepted
- **Date:** 2026-11-27

## Context

A hard budget has to block the request that would cross it. That request's cost is not
known until it ends, and long streams take minutes. Many requests for one key or team
run at once, on several replicas. Two simpler designs fail:

- **Check spend at admission, add the cost at the end.** Every concurrent request
  passes the check against the same stale total, so a burst can overshoot by its whole
  size.
- **Lock per key for the request's duration.** This serializes a key's traffic, which
  breaks streaming workloads.

## Decision

- **Reservations.** Admission reserves an estimate. For a candidate model it is
  `prompt bytes / 4` input tokens plus the output estimate (`max_tokens`, else the
  model default, else 4096) at that model's prices. The plan takes the most expensive
  candidate, since any of them may serve the request. One Lua script checks
  `spent + reserved + estimate ≤ hard limit` and adds to `reserved`, atomically. The
  same script charges the rate-limit buckets, so a request costs one round trip.
- **Settlement.** A second script removes the reservation and adds the actual cost,
  priced from provider-reported usage, to `spent`.
- **Lost settlements.** Reservations are also members of a sorted set scored by expiry
  (the request deadline plus a minute). Admission releases expired members first, so a
  replica that dies mid-request cannot leak budget for long.
- **Representation.** Counters are integer nano-USD (`HINCRBY`), with costs rounded up,
  so they never drift and never under-count. All of a team's keys share the hash tag
  `{t:<team id>}`, so the scripts also run on Redis Cluster.
- **Losing Redis.** A counter that does not exist (a new month, or Redis lost it) is
  rebuilt from the Postgres ledger with `HSETNX` before admission, so concurrent rebuilds
  cannot double-count. Settlement never recreates a missing counter.
- **Scope.** Only scopes with a budget have counters, so keys without one never touch
  Redis for budgets.

## Consequences

- **The bound.** Admission alone never lets `spent + reserved` exceed the limit. Only
  settlement can, and only by `actual − estimate` for that request. So the limit is
  exceeded by at most Σ max(0, actual − estimate) over the requests in flight when it
  is crossed. With estimates that cover the actual cost, it is never exceeded.
  `tests/unit/test_limits.py` checks the bound, and that nothing stays reserved once
  everything settles, with Hypothesis over random interleavings of admissions and
  settlements run on the real scripts.
- **Pessimism.** Estimates are conservative: a key close to its limit is refused a
  request that would have fitted. A large `max_tokens` reserves more. Clients that want
  their last cents should set `max_tokens`.
- **Slow release.** A reservation lost to a crash holds budget until it expires
  (minutes), rather than being released at once.
- **Revisit** if estimates are often far off. A per-model learned output estimate would
  tighten them.
