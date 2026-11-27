# 0006: An in-process usage queue, not a durable stream

- **Status:** accepted
- **Date:** 2026-11-27

## Context

Every authenticated request produces a `usage_events` row. That row is the ledger
behind reports and budget rebuilds. Writing it inside the request would put Postgres
latency, and Postgres outages, on the hot path. Two alternatives:

1. **A bounded in-process queue,** drained by a background task that writes batches
   with `COPY`.
2. **A durable stream** (Redis Streams, Kafka) in front of a separate consumer.

## Decision

Option 1:

- **Never blocks.** The request path only calls `put_nowait` on an `asyncio.Queue`
  (10,000 by default). A full queue drops the event and counts it in
  `switchboard_usage_events_dropped_total`, rather than slowing requests.
- **Batches.** The writer takes up to 500 rows, or whatever arrived within 250 ms,
  and writes them with `copy_records_to_table`.
- **Postgres down.** The batch is retried with backoff (up to 30 s between tries)
  while new events wait in the queue. The integration suite pauses the Postgres
  container mid-test and checks that the rows land afterwards.
- **Shutdown.** The writer gets 5 s to flush what is queued.
- **Budgets stay separate.** Budgets never depend on the ledger being current, because
  Redis counters are settled independently. The ledger is used only to rebuild a
  counter Redis has lost.

## Consequences

- **Loss window.** A hard crash (SIGKILL, OOM) loses what was queued: at steady load
  about one flush interval of rows, more if Postgres was already down. Budgets are not
  affected, but the ledger, and costs rebuilt from it, would miss those requests.
- **Simplicity.** No extra service to run, and no consumer to keep healthy or
  monitor.
- **The durable alternative.** Redis Streams with a consumer group would close the
  crash window, at the cost of one more Redis write per request and a consumer process.
  It slots in behind the same `UsageSink` interface. Switch if a deployment needs an
  exact ledger more than it needs the simpler moving parts, for example when billing
  customers from it.
