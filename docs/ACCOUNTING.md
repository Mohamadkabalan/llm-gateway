# Keys, limits, budgets and accounting

Every request to `/v1` carries a **virtual key** that belongs to a **team**. Keys and
teams can carry rate limits, monthly budgets and an allow-list of models. Every request
that gets past authentication is priced and written to the `usage_events` ledger in
Postgres, whatever its outcome. Routing and failover are described in
[ROUTING.md](ROUTING.md).

## Keys

A key looks like `sb_live_` followed by 43 base62 characters (about 256 random bits),
and is sent as `Authorization: Bearer <key>`, which is what the OpenAI SDKs do with
`api_key`.

- **Storage.** Only `HMAC-SHA256(SWITCHBOARD_KEY_PEPPER, key)` and a display prefix
  (`sb_live_4f2a`) are stored. The secret is shown once, when the key is created. The
  pepper is a server secret, so a copy of the database alone is useless for guessing
  keys. Changing the pepper invalidates every key.
- **Lookup.** A key is looked up in Postgres once, then cached in each replica for
  `SWITCHBOARD_KEY_CACHE_TTL_SECONDS` (30 s). Unknown keys are cached for 5 s, so a
  flood of bad keys does not reach the database. While Postgres is unreachable, an
  already-known key keeps working for up to 5 minutes.
- **Revocation** is pushed to every replica over Redis pub/sub (`sb:chan:invalidate`)
  and takes effect within a second; the integration suite checks this. Without Redis,
  it takes effect within the cache TTL.
- **Refusals.** Unknown, revoked and expired keys get 401 in the OpenAI error shape.
  A key restricted by `allowed_models` sees other models as if they did not exist: 404
  `model_not_found`, and `/v1/models` lists only what it may use.

### Managing keys

```bash
make team TEAM=research ARGS="--budget 100 --soft-budget 80 --rpm 600"
make key TEAM=research NAME=alice ARGS="--budget 10 --tpm 200000 --models smart,fast"
make revoke PREFIX=sb_live_4f2a
```

These run `switchboard create-team`, `create-key` and `revoke-key` inside the gateway
container. `create-key` prints the secret once. Each command also accepts
`--expires-in-days`, and `create-key` accepts `--tags`, which routing rules can match
(see [ROUTING.md](ROUTING.md#rules)).

`make up` creates team `demo`, with a $5 monthly budget and an alert at $4, and a key in
it whose secret is `SWITCHBOARD_DEMO_KEY` in `.env`. It does this with the idempotent
`switchboard ensure-key`. The budget keeps local experiments with real providers cheap.

## Rate limits

Limits are token buckets in Redis, per key and optionally per team:

| Limit | Charged at admission | Refills |
|---|---|---|
| `rpm` | 1 per request | `rpm` per minute, continuously |
| `tpm` | the prompt estimate + the output estimate | `tpm` per minute |

- **Estimates.** The prompt estimate is UTF-8 bytes / 4. The output estimate is the
  request's `max_tokens`, else the model's default, else 4096, capped at the model's
  `max_output_tokens`.
- **Correction.** After the request, the TPM bucket is corrected by actual minus
  estimated tokens. A bucket can go into debt, down to minus one full bucket.
- **All or nothing.** One script checks every bucket the request touches (key and team,
  RPM and TPM) and only then charges all of them, so a request refused by one bucket
  costs nothing from the others.
- **Large requests.** A request bigger than a whole bucket waits for a full bucket
  rather than being refused forever.
- **Clock.** Times are Redis `TIME`: one clock for every replica.

A refused request gets a 429 whose error type is `requests` or `tokens`, with code
`rate_limit_exceeded`, and these headers:

```text
retry-after: 30
x-ratelimit-limit-requests: 2
x-ratelimit-remaining-requests: 0
x-ratelimit-reset-requests: 1m0s
```

Successful responses carry the same `x-ratelimit-*` headers, for the key's own limits
if it has any, else its team's.

## Budgets

Keys and teams can each have a monthly **hard** limit (`--budget`) and a **soft** limit
(`--soft-budget`, in USD). Periods are UTC calendar months.

- **Reservation.** At admission, the request's estimated cost is reserved, priced at
  its most expensive candidate model (any of them may serve it). The script checks
  `spent + reserved + estimate ≤ hard limit`.
- **Settlement.** When the request ends, the reservation is replaced by the actual cost.
- **Counters.** Spend lives in Redis as integer nano-USD and is the real-time source
  of truth. A counter that is missing (a new month, or Redis lost it) is rebuilt from
  the ledger before the request is admitted.
- **Lost settlements.** A reservation that is never settled (a replica died
  mid-request) is released after the request's deadline plus a minute.
- **Overshoot bound.** Concurrent requests can overshoot a hard limit only by what
  they cost beyond their estimates. ADR-0005 explains why; property tests check the
  bound and that no reservation leaks.

A request over a hard limit gets a 429 with type and code `insufficient_quota` and
`x-should-retry: false`. The OpenAI SDKs honour that header, so they do not retry a
blocked request; the integration suite checks this with the Python SDK.

Crossing a soft limit fires **one** alert per scope and month:

- a `budget.soft_limit_reached` warning log line;
- the metric `switchboard_budget_alerts_total{scope}`;
- a row in `budget_alerts`;
- a Slack-compatible JSON POST (`{"text": "…"}`) to `SWITCHBOARD_ALERT_WEBHOOK_URL`, if
  set. The local stack posts to the fake provider's sink: see `webhooks` in
  `curl 127.0.0.1:7800/_control/stats`.

### When Redis is down

Rate limits fail open. Budgets fail open too, unless `SWITCHBOARD_BUDGET_ON_REDIS_FAILURE`
is `closed`; then requests from keys with a budget get 503 `limits_unavailable`. Both
degraded paths are counted in `switchboard_degraded_total{subsystem}` and logged.
Without Redis configured at all, limits and budgets are off, and the gateway says so
at startup.

## Pricing

Prices live in `config/pricing/<date>.yaml`, in USD per million tokens. The routing
config selects the active file:

```yaml
pricing: pricing/2026-11-27.yaml
```

- **Immutability.** A pricing file is never edited after it is committed. A price
  change is a new file, selected in `routing.yaml` and hot-reloaded like any other
  config change.
- **Provenance.** Every usage row records the `pricing_version`, and each file names
  its sources and the date they were read.
- **Coverage.** With a pricing file, every model in `routing.yaml` must have a row;
  `validate-config` reports any that do not.

A row has flat prices, `tiers` by prompt length, or `same_as` another model:

```yaml
anthropic/claude-sonnet-5-5:
  { input: 2, cache_write_5m: 2.50, cache_write_1h: 4, cache_read: 0.10, output: 10 }
anthropic/claude-haiku-5-5:
  tiers:                       # by the request's prompt tokens, cache reads and writes included
    - { max_prompt_tokens: 100000, input: 0.10, cache_write_5m: 0.125, cache_write_1h: 0.20,
        cache_read: 0.01, output: 0.50 }
    - { input: 0.50, cache_write_5m: 0.625, cache_write_1h: 1, cache_read: 0.05, output: 2.50 }
fake/fake-chat: { same_as: anthropic/claude-sonnet-5-5 }
```

**Cost** is computed in `Decimal` from the usage the provider reports:

- each bucket (input, cache reads, 5-minute and 1-hour cache writes, output) is
  multiplied by its price, and the total is divided by a million;
- output includes reasoning tokens, as providers bill them;
- a cache bucket with no price is charged at the input price;
- unit tests check the results against hand-computed values, including the Haiku 5.5
  tier boundary.

If a stream ends without usage (a dropped connection), the cost is estimated from the
prompt size and the output sent so far, and the row is marked `cost_estimated`.

## The usage ledger

`usage_events` has one row per authenticated request, including refusals (rate limited,
budget blocked, unknown model), upstream errors and cancelled requests. A row holds:

- the request id, key, team, requested model, the rule and the model that served it;
- the status, HTTP status and every attempt;
- tokens per bucket, `cost_usd`, `cost_estimated` and `hedge_waste_usd` (the estimated
  input cost of a cancelled hedge);
- the pricing version, TTFT, latency and the gateway's own overhead.

The table is partitioned by month. The gateway creates this month's and next month's
partitions, and rows for a month with no partition go to a default partition rather
than failing.

**Writing.** The request path never waits for Postgres. Each event goes into a bounded
queue (`SWITCHBOARD_USAGE_QUEUE_SIZE`, 10,000), and a background task writes batches of
up to 500 rows every 250 ms with `COPY`.

- **Postgres down:** the batch is retried with backoff while requests keep being
  served. The integration suite pauses the Postgres container, sends requests, and
  checks that their rows arrive after it resumes.
- **Queue full:** an event is dropped and counted in
  `switchboard_usage_events_dropped_total`. Budgets are unaffected, because Redis
  counters are settled separately.
- **Hard crash:** a crash can lose what was still queued. ADR-0006 explains the
  trade-off.

**Migrations** are plain SQL files in `migrations/`, applied once each by
`switchboard migrate` (`make migrate`; `make up` runs it). A file changed after it was
applied is an error. `/ready` reports pending migrations as not ready.

## Metrics

| Metric | Labels |
|---|---|
| `switchboard_cost_usd_total` | `team`, `provider`, `model` |
| `switchboard_tokens_total` | `type`: input, cache_read, cache_write, output, reasoning |
| `switchboard_ratelimit_rejections_total` | `scope` (key, team), `limit` (rpm, tpm) |
| `switchboard_budget_blocks_total`, `switchboard_budget_alerts_total` | `scope` |
| `switchboard_budget_utilization_ratio` (gauge) | `team`: spend over the hard budget |
| `switchboard_auth_failures_total` | `reason`: malformed, unknown, revoked, expired |
| `switchboard_usage_queue_depth` (gauge), `switchboard_usage_events_written_total`, `…_dropped_total`, `switchboard_usage_write_errors_total` | |
| `switchboard_degraded_total` | `subsystem`: limits, settlement, auth |

Labels stop at team, never key. Per-key detail lives in the ledger.

## Known gaps

- **Unpriced Anthropic modifiers.** Anthropic's `inference_geo: "us"` (a 1.1x
  multiplier), fast mode, batch prices and server-tool fees such as web search are not
  modelled. A request that uses them through `provider_options` is priced at the
  standard rate.
- **Unpriced OpenAI models.** Models whose cache-write columns use other durations
  (30-minute and 12-hour writes) are not in the 2026-11-27 file, because the usage model
  has no buckets for those writes. The routed OpenAI models list no cache-write prices,
  so they are unaffected.
- **Rebuilt counters.** A counter rebuilt from the ledger misses rows that were still
  queued when Redis lost it.
- **Key metadata.** There is no `last_used_at`, and keys are rotated by creating a new
  one and revoking the old. The admin UI comes in Phase 6.
