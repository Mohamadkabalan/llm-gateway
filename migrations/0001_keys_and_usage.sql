-- Phase 4: teams, virtual keys, the usage ledger and budget alerts (PLAN.md §9.1).
-- Applied by `switchboard migrate`; never edit a migration after it has been applied.

CREATE TABLE teams (
  id                 uuid PRIMARY KEY,
  name               text NOT NULL UNIQUE,
  monthly_budget_usd numeric(12,4) CHECK (monthly_budget_usd >= 0),   -- hard limit; NULL = none
  soft_budget_usd    numeric(12,4) CHECK (soft_budget_usd >= 0),      -- alert threshold
  rpm_limit          integer CHECK (rpm_limit > 0),
  tpm_limit          integer CHECK (tpm_limit > 0),
  allowed_models     text[],                                          -- NULL = every model
  created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE api_keys (
  id                 uuid PRIMARY KEY,
  team_id            uuid NOT NULL REFERENCES teams (id),
  name               text NOT NULL,
  key_prefix         text NOT NULL,                                   -- shown in logs and the UI
  key_hash           bytea NOT NULL UNIQUE,                           -- HMAC-SHA256(pepper, secret)
  status             text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
  monthly_budget_usd numeric(12,4) CHECK (monthly_budget_usd >= 0),
  soft_budget_usd    numeric(12,4) CHECK (soft_budget_usd >= 0),
  rpm_limit          integer CHECK (rpm_limit > 0),
  tpm_limit          integer CHECK (tpm_limit > 0),
  allowed_models     text[],
  tags               text[] NOT NULL DEFAULT '{}',
  expires_at         timestamptz,
  revoked_at         timestamptz,
  created_at         timestamptz NOT NULL DEFAULT now(),
  UNIQUE (team_id, name)
);

-- One row per request that passed authentication, whatever its outcome. Partitioned by
-- month; the gateway creates partitions ahead with switchboard_ensure_usage_partition().
CREATE TABLE usage_events (
  id                  uuid NOT NULL,                                  -- UUIDv7
  created_at          timestamptz NOT NULL,
  request_id          text NOT NULL,
  trace_id            text,
  key_id              uuid NOT NULL,
  team_id             uuid NOT NULL,
  endpoint            text NOT NULL,                                  -- chat | embeddings
  requested_model     text NOT NULL,
  route_rule          text,
  provider            text,                                           -- what served it; NULL if nothing did
  model               text,
  status              text NOT NULL,     -- ok | upstream_error | rate_limited | budget_blocked
                                         -- | client_cancelled | guardrail_blocked | invalid_request
  http_status         smallint NOT NULL,
  error_type          text,
  stream              boolean NOT NULL,
  cache_status        text NOT NULL DEFAULT 'disabled',               -- Phase 5
  fallback_used       boolean NOT NULL DEFAULT false,
  attempts            jsonb NOT NULL DEFAULT '[]',
  input_tokens        integer,
  cache_read_tokens   integer,
  cache_write_tokens  integer,
  output_tokens       integer,
  reasoning_tokens    integer,
  cost_usd            numeric(18,9) NOT NULL DEFAULT 0,
  cost_estimated      boolean NOT NULL DEFAULT false,
  saved_usd           numeric(18,9) NOT NULL DEFAULT 0,               -- Phase 5
  hedge_waste_usd     numeric(18,9) NOT NULL DEFAULT 0,
  pricing_version     text,
  ttft_ms             integer,
  latency_ms          integer NOT NULL,
  gateway_overhead_ms real,
  prompt_name         text,                                           -- Phase 7
  prompt_version      integer,
  tags                text[] NOT NULL DEFAULT '{}',
  end_user            text,
  PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE INDEX usage_events_team_time ON usage_events (team_id, created_at);
CREATE INDEX usage_events_key_time ON usage_events (key_id, created_at);
CREATE INDEX usage_events_request ON usage_events (request_id);

-- Rows whose month has no partition land here instead of failing the batch.
CREATE TABLE usage_events_default PARTITION OF usage_events DEFAULT;

CREATE FUNCTION switchboard_ensure_usage_partition(month date) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
  start_at date := date_trunc('month', month)::date;
  name text := format('usage_events_y%sm%s', to_char(start_at, 'YYYY'), to_char(start_at, 'MM'));
BEGIN
  IF to_regclass(name) IS NULL THEN
    EXECUTE format(
      'CREATE TABLE %I PARTITION OF usage_events FOR VALUES FROM (%L) TO (%L)',
      name, start_at, (start_at + interval '1 month')::date
    );
  END IF;
END;
$$;

SELECT switchboard_ensure_usage_partition(now()::date);
SELECT switchboard_ensure_usage_partition((now() + interval '1 month')::date);

CREATE TABLE budget_alerts (
  scope     text NOT NULL CHECK (scope IN ('key', 'team')),
  scope_id  uuid NOT NULL,
  period    date NOT NULL,                                            -- first day of the month
  threshold text NOT NULL,                                            -- soft
  spent_usd numeric(18,9) NOT NULL,
  limit_usd numeric(12,4) NOT NULL,
  fired_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (scope, scope_id, period, threshold)
);
