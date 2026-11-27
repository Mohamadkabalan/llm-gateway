# Switchboard

A self-hostable LLM gateway. Applications call one OpenAI-compatible endpoint with a
virtual key. Switchboard does the rest:

- routes each request to the right provider and model
- retries, and falls back to another model when one fails
- caches responses
- enforces per-key and per-team budgets and rate limits
- records cost, tokens and latency for every call

> **Status: Phase 4 (keys, budgets, rate limits, accounting).**
>
> - **API.** `/v1/chat/completions`, `/v1/embeddings` and `/v1/models` work end to end
>   with OpenAI, Anthropic (native Messages API) and Ollama.
> - **Routing.** Requests are routed by aliases, rules and weighted A/B splits. When a
>   provider fails before the first token, the gateway retries and falls back without
>   the client seeing it. Circuit breakers are shared through Redis.
> - **Keys and money.** Clients authenticate with hashed virtual keys. Keys and teams
>   carry rate limits and monthly budgets. Every request is priced from a versioned
>   pricing file and written to a ledger in Postgres.
> - **Docs.** [docs/ROUTING.md](docs/ROUTING.md) covers routing and reliability,
>   [docs/ACCOUNTING.md](docs/ACCOUNTING.md) keys, limits, budgets and pricing, and
>   [docs/PROVIDERS.md](docs/PROVIDERS.md) what is translated, dropped or refused.
> - **No keys needed.** The local stack includes a fake provider that speaks both the
>   OpenAI and Anthropic APIs.
>
> The plan and phase-by-phase acceptance criteria are in [PLAN.md](PLAN.md), and design
> decisions in [docs/adr](docs/adr). The full README is written in Phase 8, once there
> are measured results to show.

## Try it

Requirements: [uv](https://docs.astral.sh/uv/), Docker with Compose v2, make.

```bash
make up     # builds and starts the stack, and creates a demo API key (secret in .env)
export OPENAI_BASE_URL=http://127.0.0.1:7080/v1
export OPENAI_API_KEY="$(grep '^SWITCHBOARD_DEMO_KEY=' .env | cut -d= -f2-)"
```

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY
stream = client.chat.completions.create(
    model="fake",  # an alias in config/routing.yaml; "fake-claude" goes through the
    # Anthropic translation, and real models work once their keys are in .env
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Every response carries `x-request-id`, `x-switchboard-provider`,
`x-switchboard-model` and `x-switchboard-attempts`. Errors use the OpenAI error shape,
so the SDK raises its usual exceptions. To watch failover and a circuit breaker with no
keys at all, follow [Try failover locally](docs/ROUTING.md#try-failover-locally).

The demo key belongs to team `demo`, which has a $5 monthly budget. Every request it
makes is priced and recorded:

```bash
docker compose -f deploy/compose.yaml --env-file .env exec postgres \
  psql -U switchboard -c "SELECT model, status, input_tokens, output_tokens, cost_usd
                          FROM usage_events ORDER BY created_at DESC LIMIT 5"
```

More keys, with their own limits and budgets:

```bash
make team TEAM=research ARGS="--budget 100 --soft-budget 80"
make key TEAM=research NAME=alice ARGS="--budget 10 --rpm 60"   # prints the secret once
make revoke PREFIX=sb_live_4f2a                                 # rejected within a second
```

## Development

```bash
make install            # dependencies and git hooks
make lint test          # ruff, mypy --strict, actionlint; unit and component tests
make test-integration   # the OpenAI Python SDK against the running stack
make test-sdk-ts        # the OpenAI TypeScript SDK: a tool round trip via the Anthropic path
make test-live          # real Claude and OpenAI through the gateway (needs keys in .env)
make bench-quick        # gateway-added latency vs a direct baseline (about 8 min)
make reload             # reload config/routing.yaml now (edits are also picked up within 1 s)
make down
```

Ports bind to `127.0.0.1` and can be changed in `.env` (created from `.env.example`).

## License

[Apache-2.0](LICENSE)
