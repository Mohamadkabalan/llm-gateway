# Switchboard

A self-hostable LLM gateway. Applications call one OpenAI-compatible endpoint with a
virtual key. Switchboard does the rest:

- routes each request to the right provider and model
- retries, and falls back to another model when one fails
- caches responses
- enforces per-key and per-team budgets and rate limits
- records cost, tokens and latency for every call

> **Status: Phase 2 (providers).** `/v1/chat/completions`, `/v1/embeddings` and
> `/v1/models` work end to end with OpenAI, Anthropic (native Messages API, including
> tool calls, images, structured outputs and thinking across tool loops) and Ollama.
> What is translated, dropped or refused is in [docs/PROVIDERS.md](docs/PROVIDERS.md).
> The local stack includes a fake provider that speaks both the OpenAI and Anthropic
> APIs, so it runs without any provider keys. The plan, architecture and
> phase-by-phase acceptance criteria are in [PLAN.md](PLAN.md); design decisions are
> in [docs/adr](docs/adr). The full README (quickstart with real providers, SDK
> examples, benchmarks, comparison with other gateways) is written in Phase 8, once
> there are measured results to show.

## Try it

Requirements: [uv](https://docs.astral.sh/uv/), Docker with Compose v2, make.

```bash
make up     # builds and starts the stack; puts a random dev API key in .env
export OPENAI_BASE_URL=http://127.0.0.1:7080/v1
export OPENAI_API_KEY="$(grep '^SWITCHBOARD_DEV_API_KEY=' .env | cut -d= -f2-)"
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

Every response carries `x-request-id`, `x-switchboard-provider` and
`x-switchboard-model`. Errors use the OpenAI error shape, so the SDK raises its usual
exceptions.

## Development

```bash
make install            # dependencies and git hooks
make lint test          # ruff, mypy --strict, actionlint; unit and component tests
make test-integration   # the OpenAI Python SDK against the running stack
make test-sdk-ts        # the OpenAI TypeScript SDK: a tool round trip via the Anthropic path
make test-live          # real Claude and OpenAI through the gateway (needs keys in .env)
make bench-quick        # gateway-added latency vs a direct baseline (about 8 min)
make down
```

Ports bind to `127.0.0.1` and can be changed in `.env` (created from `.env.example`).

## License

[Apache-2.0](LICENSE)
