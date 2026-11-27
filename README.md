# Switchboard

A self-hostable LLM gateway. Applications call one OpenAI-compatible endpoint with a
virtual key. Switchboard does the rest:

- routes each request to the right provider and model
- retries, and falls back to another model when one fails
- caches responses
- enforces per-key and per-team budgets and rate limits
- records cost, tokens and latency for every call

> **Status: Phase 0 (foundations).** The plan, architecture and phase-by-phase
> acceptance criteria are in [PLAN.md](PLAN.md). The full README (quickstart, SDK
> examples, benchmarks, comparison with other gateways) is written in Phase 8, once
> there are measured results to show.

## Development

Requirements: [uv](https://docs.astral.sh/uv/), Docker with Compose v2, make.

```bash
make install          # dependencies and git hooks
make lint test        # ruff, mypy --strict, actionlint, unit tests
make up               # full local stack; waits until every service is healthy
curl -s http://127.0.0.1:7080/ready
make down
```

Ports bind to `127.0.0.1` and can be changed in `.env` (created from `.env.example`).

## License

[Apache-2.0](LICENSE)
