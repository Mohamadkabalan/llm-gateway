SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Host ports and credentials come from .env (created from .env.example on first `make up`).
-include .env
SB_GATEWAY_PORT ?= 7080
SB_REDIS_PORT ?= 7379
SB_POSTGRES_PORT ?= 7432
SB_FAKE_PORT ?= 7800
POSTGRES_USER ?= switchboard
POSTGRES_PASSWORD ?= switchboard-local-dev
POSTGRES_DB ?= switchboard

COMPOSE := docker compose -f deploy/compose.yaml --env-file .env
ACTIONLINT_IMAGE := rhysd/actionlint:1.7.12

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- setup / development ---------------------------------------------------------------

.env:
	cp .env.example .env
	@echo "Created .env from .env.example (add provider keys there if you have them)."

.PHONY: secrets
secrets: .env ## Make sure .env has a key pepper and a demo API key (random ones if missing)
	@python3 scripts/ensure_secrets.py .env

.PHONY: install
install: ## Install dependencies and git hooks
	uv sync --locked
	uv run pre-commit install

.PHONY: dev
dev: secrets ## Run the gateway on the host with auto-reload (Postgres/Redis from compose)
	$(COMPOSE) up -d --wait postgres redis
	$(COMPOSE) run --rm setup
	SWITCHBOARD_LOG_FORMAT=console \
	SWITCHBOARD_REDIS_URL=redis://127.0.0.1:$(SB_REDIS_PORT)/0 \
	SWITCHBOARD_DATABASE_URL=postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@127.0.0.1:$(SB_POSTGRES_PORT)/$(POSTGRES_DB) \
	uv run --env-file .env uvicorn switchboard.app:app_factory --factory --reload \
		--host 127.0.0.1 --port $(SB_GATEWAY_PORT) --no-server-header

.PHONY: validate-config
validate-config: ## Validate config/routing.yaml
	uv run switchboard validate-config config/routing.yaml

# --- quality ---------------------------------------------------------------------------

.PHONY: fmt
fmt: ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: lint
lint: ## Lint, format check and strict type check (plus actionlint if Docker is available)
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	@if command -v docker >/dev/null; then \
		docker run --rm -v "$$PWD:/repo" -w /repo $(ACTIONLINT_IMAGE) -color; \
	else echo "docker not found: skipping actionlint"; fi

.PHONY: test
test: ## Unit and component tests (no Docker needed)
	uv run pytest -m "not integration and not live"

.PHONY: test-unit
test-unit: ## Run unit tests only
	uv run pytest tests/unit

.PHONY: test-integration
test-integration: secrets ## Tests against the running compose stack (run `make up` first)
	@# Not echoed: the database URL carries the password from .env.
	@SB_GATEWAY_URL=http://127.0.0.1:$(SB_GATEWAY_PORT) \
	SB_DATABASE_URL=postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@127.0.0.1:$(SB_POSTGRES_PORT)/$(POSTGRES_DB) \
	SB_REDIS_URL=redis://127.0.0.1:$(SB_REDIS_PORT)/0 \
	SB_FAKE_URL=http://127.0.0.1:$(SB_FAKE_PORT) \
	uv run --env-file .env pytest -m integration

.PHONY: cov
cov: ## Unit and component tests with a coverage report
	uv run pytest -m "not integration and not live" --cov --cov-report=term-missing

.PHONY: test-sdk-ts
test-sdk-ts: secrets ## The OpenAI TypeScript SDK against the running stack (run `make up` first)
	cd tests/sdk-ts && npm ci --no-audit --no-fund --ignore-scripts
	OPENAI_BASE_URL=http://127.0.0.1:$(SB_GATEWAY_PORT)/v1 \
	OPENAI_API_KEY="$$(grep '^SWITCHBOARD_DEMO_KEY=' .env | cut -d= -f2-)" \
	node tests/sdk-ts/tool_roundtrip.mjs

.PHONY: test-live
test-live: ## Real Claude and OpenAI through the gateway; keys from .env (well under 1 cent)
	uv run --env-file .env pytest -m live tests/live

.PHONY: record-fixtures
record-fixtures: ## Record real provider streams as test fixtures; keys from .env (under 1 cent)
	uv run --env-file .env python scripts/record_fixtures.py

# --- stack -----------------------------------------------------------------------------

.PHONY: up
up: secrets ## Build and start the full stack, waiting until every service is healthy
	$(COMPOSE) up -d --build --wait
	@echo "Gateway:    http://127.0.0.1:$(SB_GATEWAY_PORT)/ready"
	@echo "Grafana:    http://127.0.0.1:$${SB_GRAFANA_PORT:-7300}"
	@echo "Prometheus: http://127.0.0.1:$${SB_PROMETHEUS_PORT:-7090}"

.PHONY: down
down: ## Stop the stack (keeps data volumes)
	$(COMPOSE) down

.PHONY: clean
clean: ## Stop the stack and delete its data volumes
	$(COMPOSE) down -v

.PHONY: migrate
migrate: secrets ## Apply database migrations (`make up` does this too)
	$(COMPOSE) run --rm setup

.PHONY: team
team: ## Create a team: make team TEAM=research ARGS="--budget 100 --soft-budget 80"
	$(COMPOSE) exec gateway switchboard create-team "$(TEAM)" $(ARGS)

.PHONY: key
key: ## Create an API key: make key TEAM=demo NAME=alice ARGS="--budget 10 --rpm 60"
	$(COMPOSE) exec gateway switchboard create-key --team "$(TEAM)" --name "$(NAME)" $(ARGS)

.PHONY: revoke
revoke: ## Revoke an API key everywhere: make revoke PREFIX=sb_live_4f2a
	$(COMPOSE) exec gateway switchboard revoke-key --prefix "$(PREFIX)"

.PHONY: reload
reload: ## Reload config/routing.yaml in every gateway replica (SIGHUP; the file watch also does this)
	$(COMPOSE) kill -s HUP gateway

.PHONY: ps
ps: ## Show stack status
	$(COMPOSE) ps

.PHONY: logs
logs: ## Follow gateway logs
	$(COMPOSE) logs -f gateway

# --- measurement -----------------------------------------------------------------------

.PHONY: bench
bench: ## Gateway-added latency at 50/200/500 streams, 3 reps (~70 min; PLAN.md §16.1)
	uv run python -m bench.run --levels 50,200,500 --reps 3 --warmup 30 --duration 180

.PHONY: bench-quick
bench-quick: ## Same matrix, 1 rep, short runs (~8 min); results are git-ignored
	uv run python -m bench.run --levels 50,200,500 --reps 1 --warmup 10 --duration 30 --scratch

.PHONY: bench-report
bench-report: ## Rebuild a run's report: make bench-report RUN=bench/results/<run-id>
	uv run python -m bench.report $(RUN)

.PHONY: chaos
chaos: ## Kill/degrade the primary provider under load (Phase 8)
	@echo "chaos arrives in Phase 8." && exit 1

.PHONY: cost-demo
cost-demo: ## Replay 1,000 requests with and without caching/routing (Phase 8)
	@echo "cost-demo arrives in Phase 8." && exit 1
