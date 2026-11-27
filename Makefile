SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Host ports and credentials come from .env (created from .env.example on first `make up`).
-include .env
SB_GATEWAY_PORT ?= 7080
SB_REDIS_PORT ?= 7379
SB_POSTGRES_PORT ?= 7432
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

.PHONY: install
install: ## Install dependencies and git hooks
	uv sync --locked
	uv run pre-commit install

.PHONY: dev
dev: .env ## Run the gateway on the host with auto-reload (Postgres/Redis from compose)
	$(COMPOSE) up -d --wait postgres redis
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
test: ## Run unit tests (integration tests join in Phase 1)
	uv run pytest -m "not integration and not live"

.PHONY: test-unit
test-unit: ## Run unit tests only
	uv run pytest tests/unit

.PHONY: cov
cov: ## Unit tests with a coverage report
	uv run pytest tests/unit --cov --cov-report=term-missing

.PHONY: test-live
test-live: ## Contract tests against real providers (Phase 2)
	@echo "test-live arrives in Phase 2 (needs provider keys)." && exit 1

# --- stack -----------------------------------------------------------------------------

.PHONY: up
up: .env ## Build and start the full stack, waiting until every service is healthy
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

.PHONY: ps
ps: ## Show stack status
	$(COMPOSE) ps

.PHONY: logs
logs: ## Follow gateway logs
	$(COMPOSE) logs -f gateway

# --- measurement (implemented in later phases) -----------------------------------------

.PHONY: bench
bench: ## Load test: gateway-added latency at 50/200/500 streams (Phase 1/8)
	@echo "bench arrives in Phase 1 (harness) and Phase 8 (published run)." && exit 1

.PHONY: chaos
chaos: ## Kill/degrade the primary provider under load (Phase 8)
	@echo "chaos arrives in Phase 8." && exit 1

.PHONY: cost-demo
cost-demo: ## Replay 1,000 requests with and without caching/routing (Phase 8)
	@echo "cost-demo arrives in Phase 8." && exit 1
