# syntax=docker/dockerfile:1.10
# Switchboard gateway image: multi-stage, non-root, no build tools in the runtime layer.

ARG PYTHON_IMAGE=python:3.14.9-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.8

FROM ${UV_IMAGE} AS uv

# --- build: resolve the locked dependencies into /app/.venv -----------------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src

# Dependencies first, so source edits don't invalidate this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- runtime ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="switchboard" \
      org.opencontainers.image.description="Self-hostable OpenAI-compatible LLM gateway" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN groupadd --system --gid 10001 switchboard \
 && useradd --system --uid 10001 --gid switchboard --no-create-home --shell /usr/sbin/nologin switchboard

COPY --from=build --chown=root:root /app/.venv /app/.venv
COPY --chown=root:root config /app/config
COPY --chown=root:root migrations /app/migrations

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SWITCHBOARD_ROUTING_CONFIG=/app/config/routing.yaml
WORKDIR /app
USER 10001:10001
EXPOSE 8080 9100

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).status == 200 else 1)"]

ENTRYPOINT ["switchboard"]
CMD ["serve"]
