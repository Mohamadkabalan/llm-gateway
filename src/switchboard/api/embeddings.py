"""``POST /v1/embeddings``: authenticate, validate, route, call, return.

Embeddings are short non-streaming calls, so this is a plain request handler: no commit
point, and the total timeout bounds the call.
"""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from switchboard.api.auth import authenticate
from switchboard.api.body import read_json
from switchboard.api.errors import ORJSONResponse, error_response
from switchboard.observability.metrics import REQUEST_DURATION, REQUESTS
from switchboard.providers.base import CallContext
from switchboard.providers.errors import CLIENT_FAULT_KINDS, ErrorKind, ProviderError
from switchboard.routing.config import ModelKind
from switchboard.schemas.embeddings import EmbeddingRequest
from switchboard.state import get_state

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> ORJSONResponse:
    state = get_state(request)
    authenticate(request, state.settings.dev_api_key)
    payload = await read_json(request, state.settings.max_request_bytes)
    try:
        embedding_request = EmbeddingRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_url=False)) from None

    plan = state.routing_table.resolve(embedding_request.model, ModelKind.EMBEDDING)
    target = plan.targets[0]
    adapter = state.adapters[target.provider]  # routing only returns enabled providers

    scope_state = request.scope["state"]
    ctx = CallContext(request_id=scope_state["request_id"], timeouts=target.timeouts)
    headers = {
        "x-switchboard-provider": target.provider,
        "x-switchboard-model": target.name,
        "x-switchboard-attempts": str(ctx.attempt),
    }
    labels = (plan.route_label, target.provider, target.name)
    log: dict[str, Any] = scope_state["log"]
    log.update(alias=plan.alias, provider=target.provider, model=target.name, stream=False)
    try:
        async with asyncio.timeout(target.timeouts.total):
            result = await adapter.embed(embedding_request, target, ctx)
    except TimeoutError:
        error: ProviderError | None = ProviderError(
            ErrorKind.TIMEOUT,
            f"embeddings exceeded the total timeout of {target.timeouts.total:g}s",
        )
    except ProviderError as exc:
        error = exc
    else:
        error = None
    finally:
        REQUEST_DURATION.labels(*labels, "false").observe(
            time.perf_counter() - scope_state["received_at"]
        )

    if ctx.dropped_params:
        headers["x-switchboard-dropped-params"] = ",".join(sorted(set(ctx.dropped_params)))
    if error is not None:
        client = error.client_error
        outcome = "client_error" if error.kind in CLIENT_FAULT_KINDS else "upstream_error"
        REQUESTS.labels(*labels, str(client.status), outcome).inc()
        log.update(outcome=outcome, error_kind=error.kind.value, error_detail=error.message[:500])
        return error_response(
            client.status,
            error.client_message,
            error_type=client.error_type,
            param=error.param,
            code=client.code,
            headers=headers,
        )
    REQUESTS.labels(*labels, "200", "success").inc()
    log.update(outcome="success", prompt_tokens=result.prompt_tokens)
    return ORJSONResponse(result.body, headers=headers)
