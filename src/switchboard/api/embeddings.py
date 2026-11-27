"""``POST /v1/embeddings``: authenticate, validate, route, call, return.

Embeddings are short non-streaming calls: retries, fallbacks and circuit breakers apply
as for chat, without a commit point or hedging. ``first_token`` bounds each attempt,
and the primary's ``total`` bounds the whole request.
"""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from switchboard.accounting.ledger import RequestLedger, Served
from switchboard.api.admission import check_model_access, open_ledger, recorded
from switchboard.api.auth import authenticate
from switchboard.api.body import read_json
from switchboard.api.chat import RESERVATION_MARGIN_S
from switchboard.api.errors import ORJSONResponse, error_response
from switchboard.api.tags import request_tags
from switchboard.observability.metrics import REQUEST_DURATION, REQUESTS
from switchboard.providers.base import EmbeddingResult
from switchboard.providers.errors import CLIENT_FAULT_KINDS, ErrorKind, ProviderError
from switchboard.reliability.executor import Committed, ExhaustedError, Run
from switchboard.routing.config import ModelKind
from switchboard.routing.table import RoutePlan, RouteRequest
from switchboard.schemas.embeddings import EmbeddingRequest
from switchboard.schemas.events import Usage
from switchboard.state import get_state

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> ORJSONResponse:
    state = get_state(request)
    key = await authenticate(request, state.keys)
    snapshot = state.config.current
    ledger = open_ledger(state, request, key, endpoint="embeddings")
    with recorded(ledger):
        payload = await read_json(request, state.settings.max_request_bytes)
        if isinstance(payload, dict) and isinstance(payload.get("model"), str):
            ledger.describe(requested_model=payload["model"])
        tags = request_tags(request)
        try:
            embedding_request = EmbeddingRequest.model_validate(payload)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(include_url=False)) from None
        ledger.describe(
            requested_model=embedding_request.model, end_user=embedding_request.user, tags=tags
        )
        check_model_access(key, embedding_request.model)
        route_request = RouteRequest(
            model=embedding_request.model,
            kind=ModelKind.EMBEDDING,
            key=key.key_name,
            team=key.team_name,
            tags=key.tags | tags,
            sticky_subject=str(key.key_id),
            embedding_chars=embedding_request.text_length(),
        )
        plan = snapshot.table.plan(route_request)
        total = plan.primary.timeouts.total
        await ledger.admit(
            plan, route_request.prompt_tokens, 0, hold_s=total + RESERVATION_MARGIN_S
        )

    scope_state = request.scope["state"]
    run = Run(
        plan=plan,
        request_id=scope_state["request_id"],
        scope=str(key.team_id),
        deadline=asyncio.get_running_loop().time() + total,
    )
    error: ProviderError | None = None
    try:
        async with asyncio.timeout_at(run.deadline):
            committed = await state.executor.embed(run, embedding_request, snapshot.adapters)
    except TimeoutError:
        error = ProviderError(
            ErrorKind.TIMEOUT, f"embeddings exceeded the total timeout of {total:g}s"
        )
    except ExhaustedError as exc:
        error = exc.error

    return _reply(
        plan,
        run,
        ledger=ledger,
        committed=committed if error is None else None,
        error=error,
        scope_state=scope_state,
    )


def _reply(
    plan: RoutePlan,
    run: Run,
    *,
    ledger: RequestLedger,
    committed: Committed[EmbeddingResult] | None,
    error: ProviderError | None,
    scope_state: dict[str, Any],
) -> ORJSONResponse:
    """Account for the call, record metrics and logs, and build the response."""
    target = committed.target if committed is not None else (run.last_target or plan.primary)
    usage = Usage(input=committed.value.prompt_tokens) if committed is not None else None
    cost = ledger.finish(
        Served(
            target=committed.target if committed is not None else run.last_target,
            outcome=_outcome(error),
            http_status=200 if error is None else error.client_error.status,
            usage=usage,
            error_kind=error.kind.value if error is not None else None,
            fallback=committed is not None and committed.target is not plan.primary,
            attempts=[
                {"model": a.model, "outcome": a.outcome, "http_status": a.upstream_status}
                for a in run.attempts
            ],
        )
    )
    labels = (plan.route_label, target.provider, target.name)
    REQUEST_DURATION.labels(*labels, "false").observe(
        time.perf_counter() - scope_state["received_at"]
    )
    headers = {
        "x-switchboard-provider": target.provider,
        "x-switchboard-model": target.name,
        "x-switchboard-attempts": str(run.upstream_calls),
        **ledger.rate_headers(),
    }
    log: dict[str, Any] = scope_state["log"]
    log.update(
        alias=plan.alias,
        provider=target.provider,
        model=target.name,
        stream=False,
        attempts=run.upstream_calls,
    )
    if len(run.attempts) > 1 or error is not None:
        log["attempt_log"] = [str(attempt) for attempt in run.attempts]

    if error is not None:
        client = error.client_error
        outcome = "client_error" if error.kind in CLIENT_FAULT_KINDS else "upstream_error"
        REQUESTS.labels(*labels, str(client.status), outcome).inc()
        log.update(outcome=outcome, error_kind=error.kind.value, error_detail=error.message[:500])
        if error.kind is ErrorKind.RATE_LIMITED and error.retry_after is not None:
            headers["retry-after"] = str(max(1, round(error.retry_after)))
        return error_response(
            client.status,
            error.client_message,
            error_type=client.error_type,
            param=error.param,
            code=client.code,
            headers=headers,
        )
    assert committed is not None  # noqa: S101 (no error means a committed result)
    if committed.target is not plan.primary:
        headers["x-switchboard-fallback"] = "true"
    if committed.ctx.dropped_params:
        headers["x-switchboard-dropped-params"] = ",".join(
            sorted(set(committed.ctx.dropped_params))
        )
    REQUESTS.labels(*labels, "200", "success").inc()
    log.update(outcome="success", prompt_tokens=committed.value.prompt_tokens)
    if cost.usd:
        log["cost_usd"] = str(cost.usd)
    return ORJSONResponse(committed.value.body, headers=headers)


def _outcome(error: ProviderError | None) -> str:
    if error is None:
        return "success"
    return "client_error" if error.kind in CLIENT_FAULT_KINDS else "upstream_error"
