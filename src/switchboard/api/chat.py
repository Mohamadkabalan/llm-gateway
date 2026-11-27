"""``POST /v1/chat/completions``: authenticate, validate, route, admit, then hand off.

Everything that can fail cheaply (auth, size, validation, routing, rate limits,
budgets) fails here with a normal error response, and every refusal after
authentication is still recorded in the ledger. The upstream work (retries,
fallbacks, streaming) runs in ``ChatResponder``.
"""

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.responses import Response

from switchboard.accounting.ledger import output_estimate
from switchboard.api.admission import check_model_access, open_ledger, recorded
from switchboard.api.auth import authenticate
from switchboard.api.body import read_json
from switchboard.api.errors import GatewayError
from switchboard.api.responder import ChatResponder
from switchboard.api.tags import request_tags
from switchboard.routing.config import ModelKind
from switchboard.routing.table import RouteRequest
from switchboard.schemas.chat import ChatRequest
from switchboard.state import get_state

router = APIRouter()

RESERVATION_MARGIN_S = 60.0  # how long past the request deadline a reservation lives


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    state = get_state(request)
    key = await authenticate(request, state.keys)
    snapshot = state.config.current  # this request keeps it, even across a reload
    ledger = open_ledger(state, request, key, endpoint="chat")
    with recorded(ledger):
        payload = await read_json(request, state.settings.max_request_bytes)
        if isinstance(payload, dict) and isinstance(payload.get("model"), str):
            ledger.describe(requested_model=payload["model"])
        tags = request_tags(request)
        chat = parse_chat_request(payload)
        ledger.describe(
            requested_model=chat.model, stream=chat.stream, end_user=chat.user, tags=tags
        )
        check_model_access(key, chat.model)
        route_request = RouteRequest(
            model=chat.model,
            kind=ModelKind.CHAT,
            chat=chat,
            key=key.key_name,
            team=key.team_name,
            tags=key.tags | tags,
            sticky_subject=str(key.key_id),
        )
        plan = snapshot.table.plan(route_request)
        primary = plan.primary
        await ledger.admit(
            plan,
            route_request.prompt_tokens,
            output_estimate(chat.output_limit, primary),
            hold_s=primary.timeouts.total + RESERVATION_MARGIN_S,
        )

    scope_state = request.scope["state"]
    return ChatResponder(
        chat=chat,
        plan=plan,
        executor=state.executor,
        adapters=snapshot.adapters,
        request_id=scope_state["request_id"],
        scope=str(key.team_id),  # thinking-sidecar entries never cross teams
        lifecycle=state.lifecycle,
        received_at=scope_state["received_at"],
        log_fields=scope_state["log"],
        ledger=ledger,
    )


def parse_chat_request(payload: object) -> ChatRequest:
    try:
        chat = ChatRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_url=False)) from None
    # The normalized event model carries one choice and no logprobs, so these would be
    # silently wrong rather than unsupported. Refuse them clearly instead.
    if chat.n is not None and chat.n > 1:
        raise GatewayError(400, "n > 1 is not supported.", param="n", code="unsupported_parameter")
    if chat.logprobs:
        raise GatewayError(
            400, "logprobs is not supported.", param="logprobs", code="unsupported_parameter"
        )
    return chat
