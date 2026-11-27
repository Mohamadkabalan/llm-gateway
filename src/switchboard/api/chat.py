"""``POST /v1/chat/completions``: authenticate, validate, route, then hand off.

Everything that can fail cheaply (auth, size, validation, routing) fails here with a
normal error response. The upstream call itself runs in ``ChatResponder``.
"""

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.responses import Response

from switchboard.api.auth import authenticate
from switchboard.api.body import read_json
from switchboard.api.errors import GatewayError
from switchboard.api.responder import ChatResponder
from switchboard.providers.base import CallContext
from switchboard.routing.capabilities import missing_capabilities
from switchboard.routing.config import ModelKind
from switchboard.schemas.chat import ChatRequest
from switchboard.state import get_state

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    state = get_state(request)
    authenticate(request, state.settings.dev_api_key)
    chat = parse_chat_request(await read_json(request, state.settings.max_request_bytes))

    plan = state.routing_table.resolve(chat.model, ModelKind.CHAT)
    target = plan.targets[0]
    if missing := missing_capabilities(chat, target.capabilities):
        raise GatewayError(
            400,
            f"The model '{chat.model}' does not support {'; '.join(missing)}.",
            code="model_capability_missing",
        )
    adapter = state.adapters[target.provider]  # routing only returns enabled providers

    scope_state = request.scope["state"]
    return ChatResponder(
        chat=chat,
        plan=plan,
        adapter=adapter,
        ctx=CallContext(request_id=scope_state["request_id"], timeouts=target.timeouts),
        lifecycle=state.lifecycle,
        received_at=scope_state["received_at"],
        log_fields=scope_state["log"],
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
