"""``GET /v1/models`` and ``GET /v1/models/{id}``: the aliases and models a key can use
(its own and its team's ``allowed_models``, when set)."""

from typing import Any

from fastapi import APIRouter, Request

from switchboard.api.auth import authenticate
from switchboard.api.errors import GatewayError, ORJSONResponse
from switchboard.state import get_state

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> ORJSONResponse:
    state = get_state(request)
    key = await authenticate(request, state.keys)
    data = [
        _model(name, owner)
        for name, owner, _kind in state.config.current.table.listing()
        if key.may_use(name)
    ]
    return ORJSONResponse({"object": "list", "data": data})


@router.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str, request: Request) -> ORJSONResponse:
    state = get_state(request)
    key = await authenticate(request, state.keys)
    for name, owner, _kind in state.config.current.table.listing():
        if name == model_id and key.may_use(name):
            return ORJSONResponse(_model(name, owner))
    raise GatewayError(
        404,
        f"The model '{model_id}' does not exist or you do not have access to it.",
        param="model",
        code="model_not_found",
    )


def _model(name: str, owner: str) -> dict[str, Any]:
    # `created` is required by the OpenAI schema but has no meaning for an alias.
    return {"id": name, "object": "model", "created": 0, "owned_by": owner}
