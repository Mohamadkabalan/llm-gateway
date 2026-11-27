"""Request body reading with a size limit enforced while the body streams in."""

from typing import Any

import orjson
from starlette.requests import Request

from switchboard.api.errors import GatewayError


async def read_json(request: Request, max_bytes: int) -> Any:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise _too_large(max_bytes)

    parts: list[bytes] = []
    size = 0
    async for part in request.stream():
        size += len(part)
        if size > max_bytes:
            raise _too_large(max_bytes)
        parts.append(part)
    try:
        return orjson.loads(b"".join(parts))
    except orjson.JSONDecodeError:
        raise GatewayError(
            400,
            "We could not parse the JSON body of your request.",
            code="invalid_json",
        ) from None


def _too_large(max_bytes: int) -> GatewayError:
    return GatewayError(
        413,
        f"Request body is larger than the limit of {max_bytes} bytes.",
        code="request_too_large",
    )
