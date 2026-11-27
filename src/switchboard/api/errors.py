"""Errors in the OpenAI shape: ``{"error": {"message", "type", "param", "code"}}``.

OpenAI SDKs parse this envelope into typed exceptions, so every error the gateway
returns uses it, including 404s, validation failures and unexpected 500s.
"""

from typing import Any

import orjson
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from switchboard.routing.table import RouteError

_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    405: "invalid_request_error",
    409: "conflict_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_exceeded",
}


class GatewayError(Exception):
    """An error the gateway reports in the OpenAI shape, raised from request handlers."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str | None = None,
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        self.headers = headers


class ORJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


def error_body(
    message: str, error_type: str, *, param: str | None = None, code: str | None = None
) -> dict[str, dict[str, str | None]]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def error_response(
    status: int,
    message: str,
    *,
    error_type: str | None = None,
    param: str | None = None,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> ORJSONResponse:
    resolved = error_type or _TYPE_BY_STATUS.get(status, "server_error")
    return ORJSONResponse(
        error_body(message, resolved, param=param, code=code), status_code=status, headers=headers
    )


async def _gateway_error(_request: Request, exc: GatewayError) -> Response:
    return error_response(
        exc.status,
        exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
        headers=exc.headers,
    )


async def _route_error(_request: Request, exc: RouteError) -> Response:
    return error_response(
        exc.status, exc.message, error_type=exc.error_type, param="model", code=exc.code
    )


async def _http_exception(_request: Request, exc: StarletteHTTPException) -> Response:
    headers = dict(exc.headers) if exc.headers else None
    return error_response(exc.status_code, exc.detail, headers=headers)


async def _validation_error(_request: Request, exc: RequestValidationError) -> Response:
    first = exc.errors()[0] if exc.errors() else {"loc": (), "msg": "Invalid request"}
    loc = [str(part) for part in first.get("loc", ()) if part != "body"]
    param = ".".join(loc) or None
    message = f"Invalid value for '{param}': {first['msg']}" if param else str(first["msg"])
    return error_response(400, message, param=param)


async def _unhandled(request: Request, _exc: Exception) -> Response:
    # The exception itself is logged (with its request id) by RequestContextMiddleware.
    request_id = request.scope.get("state", {}).get("request_id", "unknown")
    return error_response(
        500, f"Internal server error (request id {request_id})", error_type="server_error"
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GatewayError, _gateway_error)  # type: ignore[arg-type]
    app.add_exception_handler(RouteError, _route_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled)
