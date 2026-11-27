"""Data-plane authentication: ``Authorization: Bearer sb_live_…`` → ``KeyContext``.

Keys are virtual keys from the ``api_keys`` table, looked up by their HMAC through the
cached ``KeyResolver`` (``switchboard.auth.resolver``).
"""

from starlette.requests import Request

from switchboard.api.errors import GatewayError
from switchboard.auth.context import KeyContext
from switchboard.auth.resolver import AuthError, KeyResolver

_UNAUTHORIZED = {"www-authenticate": "Bearer"}


async def authenticate(request: Request, keys: KeyResolver) -> KeyContext:
    header = request.headers.get("authorization")
    if not header:
        raise GatewayError(
            401,
            "You didn't provide an API key. Send it as 'Authorization: Bearer <key>'.",
            code="missing_api_key",
            headers=_UNAUTHORIZED,
        )
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        raise GatewayError(
            401, "Incorrect API key provided.", code="invalid_api_key", headers=_UNAUTHORIZED
        )
    try:
        return await keys.resolve(token.strip())
    except AuthError as exc:
        headers = _UNAUTHORIZED if exc.status == 401 else None
        raise GatewayError(exc.status, exc.message, code=exc.code, headers=headers) from None
