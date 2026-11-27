"""API key check for the data plane.

Phase 1 accepts one static key from ``SWITCHBOARD_DEV_API_KEY``. Phase 4 replaces this
with hashed virtual keys looked up per request; handlers already call it through
``authenticate`` so only this module changes.
"""

import hmac

from pydantic import SecretStr
from starlette.requests import Request

from switchboard.api.errors import GatewayError

_UNAUTHORIZED = {"www-authenticate": "Bearer"}


def authenticate(request: Request, dev_api_key: SecretStr | None) -> None:
    if dev_api_key is None:
        raise GatewayError(
            401, "No API keys are configured on this gateway.", code="invalid_api_key"
        )
    header = request.headers.get("authorization")
    if not header:
        raise GatewayError(
            401,
            "You didn't provide an API key. Send it as 'Authorization: Bearer <key>'.",
            code="missing_api_key",
            headers=_UNAUTHORIZED,
        )
    scheme, _, token = header.partition(" ")
    expected = dev_api_key.get_secret_value().encode()
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip().encode(), expected):
        raise GatewayError(
            401, "Incorrect API key provided.", code="invalid_api_key", headers=_UNAUTHORIZED
        )
