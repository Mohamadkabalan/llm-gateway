"""Classified upstream failures (PLAN.md §2.2).

Adapters turn every upstream failure into a ``ProviderError`` with a ``kind``. The
kind decides two things: whether a retry or fallback may help (the table in
``switchboard.reliability.policy``) and what the client sees. A provider's own
credentials or quota problem is the gateway's fault, not the client's, so it never
reaches the client as a 401.
"""

import email.utils
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ErrorKind(StrEnum):
    CONNECT = "connect"  # DNS, TCP or TLS failure; nothing was sent
    TIMEOUT = "timeout"  # connect, first-token, idle or total timeout
    RATE_LIMITED = "rate_limited"  # 429 that may clear after retry-after
    QUOTA = "quota"  # 429 for exhausted quota or spend limit; retrying will not help
    OVERLOADED = "overloaded"  # 5xx, 529 or an in-stream error event
    AUTH = "auth"  # the provider rejected our credentials
    CONTEXT_LENGTH = "context_length"  # prompt too long for this model
    BAD_REQUEST = "bad_request"  # other 4xx: the request itself is wrong
    PROTOCOL = "protocol"  # malformed or truncated upstream response
    INVALID_REQUEST = "invalid_request"  # can't be expressed for this provider; not sent
    # Raised by the executor, not by adapters:
    CIRCUIT_OPEN = "circuit_open"  # the target's circuit breaker is open; not sent
    REFUSAL = "refusal"  # refused before the commit point, on a fallback_on_refusal route


@dataclass(frozen=True, slots=True)
class ClientError:
    """How a provider failure is reported to the client, in the OpenAI error shape."""

    status: int
    error_type: str
    code: str


_CLIENT_ERRORS = {
    ErrorKind.CONNECT: ClientError(502, "server_error", "upstream_unavailable"),
    ErrorKind.TIMEOUT: ClientError(504, "server_error", "upstream_timeout"),
    ErrorKind.RATE_LIMITED: ClientError(429, "rate_limit_exceeded", "upstream_rate_limited"),
    ErrorKind.QUOTA: ClientError(503, "server_error", "upstream_quota_exhausted"),
    ErrorKind.OVERLOADED: ClientError(502, "server_error", "upstream_error"),
    ErrorKind.AUTH: ClientError(502, "server_error", "upstream_auth_failed"),
    ErrorKind.CONTEXT_LENGTH: ClientError(400, "invalid_request_error", "context_length_exceeded"),
    ErrorKind.BAD_REQUEST: ClientError(400, "invalid_request_error", "upstream_bad_request"),
    ErrorKind.PROTOCOL: ClientError(502, "server_error", "upstream_protocol_error"),
    ErrorKind.INVALID_REQUEST: ClientError(400, "invalid_request_error", "invalid_request"),
    ErrorKind.CIRCUIT_OPEN: ClientError(503, "server_error", "upstream_circuit_open"),
    ErrorKind.REFUSAL: ClientError(502, "server_error", "upstream_refusal"),
}

# Kinds caused by the request itself rather than by the provider.
CLIENT_FAULT_KINDS = frozenset(
    {ErrorKind.CONTEXT_LENGTH, ErrorKind.BAD_REQUEST, ErrorKind.INVALID_REQUEST}
)

# Kinds where the upstream message describes the client's own request and is safe and
# useful to pass on. Other kinds get a generic message, so provider internals (account
# names, quota details) do not leak to API consumers.
_PASS_THROUGH_MESSAGE = CLIENT_FAULT_KINDS

_GENERIC_MESSAGES = {
    ErrorKind.CONNECT: "Could not connect to the upstream provider.",
    ErrorKind.TIMEOUT: "The upstream provider timed out.",
    ErrorKind.RATE_LIMITED: "The upstream provider is rate limiting requests.",
    ErrorKind.QUOTA: "The upstream provider's quota is exhausted.",
    ErrorKind.OVERLOADED: "The upstream provider returned an error.",
    ErrorKind.AUTH: "The gateway's credentials were rejected by the upstream provider.",
    ErrorKind.PROTOCOL: "The upstream provider sent an invalid response.",
    ErrorKind.CIRCUIT_OPEN: "Every upstream for this model is temporarily unavailable.",
    ErrorKind.REFUSAL: "The upstream provider refused the request.",
}


class ProviderError(Exception):
    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        should_retry: bool | None = None,
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status  # the upstream HTTP status, if there was one
        self.retry_after = retry_after
        self.should_retry = should_retry  # an explicit x-should-retry from the provider
        self.param = param  # the request field at fault, for client errors

        default = _CLIENT_ERRORS[kind]
        self.client_error = ClientError(default.status, default.error_type, code or default.code)

    @property
    def client_message(self) -> str:
        if self.kind in _PASS_THROUGH_MESSAGE:
            return self.message
        return _GENERIC_MESSAGES[self.kind]

    def __repr__(self) -> str:
        return f"ProviderError({self.kind.value}, status={self.status}, {self.message!r})"


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """``retry-after-ms`` (OpenAI), then ``retry-after`` as seconds or an HTTP date."""
    if (value := headers.get("retry-after-ms")) is not None:
        try:
            return max(float(value) / 1000, 0.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


def parse_should_retry(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() == "true"
