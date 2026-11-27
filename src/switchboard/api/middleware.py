"""Outermost ASGI middleware: request ids, log context and the access log.

Written as plain ASGI rather than Starlette's ``BaseHTTPMiddleware``, which buffers
and adds per-request overhead to streaming responses. It wraps the whole FastAPI app,
including Starlette's error middleware, so even a 500 carries ``x-request-id``.

It puts two things in the ASGI scope state for handlers:

* ``received_at``: ``time.perf_counter()`` when the request arrived, the start of
  every gateway latency measurement.
* ``log``: a dict that handlers fill with request fields (route, tokens, outcome).
  They are written on the single ``http.request`` line, so each request logs once.
"""

import re
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = b"x-request-id"
_VALID_REQUEST_ID = re.compile(rb"^[A-Za-z0-9._:-]{1,128}$")
_QUIET_PATHS = frozenset({"/health", "/ready"})

_log = structlog.get_logger("switchboard.http")


def incoming_request_id(scope: Scope) -> str | None:
    """Return the client's ``X-Request-ID`` if it is safe to reuse, else None."""
    for name, value in scope.get("headers", ()):
        if name == REQUEST_ID_HEADER:
            return value.decode("ascii") if _VALID_REQUEST_ID.match(value) else None
    return None


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, quiet_paths: frozenset[str] = _QUIET_PATHS) -> None:
        self.app = app
        self._quiet_paths = quiet_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        request_id = incoming_request_id(scope) or str(uuid.uuid7())
        log_fields: dict[str, object] = {}
        state = scope.setdefault("state", {})
        state.update(request_id=request_id, received_at=start, log=log_fields)
        header = (REQUEST_ID_HEADER, request_id.encode("ascii"))
        status_code = 500
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                message = {**message, "headers": [*message.get("headers", ()), header]}
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception:
                if not response_started:
                    raise
                # Starlette already sent the 500 and re-raised so the server can log it.
                # Log it here instead, while the request id is still bound.
                _log.exception("http.unhandled_exception", path=scope["path"])
            finally:
                path = scope["path"]
                if path not in self._quiet_paths or status_code >= 500:
                    fields: dict[str, object] = {
                        "method": scope["method"],
                        "path": path,
                        "status": status_code,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                    fields.update(log_fields)  # handlers know more (e.g. 499 for a disconnect)
                    _log.info("http.request", **fields)
