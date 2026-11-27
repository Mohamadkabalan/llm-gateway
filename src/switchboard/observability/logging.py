"""Structured logging: one JSON object per line, with request context merged in.

Both structlog and stdlib loggers (uvicorn, asyncio, libraries) go through the same
processor chain, so every line has the same shape. Values under secret-looking keys
are masked before rendering; prompt content is never logged by default (see PLAN.md §10).
"""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO

import orjson
import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from switchboard.settings import LogFormat, LogLevel

_SECRET_MARKERS = ("authorization", "api_key", "apikey", "password", "secret", "pepper", "cookie")
_MASK = "***"


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Mask values whose key looks like a credential, recursing into nested mappings."""
    return _redact(event_dict)


def _redact(mapping: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in mapping.items():
        lowered = key.lower()
        # "token" alone, or as a suffix (access_token), is a credential; "prompt_tokens"
        # and "max_tokens" are counts.
        if lowered.endswith("token") or any(marker in lowered for marker in _SECRET_MARKERS):
            mapping[key] = _MASK
        elif isinstance(value, MutableMapping):
            mapping[key] = _redact(dict(value))
    return mapping


def drop_color_message(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """uvicorn attaches an ANSI-coloured duplicate of each message; JSON logs don't want it."""
    event_dict.pop("color_message", None)
    return event_dict


def _orjson_dumps(obj: Any, **_kwargs: Any) -> str:
    return orjson.dumps(obj, default=str).decode()


def configure_logging(
    level: LogLevel = "info", fmt: LogFormat = LogFormat.JSON, stream: TextIO | None = None
) -> None:
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        drop_color_message,
        redact_secrets,
    ]
    renderer: Processor
    if fmt is LogFormat.JSON:
        shared.append(structlog.processors.dict_tracebacks)
        renderer = structlog.processors.JSONRenderer(serializer=_orjson_dumps)
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; route its records through ours instead.
    for name in ("uvicorn", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    # The gateway writes its own access log with request ids (api.middleware).
    logging.getLogger("uvicorn.access").disabled = True
    # httpx logs every upstream request at INFO; the access log already covers them.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
