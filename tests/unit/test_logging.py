import io
import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from switchboard.observability.logging import configure_logging
from switchboard.settings import LogFormat


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    configure_logging("info", LogFormat.JSON, stream=buffer)
    yield buffer
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


def last_line(stream: io.StringIO) -> dict[str, object]:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    parsed: dict[str, object] = json.loads(lines[-1])
    return parsed


def test_structlog_lines_are_json_with_context(stream: io.StringIO) -> None:
    with structlog.contextvars.bound_contextvars(request_id="req-9"):
        structlog.get_logger("switchboard.test").info("thing.happened", count=3)
    line = last_line(stream)
    assert line["event"] == "thing.happened"
    assert line["request_id"] == "req-9"
    assert line["count"] == 3
    assert line["level"] == "info"
    assert line["logger"] == "switchboard.test"
    assert str(line["timestamp"]).endswith("Z")


def test_stdlib_loggers_share_the_format(stream: io.StringIO) -> None:
    logging.getLogger("uvicorn.error").warning(
        "server says %s", "hi", extra={"color_message": "\x1b[1mserver says %s\x1b[0m"}
    )
    line = last_line(stream)
    assert line["event"] == "server says hi"
    assert line["level"] == "warning"
    assert "color_message" not in line


def test_secret_looking_fields_are_masked(stream: io.StringIO) -> None:
    structlog.get_logger().info(
        "outbound", api_key="sk-live-123", headers={"Authorization": "Bearer x"}, model="m"
    )
    line = last_line(stream)
    assert line["api_key"] == "***"
    assert line["headers"] == {"Authorization": "***"}
    assert line["model"] == "m"


def test_token_counts_are_not_mistaken_for_credentials(stream: io.StringIO) -> None:
    structlog.get_logger().info(
        "done", prompt_tokens=12, max_tokens=5, access_token="abc", token="def", session_cookie="c"
    )
    line = last_line(stream)
    assert (line["prompt_tokens"], line["max_tokens"]) == (12, 5)
    assert line["access_token"] == line["token"] == line["session_cookie"] == "***"


def test_httpx_request_lines_are_suppressed(stream: io.StringIO) -> None:
    logging.getLogger("httpx").info("HTTP Request: POST https://upstream")
    assert "HTTP Request" not in stream.getvalue()
