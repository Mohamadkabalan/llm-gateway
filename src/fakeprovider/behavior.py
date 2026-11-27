"""What the fake provider does for a request, and where that comes from.

Precedence, lowest to highest: the startup profile (YAML), the runtime control API,
``x-fake-*`` request headers, then a ``fake`` object in the JSON body. The body form
lets tests drive the fake *through* the gateway, which forwards unknown request
fields to OpenAI-compatible upstreams unchanged.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

HEADER_PREFIX = b"x-fake-"


class Fault(StrEnum):
    NONE = "none"
    STATUS = "status"  # respond with `status` and an OpenAI error body
    MID_STREAM_ERROR = "mid_stream_error"  # an SSE error event after `fault_after` chunks
    DISCONNECT = "disconnect"  # drop the connection after `fault_after` chunks
    HANG = "hang"  # never send a byte until the client gives up


class Behavior(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ttft_ms: float = Field(default=0, ge=0, description="Delay before the first content chunk.")
    interval_ms: float = Field(default=0, ge=0, description="Delay between content chunks.")
    chunks: int = Field(default=8, ge=0, le=100_000, description="Content chunks (≈ tokens).")
    # "auto" answers a tool result with text and calls a tool when tools are offered.
    scenario: Literal[
        "auto",
        "text",
        "tool_call",
        "parallel_tool_calls",
        "thinking_tool_call",
        "refusal",
        "max_tokens",
    ] = "text"
    model: str | None = Field(default=None, description="Served model name; default: echo.")

    fault: Fault = Fault.NONE
    fault_rate: float = Field(default=1.0, ge=0, le=1, description="Probability per request.")
    fault_after: int = Field(default=0, ge=0, description="Content chunks sent before a fault.")
    status: int = Field(default=500, ge=400, le=599)
    error_code: str | None = Field(default=None, description="e.g. insufficient_quota")
    retry_after: float | None = Field(default=None, ge=0)

    def merged(self, overrides: Mapping[str, Any]) -> Behavior:
        if not overrides:
            return self
        return Behavior.model_validate({**self.model_dump(), **overrides})


def load_profile(path: str | Path | None) -> Behavior:
    if path is None:
        return Behavior()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Behavior.model_validate(data)


def header_overrides(headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    """``x-fake-ttft-ms: 300`` → ``{"ttft_ms": "300"}``; pydantic converts the types."""
    return {
        name[len(HEADER_PREFIX) :].decode().replace("-", "_"): value.decode()
        for name, value in headers
        if name.startswith(HEADER_PREFIX)
    }
