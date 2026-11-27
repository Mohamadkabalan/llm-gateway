"""The provider adapter interface (PLAN.md §3.1)."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Protocol

from switchboard.routing.config import Timeouts
from switchboard.routing.table import ModelTarget
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.embeddings import EmbeddingRequest
from switchboard.schemas.events import StreamEvent


@dataclass(slots=True)
class CallTimings:
    """``time.perf_counter()`` stamps an adapter records for overhead metrics."""

    upstream_start: float | None = None  # just before the upstream request is sent
    first_content_read: float | None = None  # when the first content event's bytes arrived


@dataclass(frozen=True, slots=True)
class CallContext:
    request_id: str
    timeouts: Timeouts
    attempt: int = 1
    timings: CallTimings = field(default_factory=CallTimings)
    # Request fields the adapter could not send and dropped (unsupported_params: drop).
    # Reported to the client in x-switchboard-dropped-params.
    dropped_params: list[str] = field(default_factory=list)
    # Who the request belongs to, for state kept between requests (the thinking
    # sidecar). The team id from Phase 4; a single shared scope until then.
    scope: str = "default"


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    body: dict[str, Any]  # the OpenAI-shaped response, returned to the client as-is
    prompt_tokens: int


class ProviderAdapter(Protocol):
    name: str

    def stream_chat(
        self, request: ChatRequest, target: ModelTarget, ctx: CallContext
    ) -> AsyncGenerator[StreamEvent]:
        """Open one streaming upstream call and yield normalized events.

        The first event is always ``MessageStart``. Failures before or during the
        stream raise ``ProviderError`` (already classified). Closing the iterator
        (client disconnect, timeout, shutdown) must close the upstream connection.
        """
        ...

    async def embed(
        self, request: EmbeddingRequest, target: ModelTarget, ctx: CallContext
    ) -> EmbeddingResult:
        """One embeddings call; raises ``ProviderError`` if the provider has none."""
        ...

    async def aclose(self) -> None:
        """Close pooled connections."""
        ...
