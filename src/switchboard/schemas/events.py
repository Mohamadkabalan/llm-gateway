"""Normalized stream events (PLAN.md §3.2).

Every provider adapter turns its upstream stream into these events, and one encoder
turns them back into OpenAI responses. Events are small frozen dataclasses: they are
created per upstream chunk, so they carry no validation logic.
"""

from dataclasses import dataclass
from enum import StrEnum


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


@dataclass(frozen=True, slots=True)
class MessageStart:
    """First event of a response. ``served_model`` is what the provider says it ran."""

    upstream_id: str | None
    served_model: str | None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class RefusalDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """Model reasoning text. Dropped by the encoder unless a route exposes it."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    tool_index: int
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallArgsDelta:
    """A raw fragment of the tool call's JSON arguments, never parsed per chunk."""

    tool_index: int
    fragment: str


@dataclass(frozen=True, slots=True)
class Finish:
    reason: FinishReason


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts in explicit, non-overlapping buckets.

    ``input`` excludes cached tokens; ``output`` includes reasoning tokens, and
    ``reasoning`` is the part of ``output`` spent on reasoning.
    """

    input: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def prompt_total(self) -> int:
        return self.input + self.cache_read + self.cache_write_5m + self.cache_write_1h


type StreamEvent = (
    MessageStart
    | TextDelta
    | RefusalDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallArgsDelta
    | Finish
    | Usage
)

# Events that carry output to the client. Forwarding the first of these is the commit
# point after which fallback is no longer possible (PLAN.md §2.1).
CONTENT_EVENTS = (TextDelta, RefusalDelta, ToolCallStart, ToolCallArgsDelta, Finish)


def is_content(event: StreamEvent) -> bool:
    return isinstance(event, CONTENT_EVENTS)
