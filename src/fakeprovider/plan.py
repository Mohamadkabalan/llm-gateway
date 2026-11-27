"""What one fake response contains, independent of wire format.

Each dialect turns a request into a ``Completion`` (the facts) and then a
``StreamPlan`` (the bytes). The app plays any plan with the same timing and fault
injection, so both dialects behave identically under test and in benchmarks.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from fakeprovider.behavior import Behavior

Scenario = Literal[
    "text", "tool_call", "parallel_tool_calls", "thinking_tool_call", "refusal", "max_tokens"
]
TOOL_SCENARIOS = frozenset({"tool_call", "parallel_tool_calls", "thinking_tool_call"})
LOREM = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt"
WORDS = tuple(LOREM.split())
TOOL_ARGUMENTS = ('{"location": ', '"Paris", ', '"unit": "celsius"}')
REFUSAL_TEXT = "I can't help with that."


@dataclass(frozen=True, slots=True)
class Completion:
    completion_id: str
    created: int
    model: str
    prompt_tokens: int
    scenario: Scenario
    tool_names: tuple[str, ...]
    include_usage: bool

    def call_id(self, index: int) -> str:
        return f"call_{self.completion_id[-8:]}_{index}"

    @property
    def tools_called(self) -> tuple[str, ...]:
        if self.scenario not in TOOL_SCENARIOS:
            return ()
        names = self.tool_names or ("fake_tool",)
        if self.scenario == "parallel_tool_calls":
            return (names[0], names[1] if len(names) > 1 else names[0])
        return names[:1]


@dataclass(frozen=True, slots=True)
class StreamPlan:
    head: bytes  # sent with the status line
    pieces: list[bytes]  # content, sent at first_token + i * interval
    tail: bytes  # sent right after the last piece
    error: bytes  # replaces the rest of the stream for a mid-stream fault
    first_prefix: bytes = b""  # glued to the first piece (OpenAI's role chunk)


class Dialect(Protocol):
    path: str
    id_prefix: str

    def validate(self, payload: dict[str, Any], headers: dict[str, str]) -> str | None:
        """A reason to reject the request the way the real API would, or None."""
        ...

    def error_body(self, status: int, message: str, code: str | None) -> bytes: ...

    def resolve_scenario(self, behavior: Behavior, payload: dict[str, Any]) -> Scenario: ...

    def tool_names(self, payload: dict[str, Any]) -> tuple[str, ...]: ...

    def include_usage(self, payload: dict[str, Any]) -> bool: ...

    def stream_plan(self, behavior: Behavior, completion: Completion) -> StreamPlan: ...

    def response_body(self, behavior: Behavior, completion: Completion) -> dict[str, Any]: ...


def resolve(behavior: Behavior, *, after_tool_result: bool, has_tools: bool) -> Scenario:
    """``auto`` answers a tool result with text, calls a tool when tools are offered,
    and otherwise streams text."""
    if behavior.scenario != "auto":
        return behavior.scenario
    if after_tool_result or not has_tools:
        return "text"
    return "tool_call"
