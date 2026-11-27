"""The OpenAI Chat Completions request, which is the gateway's internal format (ADR-0002).

Only the fields the gateway reads are declared. Everything else is kept as an extra
field (``extra="allow"``) and passed through, so a new OpenAI parameter is not
rejected just because Switchboard has not heard of it yet.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "developer", "user", "assistant", "tool", "function"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    # A string, a list of content parts, or null (an assistant turn with tool calls).
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1, max_length=256)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int | None = Field(default=None, ge=1, le=128)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    logprobs: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    stop: str | list[str] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    user: str | None = None

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    @property
    def output_limit(self) -> int | None:
        """``max_completion_tokens``, or the deprecated ``max_tokens``."""
        return self.max_completion_tokens or self.max_tokens

    def has_images(self) -> bool:
        return any(
            isinstance(message.content, list)
            and any(part.get("type") == "image_url" for part in message.content)
            for message in self.messages
        )

    def to_upstream_dict(self) -> dict[str, Any]:
        """The request as the client sent it, including unknown fields."""
        return self.model_dump(mode="json", exclude_unset=True)
