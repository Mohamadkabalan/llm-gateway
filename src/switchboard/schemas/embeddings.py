"""The OpenAI Embeddings request, passed through to OpenAI-compatible upstreams."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EmbeddingInput = str | list[str] | list[int] | list[list[int]]


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1, max_length=256)
    input: EmbeddingInput
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None

    @property
    def is_tokens(self) -> bool:
        """True when ``input`` is token ids rather than text."""
        value = self.input
        return isinstance(value, list) and bool(value) and not isinstance(value[0], str)

    def text_length(self) -> int:
        """Input size in characters, for routing's prompt-size estimate (a token id
        counts as four characters, so the estimate is the token count)."""
        value = self.input
        if isinstance(value, str):
            return len(value)
        total = 0
        for item in value:
            if isinstance(item, str):
                total += len(item)
            elif isinstance(item, int):
                total += 4
            else:
                total += 4 * len(item)
        return total

    def to_upstream_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)
