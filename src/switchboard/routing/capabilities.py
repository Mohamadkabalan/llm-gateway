"""What a request needs from a model, checked against the model's declared capabilities.

The router skips candidates that lack something a request needs, so a fallback that can
serve it does; when no candidate is left, the client gets a 400 naming the gap.
"""

import math
from collections.abc import Callable

import orjson

from switchboard.routing.config import Capabilities
from switchboard.schemas.chat import ChatRequest

BYTES_PER_TOKEN = 4  # a cheap estimate for routing decisions, not for billing


def estimate_prompt_tokens(chat: ChatRequest) -> int:
    """``ceil(utf8_bytes / 4)`` over messages and tools (PLAN.md §4)."""
    size = len(orjson.dumps([m.model_dump(exclude_none=True) for m in chat.messages]))
    if chat.tools:
        size += len(orjson.dumps(chat.tools))
    return math.ceil(size / BYTES_PER_TOKEN)


def missing_capabilities(
    chat: ChatRequest,
    capabilities: Capabilities,
    prompt_tokens: Callable[[], int] | None = None,
) -> list[str]:
    """The gaps, each phrased to follow "does not support". ``prompt_tokens`` supplies a
    cached size estimate; it is only called for models with ``max_input_tokens``."""
    missing: list[str] = []
    if chat.tools and not capabilities.tools:
        missing.append("tools")
    if forces_tool_use(chat.tool_choice) and not capabilities.forced_tool_choice:
        missing.append("a forced tool_choice ('required' or a named function)")
    if not capabilities.vision and chat.has_images():
        missing.append("image input")
    format_type = (chat.response_format or {}).get("type")
    if format_type == "json_schema" and not capabilities.json_schema:
        missing.append("response_format json_schema")
    if format_type == "json_object" and not capabilities.json_object:
        missing.append("response_format json_object (use json_schema instead)")
    limit = chat.output_limit
    if limit and capabilities.max_output_tokens and limit > capabilities.max_output_tokens:
        missing.append(f"more than {capabilities.max_output_tokens} output tokens")
    if capabilities.max_input_tokens:
        estimate = prompt_tokens() if prompt_tokens else estimate_prompt_tokens(chat)
        if estimate > capabilities.max_input_tokens:
            missing.append(
                f"a prompt of about {estimate} tokens (limit {capabilities.max_input_tokens})"
            )
    return missing


def forces_tool_use(tool_choice: object) -> bool:
    return tool_choice == "required" or isinstance(tool_choice, dict)
