"""What a request needs from a model, checked against the model's declared capabilities.

Phase 2 rejects a request whose target lacks something it needs, with a 400 that names
the gap. Phase 3 uses the same check to skip such candidates, so a fallback model that
can serve the request does.
"""

import math

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


def missing_capabilities(chat: ChatRequest, capabilities: Capabilities) -> list[str]:
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
        estimate = estimate_prompt_tokens(chat)
        if estimate > capabilities.max_input_tokens:
            missing.append(
                f"a prompt of about {estimate} tokens (limit {capabilities.max_input_tokens})"
            )
    return missing


def forces_tool_use(tool_choice: object) -> bool:
    return tool_choice == "required" or isinstance(tool_choice, dict)
