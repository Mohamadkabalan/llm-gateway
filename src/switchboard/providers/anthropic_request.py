"""OpenAI Chat Completions request → Anthropic Messages request (PLAN.md §3.3).

A pure, deterministic function: the same OpenAI history always produces the same
Anthropic request, byte for byte. The thinking sidecar depends on that, because
Anthropic checks that a conversation's prefix is unchanged before it accepts thinking
blocks passed back (ADR-0009). Wire formats were checked against Anthropic's docs on
2026-11-27 (API version 2026-10-21); the details are in docs/PROVIDERS.md.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import orjson

from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.routing.table import ModelTarget
from switchboard.schemas.chat import ChatMessage, ChatRequest

# API versions from this date on reject temperature, top_p and top_k with a 400.
SAMPLING_REMOVED_IN = "2026-10-21"
FALLBACK_MAX_TOKENS = 4096  # Anthropic requires max_tokens; used if nothing else sets it
IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_DATA_URL = re.compile(r"^data:(?P<media>[\w/+.-]+);base64,(?P<data>.*)$", re.DOTALL)
_EFFORT = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}
# Fields this module translates, so they are never reported as dropped.
_TRANSLATED = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "logprobs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "reasoning_effort",
        "stop",
        "user",
        "provider_options",
    }
)
_SAMPLING = frozenset({"temperature", "top_p"})

Block = dict[str, Any]
Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Passback:
    """Stored content for one assistant turn, swapped in for the translated blocks."""

    message_index: int  # index into the OpenAI ``messages`` list
    content: list[Block]


@dataclass(slots=True)
class Translation:
    body: dict[str, Any]
    dropped: list[str] = field(default_factory=list)


def build_request(
    request: ChatRequest,
    target: ModelTarget,
    *,
    api_version: str,
    unsupported_params: Literal["drop", "error"],
    passback: Passback | None = None,
) -> Translation:
    system, messages = _messages(request.messages, passback)
    if not messages:
        raise _invalid("The conversation has no user or assistant messages.", "messages")
    if messages[-1]["role"] == "assistant":
        raise _invalid(
            "The conversation ends with an assistant message (a prefill); current Claude "
            "models reject that. End with a user or tool message.",
            "messages",
        )

    body: dict[str, Any] = {"model": target.upstream_model}
    body["max_tokens"] = (
        request.output_limit
        or target.defaults.max_tokens
        or target.capabilities.max_output_tokens
        or FALLBACK_MAX_TOKENS
    )
    if system:
        body["system"] = system
    body["messages"] = messages
    if request.tools:
        body["tools"] = [_tool(tool, i) for i, tool in enumerate(request.tools)]
        if (choice := _tool_choice(request)) is not None:
            body["tool_choice"] = choice
    if stop := request.stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if request.user:
        body["metadata"] = {"user_id": request.user}

    output_config: dict[str, Any] = {}
    if request.reasoning_effort is not None:
        effort = _EFFORT.get(request.reasoning_effort)
        if effort is None:
            raise _invalid(
                f"Unknown reasoning_effort '{request.reasoning_effort}'.", "reasoning_effort"
            )
        output_config["effort"] = effort
    if (format_ := _output_format(request.response_format)) is not None:
        output_config["format"] = format_
    if output_config:
        body["output_config"] = output_config

    translation = Translation(body)
    _sampling_and_leftovers(request, translation, api_version, unsupported_params)
    _merge_provider_options(body, request)
    body["stream"] = True
    return translation


# --- messages -------------------------------------------------------------------------


def _messages(
    source: list[ChatMessage], passback: Passback | None
) -> tuple[list[Block], list[Message]]:
    """Leading system/developer messages become ``system``; the rest become turns.

    A later system message stays in place as ``role: system`` when Anthropic allows it
    there (right after a user turn, and followed by an assistant turn or the end), since
    moving it would invalidate earlier thinking. Anywhere else it becomes a ``[system]``
    text block in the user turn.
    """
    system: list[Block] = []
    index = 0
    while index < len(source) and source[index].role in ("system", "developer"):
        system += _text_blocks(source[index].content, f"messages[{index}].content")
        index += 1

    messages: list[Message] = []
    for i in range(index, len(source)):
        message = source[i]
        where = f"messages[{i}]"
        if message.role in ("system", "developer"):
            blocks = _text_blocks(message.content, f"{where}.content")
            previous = messages[-1]["role"] if messages else None
            next_role = source[i + 1].role if i + 1 < len(source) else None
            if previous == "system":
                _append(messages, "system", blocks)  # consecutive ones form one section
            elif previous == "user" and next_role in (None, "assistant", "system", "developer"):
                _append(messages, "system", blocks)
            else:
                text = "\n".join(b["text"] for b in blocks)
                _append(messages, "user", [{"type": "text", "text": f"[system] {text}"}])
        elif message.role == "user":
            _append(messages, "user", _user_blocks(message, where))
        elif message.role == "assistant":
            if passback is not None and passback.message_index == i:
                blocks = list(passback.content)
            else:
                blocks = _assistant_blocks(message, where)
            _append(messages, "assistant", blocks)
        elif message.role == "tool":
            _append(messages, "user", [_tool_result(message, where)])
        else:  # "function": the pre-tools API
            raise _invalid("Messages with role 'function' are not supported; use 'tool'.", where)
    return system, messages


def _append(messages: list[Message], role: str, blocks: list[Block]) -> None:
    """Merge consecutive same-role turns (the API would merge them anyway)."""
    if not blocks:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def _text_blocks(content: object, where: str) -> list[Block]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[Block] = []
    for j, part in enumerate(content if isinstance(content, list) else []):
        if part.get("type") in ("text", "refusal"):
            text = part.get("text") or part.get("refusal") or ""
            if text:
                blocks.append({"type": "text", "text": text})
        else:
            raise _invalid(
                f"Content part type '{part.get('type')}' is not allowed here.", f"{where}[{j}]"
            )
    return blocks


def _user_blocks(message: ChatMessage, where: str) -> list[Block]:
    content = message.content
    if not isinstance(content, list):
        return _text_blocks(content, f"{where}.content")
    blocks: list[Block] = []
    for j, part in enumerate(content):
        kind = part.get("type")
        if kind == "text":
            if text := part.get("text"):
                blocks.append({"type": "text", "text": text})
        elif kind == "image_url":
            blocks.append(_image(part, f"{where}.content[{j}]"))
        else:
            raise _invalid(
                f"Content part type '{kind}' is not supported for Anthropic models.",
                f"{where}.content[{j}]",
            )
    return blocks


def _image(part: dict[str, Any], where: str) -> Block:
    url = str((part.get("image_url") or {}).get("url") or "")
    if match := _DATA_URL.match(url):
        media = match["media"].lower()
        if media not in IMAGE_MEDIA_TYPES:
            raise _invalid(f"Image type '{media}' is not supported (jpeg, png, gif, webp).", where)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media, "data": match["data"]},
        }
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise _invalid("Images must be a data: URL (base64) or an http(s) URL.", where)


def _assistant_blocks(message: ChatMessage, where: str) -> list[Block]:
    blocks = _text_blocks(message.content, f"{where}.content")
    for j, call in enumerate(message.tool_calls or ()):
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            parsed = orjson.loads(arguments)
        except orjson.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            raise _invalid(
                "Tool call arguments must be a JSON object.",
                f"{where}.tool_calls[{j}].function.arguments",
            )
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id"),
                "name": function.get("name"),
                "input": parsed,
            }
        )
    return blocks


def _tool_result(message: ChatMessage, where: str) -> Block:
    if not message.tool_call_id:
        raise _invalid("Tool messages need a tool_call_id.", f"{where}.tool_call_id")
    content = message.content
    text = (
        content
        if isinstance(content, str)
        else "".join(b["text"] for b in _text_blocks(content, f"{where}.content"))
    )
    return {"type": "tool_result", "tool_use_id": message.tool_call_id, "content": text}


# --- tools and output options ---------------------------------------------------------


def _tool(tool: dict[str, Any], index: int) -> Block:
    if tool.get("type", "function") != "function":
        raise _invalid(f"Tool type '{tool.get('type')}' is not supported.", f"tools[{index}].type")
    function = tool.get("function") or {}
    if not function.get("name"):
        raise _invalid("Tools need a function name.", f"tools[{index}].function.name")
    result: Block = {
        "name": function["name"],
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }
    if function.get("description"):
        result["description"] = function["description"]
    if function.get("strict") is not None:
        result["strict"] = bool(function["strict"])
    return result


def _tool_choice(request: ChatRequest) -> Block | None:
    choice = request.tool_choice
    no_parallel = request.parallel_tool_calls is False
    if choice == "none":
        return {"type": "none"}
    if choice in (None, "auto"):
        result: Block = {"type": "auto"}
    elif choice == "required":
        result = {"type": "any"}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        if not name:
            raise _invalid("A named tool_choice needs function.name.", "tool_choice")
        result = {"type": "tool", "name": name}
    else:
        raise _invalid(
            f"tool_choice {choice!r} is not supported for Anthropic models.", "tool_choice"
        )
    if no_parallel:
        result["disable_parallel_tool_use"] = True
    elif choice is None:
        return None  # the API default is already auto
    return result


def _output_format(response_format: dict[str, Any] | None) -> Block | None:
    kind = (response_format or {}).get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_schema":
        schema = ((response_format or {}).get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise _invalid(
                "response_format.json_schema.schema must be an object.", "response_format"
            )
        return {"type": "json_schema", "schema": schema}
    raise _invalid(
        f"response_format '{kind}' is not supported for Anthropic models; use json_schema.",
        "response_format",
    )


def _sampling_and_leftovers(
    request: ChatRequest,
    translation: Translation,
    api_version: str,
    policy: Literal["drop", "error"],
) -> None:
    """Sampling parameters are sent only to API versions that still accept them.
    Everything else the client sent that has no Anthropic equivalent is dropped and
    reported (``unsupported_params: drop``) or rejected (``error``)."""
    sent = request.model_fields_set | set(request.model_extra or {})
    sampling_supported = api_version < SAMPLING_REMOVED_IN
    for name in sorted(sent - _TRANSLATED):
        value = getattr(request, name, None)
        if value is None:
            continue
        if name in _SAMPLING and sampling_supported:
            translation.body[name] = min(float(value), 1.0)  # Anthropic's range is 0-1
            continue
        if policy == "error":
            raise ProviderError(
                ErrorKind.INVALID_REQUEST,
                f"'{name}' is not supported for this model (Anthropic API {api_version}).",
                param=name,
                code="unsupported_parameter",
            )
        translation.dropped.append(name)


def _merge_provider_options(body: dict[str, Any], request: ChatRequest) -> None:
    """``provider_options.anthropic`` is merged in last: an escape hatch for fields the
    translation does not cover (``thinking.display``, ``cache_control``, ...)."""
    options = ((request.model_extra or {}).get("provider_options") or {}).get("anthropic")
    if not isinstance(options, dict):
        return
    for key, value in options.items():
        if isinstance(value, dict) and isinstance(body.get(key), dict):
            body[key] = {**body[key], **value}
        else:
            body[key] = value


def _invalid(message: str, param: str) -> ProviderError:
    return ProviderError(ErrorKind.INVALID_REQUEST, message, param=param, code="invalid_request")
