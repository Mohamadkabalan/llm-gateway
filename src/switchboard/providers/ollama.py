"""Ollama adapter: its OpenAI-compatible ``/v1`` API, with Ollama's limits (PLAN.md §3.6).

Per Ollama's compatibility docs (checked 2026-11-27), ``/v1/chat/completions`` accepts
``max_tokens`` (``max_completion_tokens`` is not listed), streams usage with
``stream_options.include_usage``, and does not support ``tool_choice``, ``n``, ``user``,
logprobs or image URLs (images must be base64 data URLs). The provider's ``base_url`` is
the server root, e.g. ``http://ollama:11434``.
"""

from typing import Any

from switchboard.providers.base import CallContext
from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.providers.openai import OpenAIAdapter, apply_provider_options
from switchboard.routing.table import ModelTarget
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.embeddings import EmbeddingRequest


class OllamaAdapter(OpenAIAdapter):
    base_path = "/v1"
    options_key = "ollama"

    def prepare_chat(
        self, request: ChatRequest, target: ModelTarget, ctx: CallContext
    ) -> dict[str, Any]:
        body = request.to_upstream_dict()
        apply_provider_options(body, self.options_key)
        body["model"] = target.upstream_model
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        body.pop("max_completion_tokens", None)
        body.pop("max_tokens", None)
        if (limit := request.output_limit or target.defaults.max_tokens) is not None:
            body["max_tokens"] = limit

        choice = body.pop("tool_choice", None)
        if choice == "none":
            body.pop("tools", None)  # the same meaning, expressed the way Ollama supports
        elif choice not in (None, "auto"):
            # Forced tool use is screened by the capability check; this is the backstop.
            raise ProviderError(
                ErrorKind.INVALID_REQUEST,
                "Ollama does not support forcing a tool call (tool_choice).",
                param="tool_choice",
                code="unsupported_parameter",
            )
        for name in ("user", "parallel_tool_calls"):
            if body.pop(name, None) is not None:
                ctx.dropped_params.append(name)
        _require_inline_images(request)
        return body

    def prepare_embedding(
        self, request: EmbeddingRequest, target: ModelTarget, ctx: CallContext
    ) -> dict[str, Any]:
        if request.is_tokens:
            raise ProviderError(
                ErrorKind.INVALID_REQUEST,
                "Ollama embeddings accept text input only, not token ids.",
                param="input",
                code="unsupported_parameter",
            )
        body = super().prepare_embedding(request, target, ctx)
        if body.pop("user", None) is not None:
            ctx.dropped_params.append("user")
        return body


def _require_inline_images(request: ChatRequest) -> None:
    for message in request.messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            url = (
                (part.get("image_url") or {}).get("url", "")
                if part.get("type") == "image_url"
                else ""
            )
            if url and not url.startswith("data:"):
                raise ProviderError(
                    ErrorKind.INVALID_REQUEST,
                    "Ollama accepts images only as base64 data URLs, not links.",
                    param="messages",
                    code="unsupported_parameter",
                )
