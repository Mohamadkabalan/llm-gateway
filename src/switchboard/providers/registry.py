"""Builds one adapter per enabled provider from the routing config."""

import os
from collections.abc import Mapping

from switchboard.providers.anthropic import AnthropicAdapter
from switchboard.providers.base import ProviderAdapter
from switchboard.providers.ollama import OllamaAdapter
from switchboard.providers.openai import OpenAIAdapter
from switchboard.providers.thinking import MemoryThinkingSidecar, ThinkingSidecar
from switchboard.routing.config import ProviderType
from switchboard.routing.loader import LoadedConfig

_OPENAI_COMPATIBLE: dict[ProviderType, type[OpenAIAdapter]] = {
    ProviderType.OPENAI: OpenAIAdapter,
    ProviderType.OLLAMA: OllamaAdapter,
}


def build_adapters(
    loaded: LoadedConfig,
    env: Mapping[str, str] | None = None,
    *,
    sidecar: ThinkingSidecar | None = None,
) -> dict[str, ProviderAdapter]:
    """One adapter per enabled provider. Must run inside the event loop (lifespan)."""
    env = os.environ if env is None else env
    sidecar = sidecar or MemoryThinkingSidecar()
    adapters: dict[str, ProviderAdapter] = {}
    for name, provider in loaded.config.providers.items():
        if name in loaded.disabled_providers:
            continue
        api_key = env.get(provider.api_key_env) if provider.api_key_env else None
        if provider.type is ProviderType.ANTHROPIC:
            adapters[name] = AnthropicAdapter(name, provider, api_key=api_key, sidecar=sidecar)
        else:
            adapters[name] = _OPENAI_COMPATIBLE[provider.type](name, provider, api_key=api_key)
    return adapters
