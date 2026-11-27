"""Schema for ``config/routing.yaml``.

Every model forbids unknown keys, so a typo is a startup error rather than a
silently ignored setting. Durations accept ``250ms``, ``2s``, ``5m``, ``1h`` or a plain
number of seconds. Cross-references (alias → model → provider) are checked in
``switchboard.routing.loader``, which can report YAML line numbers.

Later phases extend this schema with rules, caching and guardrails (PLAN.md §4).
"""

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, HttpUrl, model_validator

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)\s*$")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
ENV_VAR_PATTERN = r"^[A-Z_][A-Z0-9_]*$"


def parse_duration(value: object) -> float:
    """Convert ``"250ms"`` / ``"2s"`` / ``"5m"`` / ``"1h"`` / a number into seconds."""
    if isinstance(value, bool):
        raise ValueError("expected a duration such as '250ms', '2s' or '5m'")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        match = _DURATION_RE.match(value)
        if match:
            return float(match.group(1)) * _UNIT_SECONDS[match.group(2)]
    raise ValueError("expected a duration such as '250ms', '2s' or '5m'")


Duration = Annotated[float, BeforeValidator(parse_duration), Field(gt=0)]
Name = Annotated[str, Field(pattern=NAME_PATTERN, max_length=64)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Timeouts(StrictModel):
    connect: Duration = 2.0
    first_token: Duration = 30.0
    idle: Duration = 30.0
    total: Duration = 600.0

    @model_validator(mode="after")
    def _total_is_longest(self) -> Self:
        if self.total < max(self.connect, self.first_token):
            raise ValueError("'total' must be at least as long as 'connect' and 'first_token'")
        return self


class TimeoutOverrides(StrictModel):
    """Per-provider or per-model overrides; unset fields inherit from ``defaults``."""

    connect: Duration | None = None
    first_token: Duration | None = None
    idle: Duration | None = None
    total: Duration | None = None


class RetryPolicy(StrictModel):
    max_attempts: int = Field(default=2, ge=1, le=10)
    base_delay: Duration = 0.25
    max_delay: Duration = 4.0
    honor_retry_after_up_to: Duration = 10.0
    budget_ratio: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def _delays_ordered(self) -> Self:
        if self.max_delay < self.base_delay:
            raise ValueError("'max_delay' must be >= 'base_delay'")
        return self


class BreakerPolicy(StrictModel):
    window: Duration = 30.0
    min_requests: int = Field(default=20, ge=1)
    failure_rate: float = Field(default=0.5, gt=0, le=1)
    cooldown: Duration = 30.0
    max_cooldown: Duration = 300.0
    half_open_probes: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _cooldowns_ordered(self) -> Self:
        if self.max_cooldown < self.cooldown:
            raise ValueError("'max_cooldown' must be >= 'cooldown'")
        return self


class Defaults(StrictModel):
    timeouts: Timeouts = Timeouts()
    retry: RetryPolicy = RetryPolicy()
    breaker: BreakerPolicy = BreakerPolicy()
    keepalive_after: Duration = 5.0


class ProviderType(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class PoolConfig(StrictModel):
    max_connections: int = Field(default=200, ge=1)
    max_keepalive: int = Field(default=100, ge=0)


class ProviderConfig(StrictModel):
    type: ProviderType
    base_url: HttpUrl
    api_key_env: Annotated[str, Field(pattern=ENV_VAR_PATTERN)] | None = Field(
        default=None, description="Name of the env var holding the key; never the key itself."
    )
    api_version: str | None = None
    unsupported_params: Literal["drop", "error"] = "drop"
    optional: bool = Field(
        default=False,
        description="If true, a missing api_key_env disables this provider with a warning "
        "instead of failing startup.",
    )
    timeouts: TimeoutOverrides | None = None
    pool: PoolConfig = PoolConfig()


class ModelKind(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class Capabilities(StrictModel):
    tools: bool = True
    vision: bool = False
    forced_tool_choice: bool = True
    json_schema: bool = False
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)


class ModelDefaults(StrictModel):
    max_tokens: int | None = Field(default=None, ge=1)


class ModelConfig(StrictModel):
    """A concrete deployment. Its key in ``models`` is ``<provider>/<name>``."""

    upstream_model: str = Field(min_length=1, max_length=200)
    kind: ModelKind = ModelKind.CHAT
    capabilities: Capabilities = Capabilities()
    defaults: ModelDefaults = ModelDefaults()
    timeouts: TimeoutOverrides | None = None


class Target(StrictModel):
    model: str
    weight: int = Field(default=1, ge=1)


class Sticky(StrEnum):
    NONE = "none"
    KEY = "key"
    END_USER = "end_user"


class AliasConfig(StrictModel):
    description: str | None = None
    targets: list[Target] = Field(min_length=1)
    sticky: Sticky = Sticky.NONE
    fallbacks: list[str] = Field(default_factory=list)


class RoutingConfig(StrictModel):
    version: Literal[1]
    defaults: Defaults = Defaults()
    providers: dict[Name, ProviderConfig] = Field(min_length=1)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    aliases: dict[Name, AliasConfig] = Field(default_factory=dict)
