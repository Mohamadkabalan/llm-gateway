"""Schema for ``config/routing.yaml``.

Every model forbids unknown keys, so a typo is a startup error rather than a
silently ignored setting. Durations accept ``250ms``, ``2s``, ``5m``, ``1h`` or a plain
number of seconds. Cross-references (alias → model → provider) are checked in
``switchboard.routing.loader``, which can report YAML line numbers.

Later phases extend this schema with caching and guardrails (PLAN.md §4). The full
reference is ``docs/ROUTING.md``.
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


def merge_timeouts(base: Timeouts, *overrides: TimeoutOverrides | None) -> Timeouts:
    """Apply overrides in order (provider, then model). Raises ``ValidationError`` if
    the merged result is inconsistent."""
    return merge_policy(base, *overrides)


def merge_policy[P: BaseModel](base: P, *overrides: BaseModel | None) -> P:
    """``base`` with the set fields of each override applied in order, re-validated."""
    values = base.model_dump()
    for override in overrides:
        if override is not None:
            values.update(override.model_dump(exclude_none=True))
    return type(base).model_validate(values)


class RetryPolicy(StrictModel):
    """Retries of the same target (PLAN.md §5). ``max_attempts`` counts the first try."""

    max_attempts: int = Field(default=2, ge=1, le=10)
    base_delay: Duration = 0.25
    max_delay: Duration = 4.0
    honor_retry_after_up_to: Duration = 10.0
    budget_ratio: float = Field(
        default=0.2, ge=0, le=1, description="Retries per provider, as a share of its requests."
    )
    budget_min_retries: int = Field(
        default=10,
        ge=0,
        description="Retries per provider always allowed per budget window, so a quiet "
        "provider can still be retried.",
    )

    @model_validator(mode="after")
    def _delays_ordered(self) -> Self:
        if self.max_delay < self.base_delay:
            raise ValueError("'max_delay' must be >= 'base_delay'")
        return self


class RetryOverrides(StrictModel):
    max_attempts: int | None = Field(default=None, ge=1, le=10)
    base_delay: Duration | None = None
    max_delay: Duration | None = None
    honor_retry_after_up_to: Duration | None = None
    budget_ratio: float | None = Field(default=None, ge=0, le=1)
    budget_min_retries: int | None = Field(default=None, ge=0)


BREAKER_SLOTS = 10  # the rolling window is counted in this many equal buckets


class BreakerPolicy(StrictModel):
    """One circuit breaker per ``provider/model`` (PLAN.md §5). The window slides in
    steps of a tenth of its length."""

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
        if self.window < 1:
            raise ValueError("'window' must be at least 1s")
        return self

    @property
    def bucket_seconds(self) -> float:
        return self.window / BREAKER_SLOTS


class BreakerOverrides(StrictModel):
    window: Duration | None = None
    min_requests: int | None = Field(default=None, ge=1)
    failure_rate: float | None = Field(default=None, gt=0, le=1)
    cooldown: Duration | None = None
    max_cooldown: Duration | None = None
    half_open_probes: int | None = Field(default=None, ge=1)


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
    """Upstream connection pool, per provider and per gateway replica.

    Every upstream call streams over its own HTTP/1.1 connection, so
    ``max_connections`` caps concurrent requests to this provider. Requests beyond it
    wait for a free connection for up to ``timeouts.connect``.
    """

    max_connections: int = Field(default=1000, ge=1)
    keepalive_expiry: Duration = Field(
        default=30.0, description="Close idle connections after this long."
    )


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
    retry: RetryOverrides | None = None
    breaker: BreakerOverrides | None = None
    pool: PoolConfig = PoolConfig()
    thinking_passback: bool = Field(
        default=True,
        description="Anthropic only: carry thinking blocks across OpenAI-format tool loops "
        "(PLAN.md §3.4).",
    )


class ModelKind(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class Capabilities(StrictModel):
    """What a model accepts. A candidate that cannot serve a request is skipped, so a
    fallback that can serves it; with none left the client gets a 400 naming the gap."""

    tools: bool = True
    vision: bool = False
    forced_tool_choice: bool = True
    json_schema: bool = False
    json_object: bool = True
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


class HedgePolicy(StrictModel):
    """Start the next candidate in parallel if the first has not produced a first token
    after ``after``. The first to commit wins; the other is cancelled."""

    after: Duration
    max_ratio: float = Field(
        default=0.1, gt=0, le=1, description="At most this share of the route's requests hedge."
    )


class RouteSpec(StrictModel):
    """Where requests go: an alias, or the inline ``use`` of a rule."""

    targets: list[Target] = Field(min_length=1)
    sticky: Sticky = Sticky.NONE
    fallbacks: list[str] = Field(default_factory=list)
    hedge: HedgePolicy | None = None
    fallback_on_refusal: bool = False


class AliasConfig(RouteSpec):
    description: str | None = None


class TokenBounds(StrictModel):
    lt: int | None = Field(default=None, ge=0)
    lte: int | None = Field(default=None, ge=0)
    gt: int | None = Field(default=None, ge=0)
    gte: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _some_bound(self) -> Self:
        if all(v is None for v in (self.lt, self.lte, self.gt, self.gte)):
            raise ValueError("set at least one of 'lt', 'lte', 'gt', 'gte'")
        return self

    def contains(self, value: int) -> bool:
        return (
            (self.lt is None or value < self.lt)
            and (self.lte is None or value <= self.lte)
            and (self.gt is None or value > self.gt)
            and (self.gte is None or value >= self.gte)
        )


class RuleMatch(StrictModel):
    """Every condition that is set must hold. An empty ``when`` matches every request."""

    model: str | None = Field(default=None, description="The requested alias or model.")
    key: str | None = None
    team: str | None = None
    tags_any: list[str] | None = Field(default=None, min_length=1)
    tags_all: list[str] | None = Field(default=None, min_length=1)
    prompt_tokens: TokenBounds | None = None
    has_tools: bool | None = None
    has_images: bool | None = None
    stream: bool | None = None


class RuleAction(StrictModel):
    """Either ``alias`` or an inline route (``targets`` and optionally ``fallbacks``,
    ``sticky``, ``hedge``, ``fallback_on_refusal``)."""

    alias: str | None = None
    targets: list[Target] | None = Field(default=None, min_length=1)
    sticky: Sticky | None = None
    fallbacks: list[str] | None = None
    hedge: HedgePolicy | None = None
    fallback_on_refusal: bool | None = None

    @model_validator(mode="after")
    def _alias_xor_targets(self) -> Self:
        inline = [
            name
            for name in ("targets", "sticky", "fallbacks", "hedge", "fallback_on_refusal")
            if getattr(self, name) is not None
        ]
        if self.alias is not None and inline:
            raise ValueError(f"use either 'alias' or an inline route, not both ({inline[0]})")
        if self.alias is None and self.targets is None:
            raise ValueError("set 'alias' or 'targets'")
        return self

    def route(self) -> RouteSpec | None:
        """The inline route, or None when the action names an alias."""
        if self.targets is None:
            return None
        return RouteSpec(
            targets=self.targets,
            sticky=self.sticky or Sticky.NONE,
            fallbacks=self.fallbacks or [],
            hedge=self.hedge,
            fallback_on_refusal=bool(self.fallback_on_refusal),
        )


class Rule(StrictModel):
    name: Name
    when: RuleMatch = RuleMatch()
    use: RuleAction


class RoutingConfig(StrictModel):
    version: Literal[1]
    pricing: str | None = Field(
        default=None,
        description="The active pricing file, relative to this file "
        "(e.g. pricing/2026-11-27.yaml). Without one, costs are recorded as 0.",
    )
    defaults: Defaults = Defaults()
    providers: dict[Name, ProviderConfig] = Field(min_length=1)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    aliases: dict[Name, AliasConfig] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list)
