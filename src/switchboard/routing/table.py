"""The routing table: turns a request into a ``RoutePlan`` (PLAN.md §2 step 6, §4).

For each request the table:

1. checks that the requested ``model`` is a known alias or ``provider/model``;
2. applies the first matching rule, which can send the request to another alias or to
   an inline route;
3. picks the primary target, weighted at random or by a stable hash when the route is
   sticky;
4. appends the route's fallbacks (the primary is not repeated);
5. drops candidates whose provider is disabled or that cannot serve the request
   (tools, images, forced ``tool_choice``, structured output, context length).

The table is immutable and built once per config version; hot reload swaps in a new
one, and requests in flight keep the table they started with.
"""

import bisect
import hashlib
import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from switchboard.routing.capabilities import estimate_prompt_tokens, missing_capabilities
from switchboard.routing.config import (
    BreakerPolicy,
    Capabilities,
    HedgePolicy,
    ModelConfig,
    ModelDefaults,
    ModelKind,
    ProviderConfig,
    RetryPolicy,
    RouteSpec,
    RoutingConfig,
    Rule,
    Sticky,
    Timeouts,
    merge_policy,
    merge_timeouts,
)
from switchboard.routing.loader import LoadedConfig
from switchboard.schemas.chat import ChatRequest

DIRECT_ROUTE = "direct"  # the ``alias`` metric label when a client names a concrete model
_HASH_SPACE = 2**64
_DEFAULT_RETRY = RetryPolicy()  # frozen, so one shared instance is safe
_DEFAULT_BREAKER = BreakerPolicy()


@dataclass(frozen=True, slots=True)
class ModelTarget:
    name: str  # "provider/model", as in the config
    provider: str
    upstream_model: str
    kind: ModelKind
    capabilities: Capabilities
    defaults: ModelDefaults
    timeouts: Timeouts  # defaults, then provider overrides, then model overrides
    retry: RetryPolicy = _DEFAULT_RETRY  # defaults, then provider overrides
    breaker: BreakerPolicy = _DEFAULT_BREAKER  # defaults, then provider overrides


@dataclass(frozen=True, slots=True)
class Route:
    """A compiled alias or inline rule route."""

    name: str | None  # the alias, or None for a concrete model or an inline rule route
    targets: tuple[ModelTarget, ...]
    cumulative_weights: tuple[int, ...]
    sticky: Sticky
    fallbacks: tuple[ModelTarget, ...]
    hedge: HedgePolicy | None
    fallback_on_refusal: bool

    @property
    def kind(self) -> ModelKind:
        return self.targets[0].kind


@dataclass(frozen=True, slots=True)
class Skipped:
    """A candidate left out of the plan, and why."""

    model: str
    reason: str
    capability_gaps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoutePlan:
    requested: str
    alias: str | None  # the requested alias (None if a concrete model was requested)
    route: str | None  # the alias that serves it, after rules
    rule: str | None  # the rule that matched, if any
    candidates: tuple[ModelTarget, ...]  # primary first; never empty
    hedge: HedgePolicy | None = None
    fallback_on_refusal: bool = False
    keepalive_after: float = 5.0
    skipped: tuple[Skipped, ...] = ()

    @property
    def route_label(self) -> str:
        return self.alias or DIRECT_ROUTE

    @property
    def primary(self) -> ModelTarget:
        return self.candidates[0]


class RouteError(Exception):
    def __init__(
        self, status: int, message: str, *, code: str, error_type: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.error_type = error_type


@dataclass
class RouteRequest:
    """What routing looks at. Expensive facts (the prompt-size estimate) are computed
    only if a rule or capability check asks for them."""

    model: str
    kind: ModelKind
    chat: ChatRequest | None = None
    key: str | None = None  # the virtual key's name
    team: str | None = None  # the key's team's name
    tags: frozenset[str] = frozenset()
    sticky_subject: str | None = None  # the key id, for ``sticky: key``
    embedding_chars: int = 0  # embeddings: total input length, for the size estimate

    @property
    def stream(self) -> bool:
        return self.chat is not None and self.chat.stream

    @property
    def end_user(self) -> str | None:
        return self.chat.user if self.chat is not None else None

    @cached_property
    def has_tools(self) -> bool:
        return self.chat is not None and bool(self.chat.tools)

    @cached_property
    def has_images(self) -> bool:
        return self.chat is not None and self.chat.has_images()

    @cached_property
    def prompt_tokens(self) -> int:
        if self.chat is not None:
            return estimate_prompt_tokens(self.chat)
        return -(-self.embedding_chars // 4)


class RoutingTable:
    def __init__(self, loaded: LoadedConfig, *, rng: random.Random | None = None) -> None:
        config = loaded.config
        self.version = loaded.version
        self.keepalive_after = config.defaults.keepalive_after
        self._rng = rng or random.Random()  # noqa: S311 (traffic splitting, not crypto)
        self._disabled = frozenset(loaded.disabled_providers)
        self._targets = {
            name: _target(name, model, config) for name, model in config.models.items()
        }
        self._aliases = {name: self._compile(name, alias) for name, alias in config.aliases.items()}
        self._direct = {
            name: Route(name, (target,), (1,), Sticky.NONE, (), None, False)
            for name, target in self._targets.items()
        }
        self._rules = [(rule, self._compile_rule(rule)) for rule in config.rules]

    def listing(self) -> list[tuple[str, str, ModelKind]]:
        """``(id, owner, kind)`` for every alias and model a client can use now.

        Aliases are owned by the gateway; models by their provider. An alias is listed
        if any of its targets or fallbacks is enabled; ``/v1/models`` filters it per key.
        """
        entries = [
            (name, "switchboard", route.kind)
            for name, route in self._aliases.items()
            if any(t.provider not in self._disabled for t in (*route.targets, *route.fallbacks))
        ]
        entries += [
            (name, target.provider, target.kind)
            for name, target in self._targets.items()
            if target.provider not in self._disabled
        ]
        return entries

    def target(self, name: str) -> ModelTarget | None:
        return self._targets.get(name)

    def longest_deadline(self) -> float:
        """The longest ``total`` timeout of any model: how long a request may run."""
        return max((t.timeouts.total for t in self._targets.values()), default=0.0)

    def plan(self, request: RouteRequest) -> RoutePlan:
        requested = request.model
        alias = requested if requested in self._aliases else None
        if alias is None and requested not in self._targets:
            raise RouteError(
                404,
                f"The model '{requested}' does not exist or you do not have access to it.",
                code="model_not_found",
                error_type="invalid_request_error",
            )
        rule_name: str | None = None
        route = self._aliases[alias] if alias is not None else self._direct[requested]
        for rule, rule_route in self._rules:
            if _matches(rule, request):
                rule_name, route = rule.name, rule_route
                break
        if route.kind is not request.kind:
            raise RouteError(
                400,
                f"The model '{requested}' is a {route.kind.value} model and cannot be "
                f"used for {request.kind.value}.",
                code="model_kind_mismatch",
            )

        primary = self._choose(route, request)
        ordered = [primary, *(t for t in route.fallbacks if t is not primary)]
        candidates: list[ModelTarget] = []
        skipped: list[Skipped] = []
        for target in ordered:
            if target.provider in self._disabled:
                skipped.append(Skipped(target.name, "provider disabled"))
            elif request.chat is not None and (
                gaps := missing_capabilities(
                    request.chat, target.capabilities, lambda: request.prompt_tokens
                )
            ):
                skipped.append(Skipped(target.name, "missing capability", tuple(gaps)))
            else:
                candidates.append(target)
        if not candidates:
            raise _no_candidate(requested, skipped)
        return RoutePlan(
            requested=requested,
            alias=alias,
            route=route.name,
            rule=rule_name,
            candidates=tuple(candidates),
            hedge=route.hedge,
            fallback_on_refusal=route.fallback_on_refusal,
            keepalive_after=self.keepalive_after,
            skipped=tuple(skipped),
        )

    def _choose(self, route: Route, request: RouteRequest) -> ModelTarget:
        if len(route.targets) == 1:
            return route.targets[0]
        total = route.cumulative_weights[-1]
        subject = {
            Sticky.KEY: request.sticky_subject,
            Sticky.END_USER: request.end_user,
            Sticky.NONE: None,
        }[route.sticky]
        if subject is not None:
            point = sticky_point(route.name or "", subject, total)
        else:
            point = self._rng.random() * total
        return route.targets[bisect.bisect_right(route.cumulative_weights, point)]

    def _compile(self, name: str | None, spec: RouteSpec) -> Route:
        weights = tuple(itertools.accumulate(t.weight for t in spec.targets))
        return Route(
            name=name,
            targets=tuple(self._targets[t.model] for t in spec.targets),
            cumulative_weights=weights,
            sticky=spec.sticky,
            fallbacks=tuple(self._targets[m] for m in spec.fallbacks),
            hedge=spec.hedge,
            fallback_on_refusal=spec.fallback_on_refusal,
        )

    def _compile_rule(self, rule: Rule) -> Route:
        if rule.use.alias is not None:
            return self._aliases[rule.use.alias]
        spec = rule.use.route()
        assert spec is not None  # noqa: S101 (RuleAction validation guarantees one form)
        return self._compile(None, spec)


def sticky_point(route: str, subject: str, total: int) -> float:
    """A stable point in ``[0, total)`` for this subject on this route."""
    digest = hashlib.sha256(f"{route}\0{subject}".encode()).digest()
    return int.from_bytes(digest[:8]) / _HASH_SPACE * total


def _matches(rule: Rule, request: RouteRequest) -> bool:  # noqa: PLR0911 (one exit per condition)
    when = rule.when
    if when.model is not None and when.model != request.model:
        return False
    if when.key is not None and when.key != request.key:
        return False
    if when.team is not None and when.team != request.team:
        return False
    if when.tags_any is not None and request.tags.isdisjoint(when.tags_any):
        return False
    if when.tags_all is not None and not request.tags.issuperset(when.tags_all):
        return False
    if when.stream is not None and when.stream != request.stream:
        return False
    if when.has_tools is not None and when.has_tools != request.has_tools:
        return False
    if when.has_images is not None and when.has_images != request.has_images:
        return False
    return when.prompt_tokens is None or when.prompt_tokens.contains(request.prompt_tokens)


def _no_candidate(requested: str, skipped: Sequence[Skipped]) -> RouteError:
    gaps = next((s.capability_gaps for s in skipped if s.capability_gaps), None)
    if gaps:
        return RouteError(
            400,
            f"The model '{requested}' does not support {'; '.join(gaps)}.",
            code="model_capability_missing",
        )
    return RouteError(
        503,
        f"The model '{requested}' is unavailable: its provider is not configured on this gateway.",
        code="provider_unavailable",
        error_type="server_error",
    )


def _target(name: str, model: ModelConfig, config: RoutingConfig) -> ModelTarget:
    provider_name = name.partition("/")[0]
    provider: ProviderConfig = config.providers[provider_name]
    defaults = config.defaults
    return ModelTarget(
        name=name,
        provider=provider_name,
        upstream_model=model.upstream_model,
        kind=model.kind,
        capabilities=model.capabilities,
        defaults=model.defaults,
        timeouts=merge_timeouts(defaults.timeouts, provider.timeouts, model.timeouts),
        retry=merge_policy(defaults.retry, provider.retry),
        breaker=merge_policy(defaults.breaker, provider.breaker),
    )


__all__ = [
    "DIRECT_ROUTE",
    "ModelTarget",
    "Route",
    "RouteError",
    "RoutePlan",
    "RouteRequest",
    "RoutingTable",
    "Skipped",
    "sticky_point",
]
