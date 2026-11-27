"""The routing table: turns the requested ``model`` into a ``RoutePlan``.

This is the Phase 1 static router. An alias resolves to its first target, and a
concrete ``provider/model`` resolves to itself. Weighted A/B selection, rules,
fallback chains and capability filtering replace ``resolve`` in Phase 3, behind the
same interface.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from switchboard.routing.config import (
    Capabilities,
    ModelConfig,
    ModelDefaults,
    ModelKind,
    ProviderConfig,
    Timeouts,
    merge_timeouts,
)
from switchboard.routing.loader import LoadedConfig

DIRECT_ROUTE = "direct"  # the ``alias`` metric label when a client names a concrete model


@dataclass(frozen=True, slots=True)
class ModelTarget:
    name: str  # "provider/model", as in the config
    provider: str
    upstream_model: str
    kind: ModelKind
    capabilities: Capabilities
    defaults: ModelDefaults
    timeouts: Timeouts  # defaults, then provider overrides, then model overrides


@dataclass(frozen=True, slots=True)
class RoutePlan:
    requested: str
    alias: str | None
    targets: tuple[ModelTarget, ...]  # primary first; Phase 1 always has exactly one

    @property
    def route_label(self) -> str:
        return self.alias or DIRECT_ROUTE


class RouteError(Exception):
    def __init__(
        self, status: int, message: str, *, code: str, error_type: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.error_type = error_type


class RoutingTable:
    def __init__(self, loaded: LoadedConfig) -> None:
        config = loaded.config
        self.version = loaded.version
        self._disabled = frozenset(loaded.disabled_providers)
        self._targets = {
            name: _target(name, model, config.providers, config.defaults.timeouts)
            for name, model in config.models.items()
        }
        self._aliases = {
            name: self._targets[alias.targets[0].model] for name, alias in config.aliases.items()
        }

    def listing(self) -> list[tuple[str, str, ModelKind]]:
        """``(id, owner, kind)`` for every alias and model a client can use now.

        Aliases are owned by the gateway; models by their provider. Anything whose
        provider is disabled is left out. Phase 4 filters this per key.
        """
        entries = [
            (name, "switchboard", target.kind)
            for name, target in self._aliases.items()
            if target.provider not in self._disabled
        ]
        entries += [
            (name, target.provider, target.kind)
            for name, target in self._targets.items()
            if target.provider not in self._disabled
        ]
        return entries

    def resolve(self, requested: str, kind: ModelKind) -> RoutePlan:
        alias: str | None = None
        if requested in self._aliases:
            alias, target = requested, self._aliases[requested]
        elif requested in self._targets:
            target = self._targets[requested]
        else:
            raise RouteError(
                404,
                f"The model '{requested}' does not exist or you do not have access to it.",
                code="model_not_found",
                error_type="invalid_request_error",
            )
        if target.kind is not kind:
            raise RouteError(
                400,
                f"The model '{requested}' is a {target.kind.value} model and cannot be "
                f"used for {kind.value}.",
                code="model_kind_mismatch",
            )
        if target.provider in self._disabled:
            raise RouteError(
                503,
                f"The model '{requested}' is unavailable: its provider is not configured "
                "on this gateway.",
                code="provider_unavailable",
                error_type="server_error",
            )
        return RoutePlan(requested=requested, alias=alias, targets=(target,))


def _target(
    name: str, model: ModelConfig, providers: Mapping[str, ProviderConfig], base: Timeouts
) -> ModelTarget:
    provider_name = name.partition("/")[0]
    provider = providers[provider_name]
    return ModelTarget(
        name=name,
        provider=provider_name,
        upstream_model=model.upstream_model,
        kind=model.kind,
        capabilities=model.capabilities,
        defaults=model.defaults,
        timeouts=merge_timeouts(base, provider.timeouts, model.timeouts),
    )
