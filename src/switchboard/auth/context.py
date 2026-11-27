"""What an authenticated request knows about its key and team."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Limits:
    rpm: int | None = None
    tpm: int | None = None
    hard_usd: Decimal | None = None  # monthly; requests are blocked past it
    soft_usd: Decimal | None = None  # monthly; one alert per period

    @property
    def has_budget(self) -> bool:
        return self.hard_usd is not None or self.soft_usd is not None


@dataclass(frozen=True, slots=True)
class KeyContext:
    key_id: UUID
    key_name: str
    key_prefix: str
    team_id: UUID
    team_name: str
    key_limits: Limits = Limits()
    team_limits: Limits = Limits()
    allowed_models: frozenset[str] | None = None  # the key's and the team's lists combined
    tags: frozenset[str] = frozenset()
    expires_at: datetime | None = None
    revoked: bool = False

    def may_use(self, model: str) -> bool:
        return self.allowed_models is None or model in self.allowed_models

    @property
    def limited(self) -> bool:
        """True if admission has anything to check for this key."""
        return any(
            limits.rpm or limits.tpm or limits.has_budget
            for limits in (self.key_limits, self.team_limits)
        )


def combine_allowed(key: list[str] | None, team: list[str] | None) -> frozenset[str] | None:
    """A model must be allowed by both the key and its team; NULL means "all"."""
    if key is None and team is None:
        return None
    if key is None:
        return frozenset(team or ())
    if team is None:
        return frozenset(key)
    return frozenset(key) & frozenset(team)
