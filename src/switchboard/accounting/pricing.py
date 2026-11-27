"""Versioned price tables and the cost calculator (PLAN.md §8).

A pricing file (``config/pricing/<date>.yaml``) is immutable once committed: a price
change is a new file, selected by ``pricing:`` in the routing config. Every usage event
records the version it was priced with, so past costs can be reproduced.

Prices are USD per million tokens and are read as exact decimals (a YAML ``0.10`` is
``Decimal("0.10")``, never a float). Costs are computed in ``Decimal`` from the
provider-reported usage buckets:

    cost = sum(tokens[bucket] * price[bucket] for each bucket) / 1e6

over the buckets input, cache_read, cache_write_5m, cache_write_1h and output.

``output`` includes reasoning tokens (providers bill them as output). A model with
``tiers`` is priced by the request's total prompt tokens, cache reads and writes
included, as Anthropic prices Claude Haiku 5.5. A cache bucket without a price is
charged at the input price.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

from switchboard.schemas.events import Usage

MILLION = Decimal(1_000_000)
NANO = Decimal(1_000_000_000)

PriceValue = Decimal


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rates(_Strict):
    """USD per million tokens."""

    input: PriceValue = Field(ge=0)
    output: PriceValue = Field(default=Decimal(0), ge=0)
    cache_read: PriceValue | None = Field(default=None, ge=0)
    cache_write_5m: PriceValue | None = Field(default=None, ge=0)
    cache_write_1h: PriceValue | None = Field(default=None, ge=0)


class Tier(Rates):
    max_prompt_tokens: int | None = Field(
        default=None, ge=1, description="Up to and including this many prompt tokens."
    )


class ModelPrice(_Strict):
    input: PriceValue | None = Field(default=None, ge=0)
    output: PriceValue | None = Field(default=None, ge=0)
    cache_read: PriceValue | None = Field(default=None, ge=0)
    cache_write_5m: PriceValue | None = Field(default=None, ge=0)
    cache_write_1h: PriceValue | None = Field(default=None, ge=0)
    tiers: list[Tier] | None = Field(default=None, min_length=1)
    same_as: str | None = Field(default=None, description="Priced exactly like this model.")
    note: str | None = None

    @model_validator(mode="after")
    def _one_form(self) -> Self:
        flat = self.input is not None
        forms = sum((flat, self.tiers is not None, self.same_as is not None))
        if forms != 1:
            raise ValueError("give exactly one of: flat prices (input, ...), 'tiers', 'same_as'")
        if not flat and any(
            v is not None
            for v in (self.output, self.cache_read, self.cache_write_5m, self.cache_write_1h)
        ):
            raise ValueError("per-token prices go inside 'tiers' when tiers are used")
        if self.tiers is not None:
            bounds = [t.max_prompt_tokens for t in self.tiers]
            if bounds[-1] is not None or any(b is None for b in bounds[:-1]):
                raise ValueError("every tier but the last needs 'max_prompt_tokens'")
            numeric = [b for b in bounds[:-1] if b is not None]
            if numeric != sorted(set(numeric)):
                raise ValueError("tier bounds must increase")
        return self


class Source(_Strict):
    url: HttpUrl
    retrieved: date


class PricingFile(_Strict):
    version: str = Field(min_length=1)
    currency: Literal["USD"]
    sources: dict[str, Source] = Field(default_factory=dict)
    models: dict[str, ModelPrice]


@dataclass(frozen=True, slots=True)
class Cost:
    usd: Decimal
    estimated: bool = False

    @property
    def nano(self) -> int:
        """Whole nano-USD, rounded up so budgets never under-count."""
        return int((self.usd * NANO).to_integral_value(rounding=ROUND_CEILING))


class PriceTable:
    def __init__(self, pricing: PricingFile, source: str) -> None:
        self.version = pricing.version
        self.source = source
        self._tiers: dict[str, tuple[Tier, ...]] = {}
        for name in pricing.models:
            self._tiers[name] = _resolve(name, pricing.models, set())

    def __contains__(self, model: str) -> bool:
        return model in self._tiers

    def models(self) -> list[str]:
        return list(self._tiers)

    def rates(self, model: str, prompt_tokens: int) -> Rates | None:
        tiers = self._tiers.get(model)
        if tiers is None:
            return None
        for tier in tiers:
            if tier.max_prompt_tokens is None or prompt_tokens <= tier.max_prompt_tokens:
                return tier
        return tiers[-1]

    def cost(self, model: str, usage: Usage) -> Decimal | None:
        """The cost of provider-reported usage, or None if the model has no price."""
        rates = self.rates(model, usage.prompt_total)
        if rates is None:
            return None
        total = (
            usage.input * rates.input
            + usage.cache_read * _or(rates.cache_read, rates.input)
            + usage.cache_write_5m * _or(rates.cache_write_5m, rates.input)
            + usage.cache_write_1h * _or(rates.cache_write_1h, rates.input)
            + usage.output * rates.output
        )
        return total / MILLION

    def estimate(self, model: str, prompt_tokens: int, output_tokens: int) -> Decimal | None:
        """A pre-request estimate: the whole prompt at the input price."""
        return self.cost(model, Usage(input=prompt_tokens, output=output_tokens))


def _or(value: Decimal | None, fallback: Decimal) -> Decimal:
    return fallback if value is None else value


def _resolve(name: str, models: Mapping[str, ModelPrice], seen: set[str]) -> tuple[Tier, ...]:
    if name in seen:
        raise ValueError(f"'same_as' loops back to {name}")
    price = models[name]
    if price.same_as is not None:
        if price.same_as not in models:
            raise ValueError(f"{name}: 'same_as' names an unknown model {price.same_as!r}")
        return _resolve(price.same_as, models, {*seen, name})
    if price.tiers is not None:
        return tuple(price.tiers)
    assert price.input is not None  # noqa: S101 (the validator guarantees a flat price)
    return (
        Tier(
            input=price.input,
            output=price.output or Decimal(0),
            cache_read=price.cache_read,
            cache_write_5m=price.cache_write_5m,
            cache_write_1h=price.cache_write_1h,
        ),
    )


class _DecimalLoader(yaml.SafeLoader):
    """YAML floats become exact Decimals ("0.10" stays 0.10)."""


def _decimal(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Decimal:
    return Decimal(loader.construct_scalar(node))


_DecimalLoader.add_constructor("tag:yaml.org,2002:float", _decimal)


class PricingError(Exception):
    pass


def parse_pricing(text: str, source: str) -> PriceTable:
    try:
        data: Any = yaml.load(text, Loader=_DecimalLoader)  # noqa: S506 (a SafeLoader subclass)
    except yaml.YAMLError as exc:
        raise PricingError(f"{source}: invalid YAML: {exc}") from None
    try:
        pricing = PricingFile.model_validate(data)
        return PriceTable(pricing, source)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg'].removeprefix('Value error, ')}"
            for err in exc.errors(include_url=False)
        )
        raise PricingError(f"{source}: {problems}") from None
    except ValueError as exc:
        raise PricingError(f"{source}: {exc}") from None


def load_pricing(path: Path) -> tuple[PriceTable, str]:
    """The table and the file's text (part of the config version hash)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PricingError(f"cannot read pricing file {path}: {exc}") from None
    return parse_pricing(text, str(path)), text


def estimate_tokens_cost(
    prices: PriceTable, models: list[str], prompt_tokens: int, output_tokens: int
) -> Cost:
    """The highest estimate across a plan's candidates: any of them may serve it."""
    best = Decimal(0)
    for model in models:
        value = prices.estimate(model, prompt_tokens, output_tokens)
        if value is not None and value > best:
            best = value
    return Cost(best, estimated=True)
