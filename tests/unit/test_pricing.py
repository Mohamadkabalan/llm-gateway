"""Cost calculation against hand-computed values from the verified price rows
(config/pricing/2026-11-27.yaml; sources and retrieval date are in the file)."""

from decimal import Decimal
from pathlib import Path

import pytest

from switchboard.accounting.pricing import (
    Cost,
    PriceTable,
    PricingError,
    estimate_tokens_cost,
    load_pricing,
    parse_pricing,
)
from switchboard.routing import ConfigError, load_routing_config, parse_routing_config
from switchboard.schemas.events import Usage
from tests.support import REPO_ROOT

PRICING = REPO_ROOT / "config" / "pricing" / "2026-11-27.yaml"


@pytest.fixture(scope="module")
def table() -> PriceTable:
    return load_pricing(PRICING)[0]


@pytest.mark.parametrize(
    ("model", "usage", "expected"),
    [
        # Sonnet 5.5: $2 in, $10 out, $0.10 cache hits, $2.50 5m writes, $4 1h writes.
        # 1,000 in = 0.002; 2,000 out = 0.02; 10,000 hits = 0.001; 4,000 5m = 0.01;
        # 500 1h = 0.002. Total 0.035.
        (
            "anthropic/claude-sonnet-5-5",
            Usage(
                input=1000, output=2000, cache_read=10_000, cache_write_5m=4000, cache_write_1h=500
            ),
            Decimal("0.035"),
        ),
        # Opus 5.5: 1M input at $4 and 1M output at $20.
        ("anthropic/claude-opus-5-5", Usage(input=1_000_000, output=1_000_000), Decimal(24)),
        # Haiku 5.5 at exactly 100,000 prompt tokens is the lower tier:
        # 100,000 x 0.10 / 1e6 = 0.01; 1,000 x 0.50 / 1e6 = 0.0005.
        ("anthropic/claude-haiku-5-5", Usage(input=100_000, output=1000), Decimal("0.0105")),
        # One more token is the upper tier for the whole request:
        # 100,001 x 0.50 / 1e6 = 0.0500005; 1,000 x 2.50 / 1e6 = 0.0025.
        ("anthropic/claude-haiku-5-5", Usage(input=100_001, output=1000), Decimal("0.0525005")),
        # Cache reads count toward the tier ("all of its input tokens, including cache
        # reads and cache writes"): 50,000 in + 60,000 hits = 110,000 → upper tier.
        # 50,000 x 0.50 = 0.025; 60,000 x 0.05 = 0.003; output 0.
        ("anthropic/claude-haiku-5-5", Usage(input=50_000, cache_read=60_000), Decimal("0.028")),
        # gpt-5.4-mini: $0.75 in, $0.075 cached, $4.50 out.
        # 1,000 x 0.75 = 0.00075; 3,000 x 0.075 = 0.000225; 500 x 4.50 = 0.00225.
        (
            "openai/gpt-5.4-mini",
            Usage(input=1000, cache_read=3000, output=500),
            Decimal("0.003225"),
        ),
        # Embeddings: input only, $0.02.
        ("openai/text-embedding-3-small", Usage(input=50_000), Decimal("0.001")),
        ("ollama/llama3.2", Usage(input=10_000, output=10_000), Decimal(0)),
        # Fake models are copies of real rows.
        ("fake/fake-chat", Usage(input=1000, output=2000), Decimal("0.022")),
    ],
)
def test_cost_matches_hand_computed_values(
    table: PriceTable, model: str, usage: Usage, expected: Decimal
) -> None:
    assert table.cost(model, usage) == expected


def test_reasoning_tokens_are_billed_as_output(table: PriceTable) -> None:
    # `output` already includes reasoning; `reasoning` must not be added again.
    with_reasoning = Usage(input=0, output=1000, reasoning=800)
    assert table.cost("anthropic/claude-sonnet-5-5", with_reasoning) == Decimal("0.01")


def test_unpriced_model_returns_none(table: PriceTable) -> None:
    assert table.cost("nobody/nothing", Usage(input=1)) is None


def test_prices_are_exact_decimals_not_floats() -> None:
    table = parse_pricing(
        "version: t\ncurrency: USD\nmodels:\n  a/b: { input: 0.10, output: 0.30 }\n", "t.yaml"
    )
    rates = table.rates("a/b", 0)
    assert rates is not None
    assert rates.input == Decimal("0.10")
    assert table.cost("a/b", Usage(input=3, output=7)) == Decimal("0.0000024")


def test_nano_rounds_up_so_budgets_never_undercount() -> None:
    assert Cost(Decimal("0.0000000011")).nano == 2
    assert Cost(Decimal("0.000000001")).nano == 1
    assert Cost(Decimal(0)).nano == 0


def test_estimate_takes_the_most_expensive_candidate(table: PriceTable) -> None:
    models = ["openai/gpt-5.4-mini", "anthropic/claude-opus-5-5"]
    estimate = estimate_tokens_cost(table, models, 1000, 1000)
    assert estimate.usd == Decimal("0.024")  # Opus: 1,000 x 4 + 1,000 x 20, per million
    assert estimate.estimated


@pytest.mark.parametrize(
    ("models", "problem"),
    [
        ("  a/b: { input: 1, same_as: c/d }", "exactly one of"),
        ("  a/b: { output: 1 }", "exactly one of"),
        ("  a/b: { same_as: c/d }", "unknown model 'c/d'"),
        ("  a/b: { same_as: c/d }\n  c/d: { same_as: a/b }", "loops back"),
        (
            "  a/b: { tiers: [{ input: 1 }, { max_prompt_tokens: 5, input: 2 }] }",
            "every tier but the last",
        ),
        (
            "  a/b: { tiers: [{ max_prompt_tokens: 9, input: 1 },"
            " { max_prompt_tokens: 5, input: 2 }, { input: 3 }] }",
            "must increase",
        ),
        ("  a/b: { input: -1 }", "greater than or equal to 0"),
    ],
)
def test_invalid_pricing_files(models: str, problem: str) -> None:
    with pytest.raises(PricingError, match=problem):
        parse_pricing(f"version: t\ncurrency: USD\nmodels:\n{models}\n", "t.yaml")


def test_shipped_routing_config_prices_every_model() -> None:
    loaded = load_routing_config(REPO_ROOT / "config" / "routing.yaml", env={})
    assert loaded.pricing is not None
    assert loaded.pricing.version == "2026-11-27"
    assert all(name in loaded.pricing for name in loaded.config.models)


def test_a_model_without_a_price_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / "prices.yaml").write_text(
        "version: t\ncurrency: USD\nmodels:\n  p/a: { input: 1 }\n"
    )
    text = (
        "version: 1\npricing: prices.yaml\n"
        "providers:\n  p: { type: openai, base_url: 'http://p' }\n"
        "models:\n  p/a: { upstream_model: a }\n  p/b: { upstream_model: b }\n"
    )
    with pytest.raises(ConfigError) as caught:
        parse_routing_config(text, base_dir=tmp_path)
    assert any(
        "no price for 'p/b' in pricing file version t" in str(i) for i in caught.value.issues
    )


def test_the_pricing_file_is_part_of_the_config_version(tmp_path: Path) -> None:
    prices = tmp_path / "prices.yaml"
    text = (
        "version: 1\npricing: prices.yaml\n"
        "providers:\n  p: { type: openai, base_url: 'http://p' }\n"
        "models:\n  p/a: { upstream_model: a }\n"
    )
    prices.write_text("version: one\ncurrency: USD\nmodels:\n  p/a: { input: 1 }\n")
    first = parse_routing_config(text, base_dir=tmp_path).version
    prices.write_text("version: two\ncurrency: USD\nmodels:\n  p/a: { input: 2 }\n")
    assert parse_routing_config(text, base_dir=tmp_path).version != first
