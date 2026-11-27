from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from switchboard.routing import ConfigError, parse_routing_config
from switchboard.routing.config import parse_duration
from switchboard.routing.loader import load_routing_config
from tests.support import MINIMAL_CONFIG

BASE = """\
version: 1
providers:
  anthropic:
    type: anthropic
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
  fake:
    type: openai
    base_url: http://fake-provider:8000/v1
models:
  anthropic/claude-sonnet-5-5:
    upstream_model: claude-sonnet-5-5
  fake/fake-chat:
    upstream_model: fake-chat
  fake/fake-embed:
    upstream_model: fake-embed
    kind: embedding
aliases:
  smart:
    targets:
      - model: anthropic/claude-sonnet-5-5
        weight: 3
    fallbacks: [fake/fake-chat]
"""
KEYS = {"ANTHROPIC_API_KEY": "sk-test"}


def problems(text: str, env: dict[str, str] | None = None) -> list[str]:
    with pytest.raises(ConfigError) as info:
        parse_routing_config(text, source="t.yaml", env=KEYS if env is None else env)
    return [str(issue) for issue in info.value.issues]


def test_valid_config_loads_with_defaults_applied() -> None:
    loaded = parse_routing_config(BASE, source="t.yaml", env=KEYS)
    cfg = loaded.config
    assert cfg.aliases["smart"].targets[0].weight == 3
    assert cfg.defaults.timeouts.first_token == 30.0
    assert cfg.defaults.retry.base_delay == 0.25
    assert loaded.warnings == ()
    assert loaded.disabled_providers == frozenset()
    assert len(loaded.version) == 12


def test_version_changes_with_content() -> None:
    a = parse_routing_config(BASE, env=KEYS).version
    b = parse_routing_config(BASE + "\n# comment\n", env=KEYS).version
    assert a != b


def test_yaml_syntax_error_reports_line() -> None:
    text = BASE.replace("    fallbacks: [fake/fake-chat]", "    fallbacks: [fake/fake-chat")
    (msg,) = problems(text)
    assert "YAML syntax error" in msg
    assert msg.startswith("line ")


def test_unknown_field_suggests_the_right_name_with_line() -> None:
    text = BASE.replace("    fallbacks:", "    fallback:")
    (msg,) = problems(text)
    assert msg == (
        "line 23, aliases.smart.fallback: unknown field 'fallback' (did you mean 'fallbacks'?)"
    )


def test_unknown_top_level_field_lists_allowed_fields() -> None:
    (msg,) = problems(BASE + "zzz: 1\n")
    assert "unknown field 'zzz'" in msg
    assert "expected one of: aliases, defaults, models, providers, version" in msg


def test_wrong_type_points_at_the_value() -> None:
    (msg,) = problems(BASE.replace("weight: 3", "weight: heavy"))
    assert msg.startswith("line 22, aliases.smart.targets[0].weight:")
    assert "integer" in msg


def test_invalid_duration_is_explained() -> None:
    text = BASE + "defaults:\n  timeouts:\n    connect: soon\n"
    (msg,) = problems(text)
    assert "defaults.timeouts.connect" in msg
    assert "expected a duration such as '250ms', '2s' or '5m'" in msg


def test_cross_field_validation_runs() -> None:
    text = BASE + "defaults:\n  timeouts: { connect: 10s, total: 5s }\n"
    (msg,) = problems(text)
    assert "'total' must be at least as long as 'connect' and 'first_token'" in msg


def test_duplicate_keys_are_errors_not_silent_overrides() -> None:
    text = BASE.replace(
        "  fake/fake-chat:\n    upstream_model: fake-chat\n",
        "  fake/fake-chat:\n    upstream_model: fake-chat\n"
        "  fake/fake-chat:\n    upstream_model: other\n",
    )
    (msg,) = problems(text)
    assert msg == "line 15, models[\"fake/fake-chat\"]: duplicate key 'fake/fake-chat'"


def test_unknown_model_reference_suggests_close_match() -> None:
    text = BASE.replace(
        "- model: anthropic/claude-sonnet-5-5", "- model: anthropic/claude-sonet-5-5"
    )
    (msg,) = problems(text)
    assert "unknown model 'anthropic/claude-sonet-5-5'" in msg
    assert "did you mean 'anthropic/claude-sonnet-5-5'?" in msg
    assert msg.startswith("line 21,")


def test_model_with_unknown_provider() -> None:
    text = BASE.replace("  fake/fake-chat:\n", "  fak/fake-chat:\n").replace(
        "fallbacks: [fake/fake-chat]", "fallbacks: [fak/fake-chat]"
    )
    (msg,) = problems(text)
    assert "unknown provider 'fak' (did you mean 'fake'?)" in msg


def test_model_name_must_include_provider() -> None:
    text = BASE.replace("  fake/fake-chat:\n", "  fake-chat:\n").replace(
        "fallbacks: [fake/fake-chat]", "fallbacks: [fake-chat]"
    )
    assert any("must look like '<provider>/<model>'" in p for p in problems(text))


def test_alias_cannot_mix_chat_and_embedding_models() -> None:
    text = BASE.replace("fallbacks: [fake/fake-chat]", "fallbacks: [fake/fake-embed]")
    (msg,) = problems(text)
    assert "aliases.smart" in msg
    assert "mixes chat and embedding models" in msg


def test_duplicate_fallbacks_are_rejected() -> None:
    text = BASE.replace(
        "fallbacks: [fake/fake-chat]", "fallbacks: [fake/fake-chat, fake/fake-chat]"
    )
    (msg,) = problems(text)
    assert "aliases.smart.fallbacks[1]: 'fake/fake-chat' is listed twice in fallbacks" in msg


def test_missing_required_key_env_is_an_error() -> None:
    (msg,) = problems(BASE, env={})
    assert "environment variable ANTHROPIC_API_KEY is not set" in msg
    assert "optional: true" in msg


def test_optional_provider_without_key_is_disabled_with_warnings() -> None:
    text = BASE.replace(
        "    api_key_env: ANTHROPIC_API_KEY\n",
        "    api_key_env: ANTHROPIC_API_KEY\n    optional: true\n",
    )
    loaded = parse_routing_config(text, env={})
    assert loaded.disabled_providers == frozenset({"anthropic"})
    messages = [str(w) for w in loaded.warnings]
    assert any("ANTHROPIC_API_KEY is not set; provider disabled" in m for m in messages)
    assert any("aliases.smart: every target's provider is disabled" in m for m in messages)


def test_invalid_alias_name_is_reported() -> None:
    (msg,) = problems(BASE.replace("  smart:\n", "  Smart Alias:\n"))
    assert msg.startswith('line 19, aliases["Smart Alias"]: invalid name')


def test_all_problems_are_reported_together_sorted_by_line() -> None:
    text = BASE.replace("weight: 3", "weight: 0").replace("    fallbacks:", "    fallback:")
    found = problems(text)
    assert len(found) == 2
    assert found[0].startswith("line 22")
    assert found[1].startswith("line 23")


def test_empty_file_and_non_mapping_documents() -> None:
    assert problems("") == ["<root>: the file is empty"]
    assert problems("- a\n- b\n") == ["line 1, <root>: the top level must be a mapping"]


def test_only_version_1_is_accepted() -> None:
    assert any("version" in p for p in problems(MINIMAL_CONFIG.replace("version: 1", "version: 2")))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_routing_config(tmp_path / "missing.yaml")


def test_rendered_error_lists_every_problem() -> None:
    text = BASE.replace("weight: 3", "weight: 0").replace("    fallbacks:", "    fallback:")
    with pytest.raises(ConfigError) as info:
        parse_routing_config(text, source="config/routing.yaml", env=KEYS)
    rendered = info.value.render().splitlines()
    assert rendered[0] == "Invalid routing config config/routing.yaml (2 problems):"
    assert all(line.startswith("  - line ") for line in rendered[1:])


def test_merge_keys_and_anchors_work() -> None:
    text = """\
version: 1
providers:
  fake: &fake
    type: openai
    base_url: http://fake-provider:8000/v1
  fake2:
    <<: *fake
    base_url: http://other:8000/v1
"""
    providers = parse_routing_config(text, env={}).config.providers
    assert providers["fake2"].type == providers["fake"].type
    assert str(providers["fake2"].base_url) == "http://other:8000/v1"


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [("250ms", 0.25), ("2s", 2.0), ("1.5s", 1.5), ("5m", 300.0), ("1h", 3600.0), (3, 3.0)],
)
def test_parse_duration(raw: object, seconds: float) -> None:
    assert parse_duration(raw) == pytest.approx(seconds)


@pytest.mark.parametrize("raw", ["", "5", "5 days", "-1s", True, None, [1]])
def test_parse_duration_rejects_garbage(raw: object) -> None:
    with pytest.raises(ValueError, match="expected a duration"):
        parse_duration(raw)


@given(st.integers(min_value=0, max_value=10**6), st.sampled_from(["ms", "s", "m", "h"]))
def test_parse_duration_property(value: int, unit: str) -> None:
    factor = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    assert parse_duration(f"{value}{unit}") == pytest.approx(value * factor)
