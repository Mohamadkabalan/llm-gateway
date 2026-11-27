"""The routing config shipped in the repo must always load."""

from switchboard.routing import load_routing_config
from tests.support import REPO_ROOT

SHIPPED = REPO_ROOT / "config" / "routing.yaml"


def test_shipped_config_loads_without_any_provider_keys() -> None:
    loaded = load_routing_config(SHIPPED, env={})
    assert {"anthropic", "openai"} <= loaded.disabled_providers
    assert "fake" in loaded.config.aliases  # usable with no keys at all
    assert all(w.severity == "warning" for w in loaded.warnings)


def test_shipped_config_with_keys_enables_everything() -> None:
    loaded = load_routing_config(SHIPPED, env={"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"})
    assert loaded.disabled_providers == frozenset()
    assert loaded.warnings == ()
