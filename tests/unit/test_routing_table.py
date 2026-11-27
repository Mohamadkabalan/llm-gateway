import pytest

from switchboard.routing import parse_routing_config
from switchboard.routing.config import ModelKind
from switchboard.routing.table import DIRECT_ROUTE, RouteError, RoutingTable

CONFIG = """\
version: 1
defaults:
  timeouts: { connect: 1s, first_token: 20s, idle: 15s, total: 300s }
providers:
  fast:
    type: openai
    base_url: http://fast/v1
    timeouts: { first_token: 5s }
  gone:
    type: openai
    base_url: http://gone/v1
    api_key_env: GONE_API_KEY
    optional: true
models:
  fast/chat:
    upstream_model: chat-2026
    timeouts: { idle: 2s }
  fast/second:
    upstream_model: second
  fast/embed:
    upstream_model: embed
    kind: embedding
  gone/chat:
    upstream_model: chat
aliases:
  smart:
    targets: [{ model: fast/chat, weight: 3 }, { model: fast/second }]
  vectors:
    targets: [{ model: fast/embed }]
  offline:
    targets: [{ model: gone/chat }]
"""


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable(parse_routing_config(CONFIG, source="t.yaml", env={}))


def test_alias_resolves_to_its_first_target(table: RoutingTable) -> None:
    plan = table.resolve("smart", ModelKind.CHAT)
    assert plan.alias == "smart"
    assert plan.route_label == "smart"
    assert [t.name for t in plan.targets] == ["fast/chat"]
    assert plan.targets[0].upstream_model == "chat-2026"
    assert plan.targets[0].provider == "fast"


def test_concrete_model_resolves_directly(table: RoutingTable) -> None:
    plan = table.resolve("fast/second", ModelKind.CHAT)
    assert plan.alias is None
    assert plan.route_label == DIRECT_ROUTE
    assert plan.targets[0].name == "fast/second"


def test_timeouts_merge_defaults_then_provider_then_model(table: RoutingTable) -> None:
    timeouts = table.resolve("smart", ModelKind.CHAT).targets[0].timeouts
    assert (timeouts.connect, timeouts.first_token, timeouts.idle, timeouts.total) == (
        1.0,
        5.0,
        2.0,
        300.0,
    )


@pytest.mark.parametrize(
    ("requested", "kind", "status", "code"),
    [
        ("missing", ModelKind.CHAT, 404, "model_not_found"),
        ("fast/missing", ModelKind.CHAT, 404, "model_not_found"),
        ("vectors", ModelKind.CHAT, 400, "model_kind_mismatch"),
        ("smart", ModelKind.EMBEDDING, 400, "model_kind_mismatch"),
        ("offline", ModelKind.CHAT, 503, "provider_unavailable"),
        ("gone/chat", ModelKind.CHAT, 503, "provider_unavailable"),
    ],
)
def test_resolution_errors(
    table: RoutingTable, requested: str, kind: ModelKind, status: int, code: str
) -> None:
    with pytest.raises(RouteError) as caught:
        table.resolve(requested, kind)
    assert caught.value.status == status
    assert caught.value.code == code
