"""The router: aliases, rules, weighted and sticky A/B, fallbacks and the capability filter."""

import random
from collections import Counter
from typing import Any

import pytest

from switchboard.routing import parse_routing_config
from switchboard.routing.capabilities import estimate_prompt_tokens
from switchboard.routing.config import ModelKind
from switchboard.routing.table import (
    DIRECT_ROUTE,
    RouteError,
    RoutePlan,
    RouteRequest,
    RoutingTable,
    sticky_point,
)
from switchboard.schemas.chat import ChatRequest

CONFIG = """\
version: 1
defaults:
  timeouts: { connect: 1s, first_token: 20s, idle: 15s, total: 300s }
  retry: { max_attempts: 3 }
providers:
  fast:
    type: openai
    base_url: http://fast/v1
    timeouts: { first_token: 5s }
    retry: { max_attempts: 1 }
    breaker: { min_requests: 4 }
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
  fast/vision:
    upstream_model: vision
    capabilities: { vision: true, max_input_tokens: 100000 }
  fast/small:
    upstream_model: small
    capabilities: { max_input_tokens: 50 }
  fast/embed:
    upstream_model: embed
    kind: embedding
  gone/chat:
    upstream_model: chat
aliases:
  smart:
    targets: [{ model: fast/chat, weight: 3 }, { model: fast/second }]
    fallbacks: [fast/second, fast/vision]
  sticky-caller:
    targets: [{ model: fast/chat }, { model: fast/second }]
    sticky: key
  sticky-user:
    targets: [{ model: fast/chat }, { model: fast/second }]
    sticky: end_user
  vectors:
    targets: [{ model: fast/embed }]
  offline:
    targets: [{ model: gone/chat }]
  offline-then-fast:
    targets: [{ model: gone/chat }]
    fallbacks: [fast/chat]
  tiny:
    targets: [{ model: fast/small }]
    fallbacks: [fast/vision]
  fast-only:
    targets: [{ model: fast/second }]
rules:
  - name: tagged-goes-fast
    when: { model: smart, tags_any: [cheap, batch] }
    use: { alias: fast-only }
  - name: all-tags
    when: { model: smart, tags_all: [a, b] }
    use: { targets: [{ model: fast/vision }], fallbacks: [fast/chat] }
  - name: short-tool-free
    when: { model: smart, prompt_tokens: { lt: 20 }, has_tools: false, has_images: false }
    use: { alias: fast-only }
  - name: streams-with-images
    when: { model: smart, stream: true, has_images: true }
    use: { targets: [{ model: fast/vision }] }
"""
LONG = "word " * 100  # about 125 tokens by the estimate, past short-tool-free
IMAGE = [
    {"type": "text", "text": LONG},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
]


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable(parse_routing_config(CONFIG, source="t.yaml", env={}), rng=random.Random(7))


def chat(model: str = "smart", content: Any = LONG, **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    )


def plan(table: RoutingTable, model: str = "smart", **kwargs: Any) -> RoutePlan:
    body = kwargs.pop("chat", None) or chat(model)
    return table.plan(RouteRequest(model=model, kind=ModelKind.CHAT, chat=body, **kwargs))


def names(p: RoutePlan) -> list[str]:
    return [t.name for t in p.candidates]


# --- resolution ------------------------------------------------------------------------


def test_alias_plan_has_primary_then_fallbacks_without_repeating_it() -> None:
    table = RoutingTable(parse_routing_config(CONFIG, env={}), rng=random.Random(0))
    seen = {tuple(names(plan(table))) for _ in range(50)}
    # The primary is fast/chat or fast/second; a fallback equal to it is not repeated.
    assert seen == {("fast/chat", "fast/second", "fast/vision"), ("fast/second", "fast/vision")}


def test_concrete_model_routes_directly(table: RoutingTable) -> None:
    p = plan(table, "fast/second")
    assert p.alias is None
    assert p.route_label == DIRECT_ROUTE
    assert names(p) == ["fast/second"]


def test_policies_merge_defaults_then_provider_then_model(table: RoutingTable) -> None:
    target = plan(table, "fast/chat").primary
    timeouts = target.timeouts
    assert (timeouts.connect, timeouts.first_token, timeouts.idle, timeouts.total) == (
        1.0,
        5.0,
        2.0,
        300.0,
    )
    assert target.retry.max_attempts == 1  # provider override beats defaults (3)
    assert target.retry.base_delay == 0.25  # untouched fields keep the default
    assert target.breaker.min_requests == 4
    assert target.breaker.window == 30.0


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
    body = chat(requested) if kind is ModelKind.CHAT else None
    with pytest.raises(RouteError) as caught:
        table.plan(RouteRequest(model=requested, kind=kind, chat=body))
    assert caught.value.status == status
    assert caught.value.code == code


def test_a_disabled_primary_falls_through_to_its_fallback(table: RoutingTable) -> None:
    p = plan(table, "offline-then-fast")
    assert names(p) == ["fast/chat"]
    assert [(s.model, s.reason) for s in p.skipped] == [("gone/chat", "provider disabled")]


def test_listing_includes_aliases_with_any_enabled_candidate(table: RoutingTable) -> None:
    listed = {name for name, _, _ in table.listing()}
    assert "offline-then-fast" in listed
    assert "offline" not in listed
    assert "gone/chat" not in listed


# --- capability filter -------------------------------------------------------------------


def test_capability_skip_lets_a_capable_fallback_serve(table: RoutingTable) -> None:
    body = chat("tiny")  # the prompt is far over fast/small's 50-token limit
    p = plan(table, "tiny", chat=body)
    assert names(p) == ["fast/vision"]
    (skipped,) = p.skipped
    assert skipped.model == "fast/small"
    estimate = estimate_prompt_tokens(body)
    assert skipped.capability_gaps == (f"a prompt of about {estimate} tokens (limit 50)",)


def test_no_capable_candidate_is_a_400_naming_the_gap(table: RoutingTable) -> None:
    with pytest.raises(RouteError) as caught:
        plan(table, "fast-only", chat=chat("fast-only", IMAGE))
    assert caught.value.status == 400
    assert caught.value.code == "model_capability_missing"
    assert caught.value.message == "The model 'fast-only' does not support image input."


# --- rules -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_kwargs", "chat_kwargs", "rule", "first"),
    [
        ({"tags": frozenset({"batch"})}, {}, "tagged-goes-fast", "fast/second"),
        ({"tags": frozenset({"a"})}, {}, None, None),
        ({"tags": frozenset({"a", "b"})}, {}, "all-tags", "fast/vision"),
        ({}, {"content": "hi"}, "short-tool-free", "fast/second"),
        (
            {},
            {"content": "hi", "tools": [{"type": "function", "function": {"name": "f"}}]},
            None,
            None,
        ),
        ({}, {"content": IMAGE, "stream": True}, "streams-with-images", "fast/vision"),
        ({}, {"content": IMAGE}, None, None),
        # First match wins: tags_any matches before the short-prompt rule.
        ({"tags": frozenset({"cheap"})}, {"content": "hi"}, "tagged-goes-fast", "fast/second"),
    ],
)
def test_rules(
    table: RoutingTable,
    request_kwargs: dict[str, Any],
    chat_kwargs: dict[str, Any],
    rule: str | None,
    first: str | None,
) -> None:
    content = chat_kwargs.pop("content", LONG)
    p = plan(table, chat=chat("smart", content, **chat_kwargs), **request_kwargs)
    assert p.rule == rule
    assert p.alias == "smart"  # the requested alias, whatever served it
    if first is not None:
        assert p.primary.name == first


def test_rule_to_an_alias_reports_the_serving_route(table: RoutingTable) -> None:
    p = plan(table, tags=frozenset({"cheap"}))
    assert p.route == "fast-only"
    assert names(p) == ["fast/second"]


def test_inline_rule_route_has_its_own_fallbacks(table: RoutingTable) -> None:
    p = plan(table, tags=frozenset({"a", "b"}))
    assert p.route is None
    assert names(p) == ["fast/vision", "fast/chat"]


def test_rules_ignore_requests_for_other_models(table: RoutingTable) -> None:
    p = plan(table, "fast/chat", tags=frozenset({"cheap"}))
    assert p.rule is None


def test_key_and_team_conditions() -> None:
    text = CONFIG.replace(
        "rules:\n",
        "rules:\n"
        "  - name: research\n"
        "    when: { team: research, key: lab }\n"
        "    use: { alias: fast-only }\n",
    )
    table = RoutingTable(parse_routing_config(text, env={}))
    assert plan(table, team="research", key="lab").rule == "research"
    assert plan(table, team="research", key="other").rule != "research"
    assert plan(table, team=None, key="lab").rule != "research"


# --- A/B selection ---------------------------------------------------------------------


def chi_square(observed: Counter[str], expected: dict[str, float]) -> float:
    return sum((observed[k] - e) ** 2 / e for k, e in expected.items())


# Critical values at p = 0.001; a correct splitter fails these about once in 1000 seeds,
# and the seeds here are fixed, so the tests are deterministic.
CHI2_CRITICAL_DF1 = 10.83


def test_weighted_split_matches_weights() -> None:
    table = RoutingTable(parse_routing_config(CONFIG, env={}), rng=random.Random(42))
    n = 20_000
    counts = Counter(plan(table).primary.name for _ in range(n))
    expected = {"fast/chat": n * 3 / 4, "fast/second": n / 4}
    assert chi_square(counts, expected) < CHI2_CRITICAL_DF1


def test_sticky_key_is_stable_per_subject(table: RoutingTable) -> None:
    for subject in ("subject-1", "subject-2", "subject-3"):
        picks = {
            plan(table, "sticky-caller", sticky_subject=subject).primary.name for _ in range(20)
        }
        assert len(picks) == 1


def test_sticky_split_across_subjects_matches_weights(table: RoutingTable) -> None:
    n = 10_000
    counts = Counter(
        plan(table, "sticky-caller", sticky_subject=f"subject-{i}").primary.name for i in range(n)
    )
    assert chi_square(counts, {"fast/chat": n / 2, "fast/second": n / 2}) < CHI2_CRITICAL_DF1


def test_sticky_end_user_uses_the_user_field(table: RoutingTable) -> None:
    for user in ("u1", "u2", "u3", "u4"):
        body = chat("sticky-user", user=user)
        picks = {plan(table, "sticky-user", chat=body).primary.name for _ in range(10)}
        assert len(picks) == 1


def test_sticky_without_a_subject_is_random(table: RoutingTable) -> None:
    picks = {plan(table, "sticky-user").primary.name for _ in range(200)}  # no `user`
    assert picks == {"fast/chat", "fast/second"}


def test_sticky_point_depends_on_route_and_subject() -> None:
    assert sticky_point("a", "x", 10) == sticky_point("a", "x", 10)
    assert sticky_point("a", "x", 10) != sticky_point("b", "x", 10)
    assert 0 <= sticky_point("a", "x", 10) < 10


def test_embedding_requests_route_too(table: RoutingTable) -> None:
    p = table.plan(RouteRequest(model="vectors", kind=ModelKind.EMBEDDING, embedding_chars=40))
    assert names(p) == ["fast/embed"]
