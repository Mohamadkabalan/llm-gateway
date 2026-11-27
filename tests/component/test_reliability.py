"""Phase 3 acceptance through real sockets: failover, breakers, hedging, keep-alive,
rules and hot reload, driven by the OpenAI SDK and raw HTTP.

One fake provider plays both the failing primary and the healthy fallback: faults
carry ``fault_models``, so they hit only the named upstream model.
"""

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Any

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam
from prometheus_client import REGISTRY

from tests.component.conftest import Stack, StackFactory

USER: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]
FAST = {"ttft_ms": 0, "interval_ms": 0, "chunks": 4}
TEXT = "lorem ipsum dolor sit "
MODELS = """\
  fake/fake-backup:
    upstream_model: fake-backup
  fake/seeing:
    upstream_model: seeing
    capabilities: { vision: true }
"""
ALIASES = """\
  failover:
    targets: [{ model: fake/fake-chat }]
    fallbacks: [fake/fake-backup]
  hedged:
    targets: [{ model: fake/fake-chat }]
    fallbacks: [fake/fake-backup]
    hedge: { after: 200ms, max_ratio: 1.0 }
  sees-images:
    targets: [{ model: fake/fake-chat }]
    fallbacks: [fake/seeing]
"""


def on_primary(**fault: Any) -> dict[str, Any]:
    """A fault for the primary (upstream ``fake-chat``) only."""
    return {"fake": {**FAST, "fault_models": ["fake-chat"], **fault}}


def upstream_models(stack: Stack) -> list[str]:
    return [entry["body"]["model"] for entry in stack.fake.stats.recent]


def counter(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(f"{name}_total", labels) or 0.0


async def wait_for(condition: Any, within: float = 5.0) -> None:
    async with asyncio.timeout(within):
        while not condition():  # noqa: ASYNC110 (another server's state; nothing to await)
            await asyncio.sleep(0.02)


@pytest.fixture
async def stack(make_stack: StackFactory) -> Any:
    async with make_stack(models=MODELS, aliases=ALIASES) as running:
        yield running


# --- fallback before the commit point ---------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
async def test_503_on_the_primary_falls_back_and_the_client_sees_success(
    stack: Stack, stream: bool
) -> None:
    raw = await stack.sdk().chat.completions.with_raw_response.create(
        model="failover",
        messages=USER,
        stream=stream,
        extra_body=on_primary(fault="status", status=503),
    )
    if stream:
        chunks = [chunk async for chunk in raw.parse()]  # type: ignore[union-attr]
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    else:
        text = raw.parse().choices[0].message.content or ""  # type: ignore[union-attr]
    assert text == TEXT
    assert raw.headers["x-switchboard-model"] == "fake/fake-backup"
    assert raw.headers["x-switchboard-fallback"] == "true"
    assert raw.headers["x-switchboard-attempts"] == "3"  # primary, its retry, the fallback
    assert upstream_models(stack) == ["fake-chat", "fake-chat", "fake-backup"]


async def test_first_token_timeout_falls_back(make_stack: StackFactory) -> None:
    defaults = "defaults: { timeouts: { first_token: 300ms } }"
    async with make_stack(defaults=defaults, models=MODELS, aliases=ALIASES) as stack:
        raw = await stack.sdk().chat.completions.with_raw_response.create(
            model="failover", messages=USER, extra_body=on_primary(fault="hang")
        )
        assert raw.parse().choices[0].message.content == TEXT
        assert raw.headers["x-switchboard-model"] == "fake/fake-backup"
        # A timed-out target is retried once; both hung calls were hung up on.
        await wait_for(lambda: stack.fake.stats.client_disconnects == 2)


@pytest.mark.parametrize("fault", ["mid_stream_error", "disconnect"])
async def test_failure_before_the_first_token_is_invisible_to_the_sdk(
    stack: Stack, fault: str
) -> None:
    stream = await stack.sdk().chat.completions.create(
        model="failover",
        messages=USER,
        stream=True,
        extra_body=on_primary(fault=fault, fault_after=0),
    )
    chunks = [chunk async for chunk in stream]
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == TEXT
    assert {c.model for c in chunks} == {"fake-backup"}


@pytest.mark.parametrize("fault", ["mid_stream_error", "disconnect"])
async def test_failure_after_the_first_token_reaches_the_sdk_without_fallback(
    stack: Stack, fault: str
) -> None:
    stream = await stack.sdk().chat.completions.create(
        model="failover",
        messages=USER,
        stream=True,
        extra_body=on_primary(fault=fault, fault_after=2),
    )
    with pytest.raises(openai.APIError, match="upstream provider"):
        async for _ in stream:
            pass
    assert "fake-backup" not in upstream_models(stack)


async def test_a_capable_fallback_serves_what_the_primary_cannot(stack: Stack) -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    async with stack.http() as http:
        response = await http.post(
            "/v1/chat/completions",
            json={
                "model": "sees-images",
                "messages": [{"role": "user", "content": [image]}],
                **on_primary(),
            },
        )
    assert response.status_code == 200
    assert response.headers["x-switchboard-model"] == "fake/seeing"
    assert response.headers["x-switchboard-attempts"] == "1"  # skipped, not tried
    assert upstream_models(stack) == ["seeing"]


# --- circuit breaker ---------------------------------------------------------------------


async def test_breaker_opens_at_the_threshold_and_recovers_through_half_open(
    make_stack: StackFactory,
) -> None:
    defaults = (
        "defaults:\n"
        "  retry: { max_attempts: 1 }\n"
        "  breaker: { min_requests: 3, window: 10s, cooldown: 1s, half_open_probes: 2 }\n"
    )
    async with make_stack(defaults=defaults, models=MODELS, aliases=ALIASES) as stack:
        client = stack.sdk()
        state = {"model": "fake/fake-chat"}

        async def served_by() -> str:
            raw = await client.chat.completions.with_raw_response.create(
                model="failover", messages=USER, extra_body={"fake": FAST}
            )
            return raw.headers["x-switchboard-model"]

        stack.fake.runtime = {"fault": "status", "status": 503, "fault_models": ["fake-chat"]}
        assert [await served_by() for _ in range(3)] == ["fake/fake-backup"] * 3
        assert REGISTRY.get_sample_value("switchboard_breaker_state", state) == 2  # open
        primary_calls = upstream_models(stack).count("fake-chat")
        assert primary_calls == 3

        assert await served_by() == "fake/fake-backup"
        assert upstream_models(stack).count("fake-chat") == primary_calls  # skipped: no call

        stack.fake.runtime = {}  # the primary recovers
        await asyncio.sleep(1.05)  # the cooldown
        assert await served_by() == "fake/fake-chat"  # probe 1
        assert REGISTRY.get_sample_value("switchboard_breaker_state", state) == 1  # half-open
        assert await served_by() == "fake/fake-chat"  # probe 2 closes it
        assert REGISTRY.get_sample_value("switchboard_breaker_state", state) == 0
        assert await served_by() == "fake/fake-chat"


# --- hedging -----------------------------------------------------------------------------


async def test_hedge_wins_over_a_slow_primary_and_the_loser_is_cancelled(stack: Stack) -> None:
    won = counter("switchboard_hedges", alias="hedged", outcome="won")
    started = time.perf_counter()
    raw = await stack.sdk().chat.completions.with_raw_response.create(
        model="hedged", messages=USER, extra_body=on_primary(fault="hang")
    )
    elapsed = time.perf_counter() - started
    assert raw.parse().choices[0].message.content == TEXT
    assert raw.headers["x-switchboard-model"] == "fake/fake-backup"
    assert elapsed < 2.0  # the hedge started at 200 ms; the primary never answered
    await wait_for(lambda: stack.fake.stats.client_disconnects == 1)  # the primary, cancelled
    assert counter("switchboard_hedges", alias="hedged", outcome="won") == won + 1


# --- keep-alive --------------------------------------------------------------------------


async def test_keepalive_pings_before_a_slow_first_token(make_stack: StackFactory) -> None:
    async with make_stack(defaults="defaults: { keepalive_after: 200ms }") as stack:
        slow = {**FAST, "ttft_ms": 700}
        body = {"model": "fake", "messages": USER, "stream": True, "fake": slow}
        async with (
            stack.http() as http,
            http.stream("POST", "/v1/chat/completions", json=body) as r,
        ):
            lines = [line async for line in r.aiter_lines() if line]
        assert r.status_code == 200
        first_data = next(i for i, line in enumerate(lines) if line.startswith("data:"))
        assert lines[:first_data].count(": ping") >= 2
        assert "x-switchboard-model" not in r.headers  # sent before routing was decided
        # The SDK ignores the comments.
        stream = await stack.sdk().chat.completions.create(
            model="fake", messages=USER, stream=True, extra_body={"fake": slow}
        )
        text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
        assert text == TEXT


async def test_total_failure_after_keepalive_is_an_in_stream_error(
    make_stack: StackFactory,
) -> None:
    defaults = "defaults: { keepalive_after: 200ms, timeouts: { first_token: 500ms } }"
    async with make_stack(defaults=defaults) as stack:
        stream = await stack.sdk().chat.completions.create(
            model="fake", messages=USER, stream=True, extra_body={"fake": {"fault": "hang"}}
        )
        with pytest.raises(openai.APIError, match="timed out"):
            async for _ in stream:
                pass


# --- timeouts ----------------------------------------------------------------------------


async def test_idle_timeout_is_independent_of_the_first_token_timeout(
    make_stack: StackFactory,
) -> None:
    defaults = "defaults: { timeouts: { first_token: 5s, idle: 300ms } }"
    async with make_stack(defaults=defaults) as stack:
        started = time.perf_counter()
        stream = await stack.sdk().chat.completions.create(
            model="fake",
            messages=USER,
            stream=True,
            extra_body={"fake": {**FAST, "interval_ms": 1500, "chunks": 3}},
        )
        with pytest.raises(openai.APIError, match="timed out"):
            async for _ in stream:
                pass
        assert time.perf_counter() - started < 2.0  # not the 5 s first-token timeout


# --- refusals ----------------------------------------------------------------------------

CLAUDE = "  claude: { type: anthropic, base_url: '{fake_url}' }"
CLAUDE_MODEL = (
    "  claude/sonnet: { upstream_model: claude-sonnet-5-5, defaults: { max_tokens: 64 } }"
)
REFUSAL_ALIASES = """\
  careful:
    targets: [{ model: claude/sonnet }]
    fallbacks: [fake/fake-chat]
  careful-with-fallback:
    targets: [{ model: claude/sonnet }]
    fallbacks: [fake/fake-chat]
    fallback_on_refusal: true
"""


@pytest.mark.parametrize(
    ("alias", "finish", "model"),
    [
        ("careful", "content_filter", "claude/sonnet"),
        ("careful-with-fallback", "stop", "fake/fake-chat"),
    ],
)
async def test_refusal_before_the_first_token_falls_back_only_when_configured(
    make_stack: StackFactory, alias: str, finish: str, model: str
) -> None:
    async with make_stack(providers=CLAUDE, models=CLAUDE_MODEL, aliases=REFUSAL_ALIASES) as stack:
        raw = await stack.sdk().chat.completions.with_raw_response.create(
            model=alias,
            messages=USER,
            extra_body={
                # The Anthropic primary refuses; the OpenAI fallback answers.
                "provider_options": {"anthropic": {"fake": {**FAST, "scenario": "refusal"}}},
                "fake": FAST,
            },
        )
        assert raw.parse().choices[0].finish_reason == finish
        assert raw.headers["x-switchboard-model"] == model


# --- rules and tags ----------------------------------------------------------------------


async def test_request_tags_select_a_rule(make_stack: StackFactory) -> None:
    rules = (
        "  - name: tagged-backup\n"
        "    when: { model: fake, tags_any: [backup] }\n"
        "    use: { targets: [{ model: fake/fake-backup }] }\n"
    )
    async with make_stack(models=MODELS, rules=rules) as stack, stack.http() as http:
        body = {"model": "fake", "messages": USER, "fake": FAST}
        tagged = await http.post(
            "/v1/chat/completions", json=body, headers={"x-switchboard-tags": "batch, backup"}
        )
        plain = await http.post("/v1/chat/completions", json=body)
        too_many = await http.post(
            "/v1/chat/completions",
            json=body,
            headers={"x-switchboard-tags": ",".join(f"t{i}" for i in range(40))},
        )
    assert tagged.headers["x-switchboard-model"] == "fake/fake-backup"
    assert plain.headers["x-switchboard-model"] == "fake/fake-chat"
    assert too_many.status_code == 400
    assert too_many.json()["error"]["code"] == "invalid_tags"


# --- hot reload --------------------------------------------------------------------------


async def ready_version(stack: Stack) -> str:
    async with stack.http() as http:
        return str((await http.get("/ready")).json()["config_version"])


async def test_file_watch_reload_and_an_invalid_reload_keeps_the_old_config(
    make_stack: StackFactory, tmp_path: Path
) -> None:
    config = tmp_path / "routing.yaml"
    async with make_stack(config_file=config, config_watch_seconds=0.1) as stack:
        original = config.read_text()
        version = await ready_version(stack)
        errors = counter("switchboard_config_reloads", trigger="watch", result="error")

        config.write_text(original + "\n  broken: [\n")
        await wait_for(
            lambda: (
                counter("switchboard_config_reloads", trigger="watch", result="error") == errors + 1
            )
        )
        assert await ready_version(stack) == version
        async with stack.http() as http:
            ok = await http.post(
                "/v1/chat/completions", json={"model": "fake", "messages": USER, "fake": FAST}
            )
        assert ok.status_code == 200  # still serving on the old config

        successes = counter("switchboard_config_reloads", trigger="watch", result="success")
        config.write_text(original.replace("  fake:\n    targets", "  renamed:\n    targets", 1))
        await wait_for(
            lambda: (
                counter("switchboard_config_reloads", trigger="watch", result="success")
                == successes + 1
            )
        )
        assert await ready_version(stack) != version
        async with stack.http() as http:
            renamed = await http.post(
                "/v1/chat/completions", json={"model": "renamed", "messages": USER, "fake": FAST}
            )
            gone = await http.post(
                "/v1/chat/completions", json={"model": "fake", "messages": USER, "fake": FAST}
            )
        assert renamed.status_code == 200
        assert gone.status_code == 404


async def test_sighup_reloads_the_config(make_stack: StackFactory, tmp_path: Path) -> None:
    config = tmp_path / "routing.yaml"
    async with make_stack(config_file=config, config_watch_seconds=0) as stack:
        version = await ready_version(stack)
        config.write_text(config.read_text() + "\n# a comment changes the hash\n")
        before = counter("switchboard_config_reloads", trigger="signal", result="success")
        os.kill(os.getpid(), signal.SIGHUP)
        await wait_for(
            lambda: (
                counter("switchboard_config_reloads", trigger="signal", result="success")
                == before + 1
            )
        )
        assert await ready_version(stack) != version
