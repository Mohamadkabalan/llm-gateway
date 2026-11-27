"""Fixture streams through each adapter's real stream code (PLAN.md §15.1).

Fixtures live in tests/fixtures/<provider>/<scenario>.sse with a .meta.json saying
where the bytes came from (``source: docs`` until scripts/record_fixtures.py records
real ones). Every fixture is also replayed split at random byte boundaries, and must
produce exactly the same events.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from switchboard.providers import anthropic, openai
from switchboard.providers.base import CallTimings
from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.schemas.encode import Aggregator
from switchboard.schemas.events import (
    Finish,
    FinishReason,
    MessageStart,
    ReasoningDelta,
    RefusalDelta,
    StreamEvent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStart,
    Usage,
)
from tests.support import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures"


class FakeResponse:
    """Stands in for ``UpstreamResponse``: yields the fixture in the given pieces."""

    def __init__(self, parts: list[bytes]) -> None:
        self.parts = parts

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            yield part


async def replay_anthropic(
    parts: list[bytes],
) -> tuple[list[StreamEvent], anthropic.AnthropicStream]:
    stream = anthropic.AnthropicStream()
    events = [e async for e in anthropic._events(FakeResponse(parts), stream, CallTimings())]  # type: ignore[arg-type]
    return events, stream


async def replay_openai(parts: list[bytes]) -> list[StreamEvent]:
    return [e async for e in openai._stream_events(FakeResponse(parts), CallTimings())]  # type: ignore[arg-type]


def fixture(provider: str, name: str) -> bytes:
    return (FIXTURES / provider / f"{name}.sse").read_bytes()


def split(data: bytes, cuts: list[int]) -> list[bytes]:
    points = sorted({min(c, len(data)) for c in cuts})
    return [data[a:b] for a, b in zip([0, *points], [*points, len(data)], strict=True) if a != b]


# --- Anthropic -------------------------------------------------------------------------


async def test_anthropic_text() -> None:
    events, _ = await replay_anthropic([fixture("anthropic", "text")])
    assert events == [
        MessageStart("msg_01XYZ", "claude-sonnet-5-5"),
        TextDelta("Hello"),
        TextDelta("! How can I help?"),
        Finish(FinishReason.STOP),
        Usage(input=25, output=15),
    ]


async def test_anthropic_tool_use_counts_tool_calls_not_blocks() -> None:
    events, stream = await replay_anthropic([fixture("anthropic", "tool_use")])
    assert events[1:] == [
        TextDelta("Let me check the weather."),
        ToolCallStart(0, "toolu_01T1x1fJ34qAmk2tNTrN7Up6", "get_weather"),  # block 1, tool 0
        ToolCallArgsDelta(0, '{"location": "San Fra'),
        ToolCallArgsDelta(0, 'ncisco, CA", "unit": "fahrenheit"}'),
        Finish(FinishReason.TOOL_CALLS),
        Usage(input=472, output=89),
    ]
    assert stream.passback_content() is None  # no thinking: nothing worth storing


async def test_anthropic_thinking_is_hidden_and_kept_whole_for_passback() -> None:
    events, stream = await replay_anthropic([fixture("anthropic", "thinking_tool_use")])
    assert not any(isinstance(e, ReasoningDelta) for e in events)  # display omitted: empty
    assert events[-1] == Usage(input=600, output=120, reasoning=90)
    saved = stream.passback_content()
    assert saved is not None
    ids, content = saved
    assert ids == ["toolu_01A09q90qw90lq917835lq9"]
    assert [b["type"] for b in content] == ["thinking", "redacted_thinking", "tool_use"]
    assert content[0]["signature"].startswith("EqQBCgIYAhIM")
    assert content[1]["data"].startswith("EmwKAhgBEgy3")
    assert content[2]["input"] == {"location": "Paris"}


async def test_anthropic_parallel_tool_calls() -> None:
    events, _ = await replay_anthropic([fixture("anthropic", "parallel_tool_use")])
    aggregator = Aggregator(completion_id="c", created=0, model="m")
    for event in events:
        aggregator.add(event)
    calls = aggregator.result()["choices"][0]["message"]["tool_calls"]
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls] == [
        ("toolu_a", "get_weather", '{"location": "Paris"}'),
        ("toolu_b", "get_time", '{"timezone": "Europe/Paris"}'),
    ]


async def test_anthropic_refusal_before_any_content() -> None:
    events, stream = await replay_anthropic([fixture("anthropic", "refusal")])
    assert events[1:] == [Finish(FinishReason.CONTENT_FILTER), Usage(input=25, output=0)]
    assert stream.refusal_category == "cyber"


async def test_anthropic_max_tokens() -> None:
    events, _ = await replay_anthropic([fixture("anthropic", "max_tokens")])
    assert Finish(FinishReason.LENGTH) in events


async def test_anthropic_cache_usage_buckets_are_exclusive() -> None:
    events, _ = await replay_anthropic([fixture("anthropic", "cache_usage")])
    usage = events[-1]
    assert usage == Usage(
        input=10, cache_read=3000, cache_write_5m=200, cache_write_1h=1000, output=7
    )
    assert isinstance(usage, Usage)
    assert usage.prompt_total == 4210  # what OpenAI clients see as prompt_tokens


async def test_anthropic_mid_stream_error_event() -> None:
    with pytest.raises(ProviderError) as caught:
        await replay_anthropic([fixture("anthropic", "error_midstream")])
    assert caught.value.kind is ErrorKind.OVERLOADED


async def test_anthropic_truncated_stream_is_a_protocol_error() -> None:
    data = fixture("anthropic", "text")
    with pytest.raises(ProviderError) as caught:
        await replay_anthropic([data[: data.index(b"event: message_stop")]])
    assert caught.value.kind is ErrorKind.PROTOCOL


# --- OpenAI ----------------------------------------------------------------------------


async def test_openai_text_with_usage() -> None:
    events = await replay_openai([fixture("openai", "text_with_usage")])
    assert events == [
        MessageStart("chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT", "gpt-5.4-mini-2026-03-05"),
        TextDelta("Hello"),
        TextDelta(" there"),
        Finish(FinishReason.STOP),
        Usage(input=176, cache_read=1024, output=30, reasoning=12),
    ]


async def test_openai_parallel_tool_calls() -> None:
    events = await replay_openai([fixture("openai", "parallel_tool_calls")])
    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert starts == [
        ToolCallStart(0, "call_a", "get_weather"),
        ToolCallStart(1, "call_b", "get_time"),
    ]
    assert events[-1] == Finish(FinishReason.TOOL_CALLS)


async def test_openai_refusal() -> None:
    events = await replay_openai([fixture("openai", "refusal")])
    assert RefusalDelta("I can't help with that.") in events


# --- any split, same events -------------------------------------------------------------


def _all_fixtures() -> list[tuple[str, str]]:
    return sorted(
        (path.parent.name, path.stem)
        for path in FIXTURES.glob("*/*.sse")
        if path.stem != "error_midstream"
    )


@pytest.mark.parametrize(("provider", "name"), _all_fixtures())
@settings(max_examples=40, deadline=None)
@given(cuts=st.lists(st.integers(min_value=0, max_value=20_000), max_size=25))
async def test_any_split_gives_the_same_events(provider: str, name: str, cuts: list[int]) -> None:
    data = fixture(provider, name)
    if provider == "anthropic":
        whole, _ = await replay_anthropic([data])
        pieces, _ = await replay_anthropic(split(data, cuts))
    else:
        whole = await replay_openai([data])
        pieces = await replay_openai(split(data, cuts))
    assert pieces == whole


def test_every_fixture_says_where_it_came_from() -> None:
    for sse in FIXTURES.glob("*/*.sse"):
        meta = json.loads(Path(str(sse).removesuffix(".sse") + ".meta.json").read_text())
        assert meta["source"] in ("docs", "recorded")


def _recorded() -> list[tuple[str, str]]:
    return sorted((p.parent.name, p.stem) for p in FIXTURES.glob("*/recorded_*.sse"))


@pytest.mark.parametrize(("provider", "name"), _recorded())
async def test_recorded_fixtures_match_their_expectations(provider: str, name: str) -> None:
    meta = json.loads((FIXTURES / provider / f"{name}.meta.json").read_text())
    if provider == "anthropic":
        events, _ = await replay_anthropic([fixture(provider, name)])
    else:
        events = await replay_openai([fixture(provider, name)])
    aggregator = Aggregator(completion_id="c", created=0, model="m")
    for event in events:
        aggregator.add(event)
    result = aggregator.result()
    assert result["choices"][0]["finish_reason"] == meta["expect"]["finish_reason"]
    assert result["usage"]["completion_tokens"] > 0
