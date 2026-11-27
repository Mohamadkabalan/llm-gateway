"""The executor's walk over a plan, against a scripted in-memory adapter.

Each upstream model has a script of outcomes, consumed one per call: an error kind,
``"hang"``, ``"slow:<seconds>"``, ``"refusal"`` or ``"ok"`` (also the default).
"""

import asyncio
import random
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from switchboard.providers.base import CallContext, EmbeddingResult
from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.reliability.breaker import Breakers
from switchboard.reliability.executor import Committed, Executor, ExhaustedError, OpenStream, Run
from switchboard.routing import parse_routing_config
from switchboard.routing.config import ModelKind
from switchboard.routing.table import ModelTarget, RouteRequest, RoutingTable
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.embeddings import EmbeddingRequest
from switchboard.schemas.events import (
    Finish,
    FinishReason,
    MessageStart,
    RefusalDelta,
    StreamEvent,
    TextDelta,
)

CONFIG = """\
version: 1
defaults:
  timeouts: { connect: 1s, first_token: 500ms, idle: 1s, total: 10s }
  retry: { max_attempts: 2, base_delay: 1ms, max_delay: 2ms, honor_retry_after_up_to: 50ms }
  breaker: { min_requests: 3, cooldown: 30s }
providers:
  p: { type: openai, base_url: "http://p/v1" }
models:
  p/a: { upstream_model: a, capabilities: { max_input_tokens: 100 } }
  p/b: { upstream_model: b, capabilities: { max_input_tokens: 50 } }
  p/c: { upstream_model: c, capabilities: { max_input_tokens: 1000 } }
  p/e1: { upstream_model: e1, kind: embedding }
  p/e2: { upstream_model: e2, kind: embedding }
aliases:
  abc: { targets: [{ model: p/a }], fallbacks: [p/b, p/c] }
  ab: { targets: [{ model: p/a }], fallbacks: [p/b] }
  refusing: { targets: [{ model: p/a }], fallbacks: [p/b], fallback_on_refusal: true }
  hedged:
    targets: [{ model: p/a }]
    fallbacks: [p/b]
    hedge: { after: 50ms, max_ratio: 1.0 }
  hedged-capped:
    targets: [{ model: p/a }]
    fallbacks: [p/b]
    hedge: { after: 50ms, max_ratio: 0.1 }
  vectors: { targets: [{ model: p/e1 }], fallbacks: [p/e2] }
"""
CHAT = ChatRequest.model_validate({"model": "x", "messages": [{"role": "user", "content": "hi"}]})


class ScriptedAdapter:
    name = "p"

    def __init__(self, **scripts: list[Any]) -> None:
        self.scripts = {model: list(outcomes) for model, outcomes in scripts.items()}
        self.calls: list[str] = []
        self.closed: list[str] = []

    def _next(self, model: str) -> Any:
        self.calls.append(model)
        script = self.scripts.get(model)
        return script.pop(0) if script else "ok"

    async def stream_chat(
        self, request: ChatRequest, target: ModelTarget, ctx: CallContext
    ) -> AsyncGenerator[StreamEvent]:
        outcome = self._next(target.upstream_model)
        try:
            if isinstance(outcome, ProviderError):
                raise outcome
            if outcome == "hang":
                await asyncio.sleep(3600)
            if isinstance(outcome, str) and outcome.startswith("slow:"):
                await asyncio.sleep(float(outcome.removeprefix("slow:")))
            yield MessageStart(upstream_id=None, served_model=target.upstream_model)
            if outcome == "refusal":
                yield RefusalDelta("I can't help with that.")
                yield Finish(FinishReason.CONTENT_FILTER)
                return
            yield TextDelta(f"from {target.upstream_model}")
            yield Finish(FinishReason.STOP)
        finally:
            self.closed.append(target.upstream_model)

    async def embed(
        self, request: EmbeddingRequest, target: ModelTarget, ctx: CallContext
    ) -> EmbeddingResult:
        outcome = self._next(target.upstream_model)
        if isinstance(outcome, ProviderError):
            raise outcome
        return EmbeddingResult(body={"model": target.upstream_model, "data": []}, prompt_tokens=1)

    async def aclose(self) -> None:
        return None


def err(kind: ErrorKind, **kwargs: Any) -> ProviderError:
    return ProviderError(kind, kind.value, **kwargs)


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable(parse_routing_config(CONFIG, env={}))


def new_run(table: RoutingTable, alias: str, *, deadline: float = 10.0) -> Run:
    kind = ModelKind.EMBEDDING if alias == "vectors" else ModelKind.CHAT
    request = RouteRequest(model=alias, kind=kind, chat=CHAT if kind is ModelKind.CHAT else None)
    plan = table.plan(request)
    loop_time = asyncio.get_running_loop().time()
    return Run(plan=plan, request_id="r", scope="default", deadline=loop_time + deadline)


def executor() -> Executor:
    return Executor(Breakers(), rng=random.Random(0))


async def open_chat(
    table: RoutingTable, alias: str, adapter: ScriptedAdapter, ex: Executor | None = None
) -> tuple[Committed[OpenStream], Run]:
    run = new_run(table, alias)
    committed = await (ex or executor()).open_chat(run, CHAT, {"p": adapter})
    await committed.value.aclose()
    return committed, run


def log(run: Run) -> list[str]:
    return [str(entry) for entry in run.attempts]


# --- retries and fallback ----------------------------------------------------------------


async def test_first_try_success(table: RoutingTable) -> None:
    adapter = ScriptedAdapter()
    committed, run = await open_chat(table, "abc", adapter)
    assert committed.target.name == "p/a"
    assert log(run) == ["p/a:success"]
    assert run.upstream_calls == 1


async def test_retryable_error_is_retried_on_the_same_target(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.OVERLOADED, status=503)])
    committed, run = await open_chat(table, "abc", adapter)
    assert committed.target.name == "p/a"
    assert log(run) == ["p/a:overloaded:503", "p/a:success"]


async def test_retries_exhausted_then_fallback(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.OVERLOADED)] * 2)
    committed, _ = await open_chat(table, "abc", adapter)
    assert committed.target.name == "p/b"
    assert adapter.calls == ["a", "a", "b"]


@pytest.mark.parametrize(
    ("kind", "calls"),
    [
        (ErrorKind.QUOTA, ["a", "b"]),  # no retry, falls back
        (ErrorKind.AUTH, ["a", "b"]),
        (ErrorKind.INVALID_REQUEST, ["a", "b"]),
        (ErrorKind.CONNECT, ["a", "a", "b"]),
        (ErrorKind.PROTOCOL, ["a", "a", "b"]),
        (ErrorKind.TIMEOUT, ["a", "a", "b"]),
    ],
)
async def test_kinds_that_fall_back(table: RoutingTable, kind: ErrorKind, calls: list[str]) -> None:
    adapter = ScriptedAdapter(a=[err(kind)] * 3)
    committed, _ = await open_chat(table, "abc", adapter)
    assert committed.target.name == "p/b"
    assert adapter.calls == calls


async def test_bad_request_stops_everything(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.BAD_REQUEST, status=400)])
    with pytest.raises(ExhaustedError) as caught:
        await open_chat(table, "abc", adapter)
    assert caught.value.error.kind is ErrorKind.BAD_REQUEST
    assert adapter.calls == ["a"]


async def test_context_length_skips_smaller_models(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.CONTEXT_LENGTH)])
    committed, run = await open_chat(table, "abc", adapter)
    assert committed.target.name == "p/c"  # 1000 > 100; p/b (50) is skipped
    assert log(run) == ["p/a:context_length", "p/b:skipped", "p/c:success"]


async def test_rate_limit_retry_after_within_the_cap_is_retried(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.RATE_LIMITED, retry_after=0.01)])
    committed, _ = await open_chat(table, "abc", adapter)
    assert adapter.calls == ["a", "a"]
    assert committed.target.name == "p/a"


async def test_rate_limit_retry_after_past_the_cap_falls_back(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.RATE_LIMITED, retry_after=5)])
    committed, _ = await open_chat(table, "abc", adapter)
    assert adapter.calls == ["a", "b"]
    assert committed.target.name == "p/b"


async def test_timeouts_are_retried_only_once(table: RoutingTable) -> None:
    text = CONFIG.replace("max_attempts: 2", "max_attempts: 5")
    table = RoutingTable(parse_routing_config(text, env={}))
    adapter = ScriptedAdapter(a=["hang"] * 5)
    committed, run = await open_chat(table, "abc", adapter)
    assert adapter.calls[:3] == ["a", "a", "b"]
    assert log(run)[:2] == ["p/a:timeout", "p/a:timeout"]
    assert committed.target.name == "p/b"
    assert adapter.closed.count("a") == 2  # each timed-out stream was closed


async def test_x_should_retry_false_is_obeyed(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.OVERLOADED, should_retry=False)])
    await open_chat(table, "abc", adapter)
    assert adapter.calls == ["a", "b"]


async def test_all_candidates_failing_reports_the_last_upstream_error(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(
        a=[err(ErrorKind.OVERLOADED)] * 2,
        b=[err(ErrorKind.QUOTA)],
        c=[err(ErrorKind.RATE_LIMITED, retry_after=9)],
    )
    with pytest.raises(ExhaustedError) as caught:
        await open_chat(table, "abc", adapter)
    assert caught.value.error.kind is ErrorKind.RATE_LIMITED
    assert caught.value.target is not None
    assert caught.value.target.name == "p/c"


async def test_retry_budget_exhaustion_moves_on(table: RoutingTable) -> None:
    no_budget = "max_attempts: 2, budget_ratio: 0, budget_min_retries: 0,"
    text = CONFIG.replace("max_attempts: 2,", no_budget)
    table = RoutingTable(parse_routing_config(text, env={}))
    adapter = ScriptedAdapter(a=[err(ErrorKind.OVERLOADED)])
    await open_chat(table, "abc", adapter)
    assert adapter.calls == ["a", "b"]


async def test_no_retry_whose_backoff_would_cross_the_deadline(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.RATE_LIMITED, retry_after=0.04)])
    run = new_run(table, "abc", deadline=0.03)
    committed = await executor().open_chat(run, CHAT, {"p": adapter})
    await committed.value.aclose()
    assert adapter.calls == ["a", "b"]


# --- circuit breakers --------------------------------------------------------------------


async def test_an_open_breaker_skips_its_target_without_a_call(table: RoutingTable) -> None:
    ex = executor()
    adapter = ScriptedAdapter(a=[err(ErrorKind.QUOTA)] * 3)
    for _ in range(3):  # three counted failures (min_requests 3) open p/a's breaker
        await open_chat(table, "ab", adapter, ex)
    adapter.calls.clear()
    committed, run = await open_chat(table, "ab", adapter, ex)
    assert adapter.calls == ["b"]
    assert log(run) == ["p/a:circuit_open", "p/b:success"]
    assert committed.target.name == "p/b"


async def test_client_faults_do_not_count_against_the_breaker(table: RoutingTable) -> None:
    ex = executor()
    adapter = ScriptedAdapter(a=[err(ErrorKind.INVALID_REQUEST)] * 5)
    for _ in range(5):
        await open_chat(table, "ab", adapter, ex)
    adapter.calls.clear()
    await open_chat(table, "ab", adapter, ex)
    assert adapter.calls == ["a"]  # still tried: the breaker stayed closed


# --- refusals ----------------------------------------------------------------------------


async def test_refusal_is_served_without_fallback_on_refusal(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=["refusal"])
    committed, _ = await open_chat(table, "ab", adapter)
    assert committed.target.name == "p/a"
    assert committed.value.refused
    assert adapter.calls == ["a"]


async def test_fallback_on_refusal_tries_the_next_candidate(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=["refusal"])
    committed, run = await open_chat(table, "refusing", adapter)
    assert committed.target.name == "p/b"
    assert log(run) == ["p/a:success", "p/a:refusal", "p/b:success"]
    assert "a" in adapter.closed  # the held refusal was closed


async def test_held_refusal_is_served_when_every_fallback_fails(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=["refusal"], b=[err(ErrorKind.QUOTA)])
    committed, _ = await open_chat(table, "refusing", adapter)
    assert committed.target.name == "p/a"
    assert committed.refused


# --- hedging -----------------------------------------------------------------------------


async def test_hedge_wins_when_the_primary_is_slow(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=["hang"])
    committed, run = await open_chat(table, "hedged", adapter)
    assert committed.target.name == "p/b"
    assert run.hedged
    assert adapter.calls == ["a", "b"]
    assert "a" in adapter.closed  # the loser was cancelled and its stream closed
    assert "p/a:cancelled" in log(run)


async def test_primary_wins_a_hedge_it_answers_first(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=["slow:0.08"], b=["slow:2"])
    committed, run = await open_chat(table, "hedged", adapter)
    assert committed.target.name == "p/a"
    assert run.hedged
    assert "b" in adapter.closed


async def test_a_fast_primary_failure_falls_back_without_hedging(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(a=[err(ErrorKind.QUOTA)])
    committed, run = await open_chat(table, "hedged", adapter)
    assert committed.target.name == "p/b"
    assert not run.hedged


async def test_hedges_are_capped_by_ratio(table: RoutingTable) -> None:
    ex = executor()
    _, run1 = await open_chat(table, "hedged-capped", ScriptedAdapter(a=["slow:0.1"]), ex)
    assert run1.hedged  # one hedge per window is always allowed
    second, run2 = await open_chat(table, "hedged-capped", ScriptedAdapter(a=["slow:0.1"]), ex)
    assert not run2.hedged  # 10% of two requests rounds down to none
    assert second.target.name == "p/a"


async def test_hedge_race_both_failing_reports_an_error(table: RoutingTable) -> None:
    class SlowQuota(ScriptedAdapter):
        """Every call fails with QUOTA after 100 ms, so the hedge starts first."""

        async def stream_chat(
            self, request: ChatRequest, target: ModelTarget, ctx: CallContext
        ) -> AsyncGenerator[StreamEvent]:
            self.calls.append(target.upstream_model)
            await asyncio.sleep(0.1)
            if self.calls:  # always true; the yield below keeps this an async generator
                raise err(ErrorKind.QUOTA)
            yield MessageStart(upstream_id=None, served_model=None)  # pragma: no cover

    adapter = SlowQuota()
    with pytest.raises(ExhaustedError) as caught:
        await open_chat(table, "hedged", adapter)
    assert caught.value.error.kind is ErrorKind.QUOTA
    assert adapter.calls == ["a", "b"]


# --- embeddings --------------------------------------------------------------------------


async def test_embeddings_fall_back_too(table: RoutingTable) -> None:
    adapter = ScriptedAdapter(e1=[err(ErrorKind.OVERLOADED)] * 2)
    run = new_run(table, "vectors")
    request = EmbeddingRequest(model="vectors", input="hi")
    committed = await executor().embed(run, request, {"p": adapter})
    assert committed.target.name == "p/e2"
    assert adapter.calls == ["e1", "e1", "e2"]
