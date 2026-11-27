"""Walks a ``RoutePlan`` until one candidate commits (PLAN.md §2 step 9, §2.1, §5).

For each candidate, in order:

* the circuit breaker is asked first; an open breaker skips the candidate without a call;
* an attempt opens the upstream stream and reads up to the first content event within
  ``first_token`` (that event is the commit point);
* a failure is classified (``policy``): retried on the same target with full-jitter
  backoff when that may help and the provider's retry budget allows, otherwise the walk
  moves to the next candidate if the failure allows a fallback.

Nothing reaches the client before the commit point, so every failure up to it is
invisible to the client. With ``hedge`` on a route, if the primary has not committed
after ``hedge.after``, the remaining candidates start in parallel and the first to
commit wins; the other side is cancelled, which closes its upstream connection.

With ``fallback_on_refusal``, a refusal before the commit point counts as a failure; the
first refusal is held open and served only if no other candidate commits.

The whole walk runs inside the request deadline the caller sets (``asyncio.timeout``);
retries are not started if their backoff would cross it.
"""

import asyncio
import random
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

import structlog

from switchboard.observability.metrics import (
    ATTEMPTS,
    FALLBACKS,
    HEDGES,
    RETRIES,
    RETRY_BUDGET_EXHAUSTED,
)
from switchboard.providers.base import CallContext, EmbeddingResult, ProviderAdapter
from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.reliability.backoff import retry_delay
from switchboard.reliability.breaker import Admission, Breakers
from switchboard.reliability.budget import RatioBudget
from switchboard.reliability.policy import (
    TIMEOUT_RETRIES,
    disposition,
    may_fall_back,
    may_retry,
    stops_the_walk,
)
from switchboard.routing.table import ModelTarget, RoutePlan
from switchboard.schemas.chat import ChatRequest
from switchboard.schemas.embeddings import EmbeddingRequest
from switchboard.schemas.events import (
    Finish,
    FinishReason,
    RefusalDelta,
    StreamEvent,
    is_content,
)

_log = structlog.get_logger("switchboard.reliability.executor")
_HEDGE_MINIMUM = 1  # hedges always allowed per budget window, even at low traffic


@dataclass(slots=True)
class OpenStream:
    """An upstream stream read up to and including its first content event."""

    events: AsyncGenerator[StreamEvent]
    head: list[StreamEvent]

    @property
    def refused(self) -> bool:
        last = self.head[-1] if self.head else None
        return isinstance(last, RefusalDelta) or (
            isinstance(last, Finish) and last.reason is FinishReason.CONTENT_FILTER
        )

    async def aclose(self) -> None:
        await self.events.aclose()


@dataclass(slots=True)
class Committed[T]:
    target: ModelTarget
    ctx: CallContext
    value: T
    refused: bool = False


@dataclass(frozen=True, slots=True)
class AttemptLog:
    model: str
    outcome: str
    upstream_status: int | None = None

    def __str__(self) -> str:
        status = f":{self.upstream_status}" if self.upstream_status else ""
        return f"{self.model}:{self.outcome}{status}"


class ExhaustedError(Exception):
    """No candidate committed. ``error`` is what the client should see."""

    def __init__(self, error: ProviderError, target: ModelTarget | None) -> None:
        super().__init__(error.message)
        self.error = error
        self.target = target  # the last candidate that was tried


@dataclass(slots=True)
class Run:
    """Per-request state shared by every attempt (and both sides of a hedge)."""

    plan: RoutePlan
    request_id: str
    scope: str
    deadline: float  # event-loop time
    attempts: list[AttemptLog] = field(default_factory=list)
    first_ctx: CallContext | None = None  # for the pre-upstream overhead metric
    upstream_calls: int = 0
    last_target: ModelTarget | None = None
    hedged: bool = False

    def new_ctx(self, target: ModelTarget) -> CallContext:
        self.upstream_calls += 1
        self.last_target = target
        ctx = CallContext(
            request_id=self.request_id,
            timeouts=target.timeouts,
            attempt=self.upstream_calls,
            scope=self.scope,
        )
        if self.first_ctx is None:
            self.first_ctx = ctx
        return ctx

    def log(self, target: ModelTarget, outcome: str, status: int | None = None) -> None:
        self.attempts.append(AttemptLog(target.name, outcome, status))
        ATTEMPTS.labels(target.provider, target.name, outcome).inc()


class _TargetFailedError(Exception):
    def __init__(self, error: ProviderError) -> None:
        super().__init__(error.message)
        self.error = error


type Attempt[T] = Callable[[ModelTarget, CallContext], Awaitable[T]]
type Close[T] = Callable[[T], Awaitable[None]]
type IsRefusal[T] = Callable[[T], bool]


class Executor:
    """One per process: owns the breakers and the retry and hedge budgets."""

    def __init__(self, breakers: Breakers, *, rng: random.Random | None = None) -> None:
        self.breakers = breakers
        self._rng = rng or random.Random()  # noqa: S311 (jitter, not crypto)
        self._retry_budgets: dict[str, RatioBudget] = {}
        self._hedge_budgets: dict[str, RatioBudget] = {}

    # --- chat completions --------------------------------------------------------------

    async def open_chat(
        self, run: Run, chat: ChatRequest, adapters: Mapping[str, ProviderAdapter]
    ) -> Committed[OpenStream]:
        async def attempt(target: ModelTarget, ctx: CallContext) -> OpenStream:
            return await open_stream(adapters[target.provider], chat, target, ctx)

        refusal: IsRefusal[OpenStream] | None = (
            _is_refusal if run.plan.fallback_on_refusal else None
        )
        walk = _Walk(self, run, attempt, OpenStream.aclose, refusal)
        candidates = run.plan.candidates
        hedge = run.plan.hedge
        if hedge is None or len(candidates) < 2:
            return await walk.run(candidates)

        budget = self._budget(self._hedge_budgets, run.plan.route or run.plan.requested)
        budget.configure(hedge.max_ratio, _HEDGE_MINIMUM)
        budget.note_request()
        primary = asyncio.create_task(walk.run(candidates[:1]))
        try:
            done, _ = await asyncio.wait({primary}, timeout=hedge.after)
            if not done and budget.try_spend():
                run.hedged = True
                rest = asyncio.create_task(walk.run(candidates[1:]))
                return await self._race(run, primary, rest)
            if not done:
                HEDGES.labels(run.plan.route_label, "skipped").inc()
            try:
                first = await primary
            except ExhaustedError as exc:
                if stops_the_walk(exc.error):
                    raise
                return await walk.run(candidates[1:], after=exc)
            if first.refused:
                return await walk.run(candidates[1:], held=first)
            return first
        finally:
            if not primary.done():
                primary.cancel()
                await asyncio.wait({primary})

    async def _race(  # noqa: PLR0912 (each branch is one race outcome)
        self,
        run: Run,
        primary: asyncio.Task[Committed[OpenStream]],
        hedge: asyncio.Task[Committed[OpenStream]],
    ) -> Committed[OpenStream]:
        """Both sides run; the first to commit (without refusing) wins."""
        pending = {primary, hedge}
        refused: dict[asyncio.Task[Committed[OpenStream]], Committed[OpenStream]] = {}
        failures: list[ExhaustedError] = []
        winner: Committed[OpenStream] | None = None
        try:
            while pending and winner is None:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda t: t is not primary):  # primary first
                    try:
                        result = task.result()
                    except ExhaustedError as exc:
                        failures.append(exc)
                        continue
                    if winner is not None:
                        await result.value.aclose()  # both committed at once
                    elif result.refused:
                        refused[task] = result
                    else:
                        winner = result
                        outcome = "won" if task is hedge else "lost"
                        HEDGES.labels(run.plan.route_label, outcome).inc()
            if winner is None and refused:
                winner = refused.pop(primary if primary in refused else hedge)
            if winner is not None:
                return winner
            raise _pick(failures)
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(pending)
            for held in refused.values():
                if held is not winner:
                    await held.value.aclose()

    def post_commit_failure(self, committed: Committed[OpenStream], error: ProviderError) -> None:
        """A committed stream failed; count it against the target if it was the provider's
        fault (the commit already counted as a success)."""
        if disposition(error).breaker:
            self.breakers.record(committed.target, ok=False, probe=False)

    # --- embeddings --------------------------------------------------------------------

    async def embed(
        self, run: Run, request: EmbeddingRequest, adapters: Mapping[str, ProviderAdapter]
    ) -> Committed[EmbeddingResult]:
        async def attempt(target: ModelTarget, ctx: CallContext) -> EmbeddingResult:
            try:
                async with asyncio.timeout(target.timeouts.first_token):
                    return await adapters[target.provider].embed(request, target, ctx)
            except TimeoutError:
                message = f"no response within {target.timeouts.first_token:g}s"
                raise ProviderError(ErrorKind.TIMEOUT, message) from None

        async def close(_: EmbeddingResult) -> None:
            return None

        return await _Walk(self, run, attempt, close, None).run(run.plan.candidates)

    # --- shared ------------------------------------------------------------------------

    def retry_budget(self, target: ModelTarget) -> RatioBudget:
        budget = self._budget(self._retry_budgets, target.provider)
        budget.configure(target.retry.budget_ratio, target.retry.budget_min_retries)
        return budget

    def retry_delay(
        self,
        run: Run,
        target: ModelTarget,
        error: ProviderError,
        *,
        retries: int,
        timeout_retries: int,
        probe: bool,
    ) -> float | None:
        """Seconds to wait before retrying ``target``, or None to move on."""
        if probe or not may_retry(error) or retries + 1 >= target.retry.max_attempts:
            return None
        if error.kind is ErrorKind.TIMEOUT and timeout_retries >= TIMEOUT_RETRIES:
            return None
        delay = retry_delay(error, retries, target.retry, self._rng)
        if delay is None or asyncio.get_running_loop().time() + delay >= run.deadline:
            return None
        if not self.retry_budget(target).try_spend():
            RETRY_BUDGET_EXHAUSTED.labels(target.provider).inc()
            return None
        return delay

    @staticmethod
    def _budget(budgets: dict[str, RatioBudget], key: str) -> RatioBudget:
        budget = budgets.get(key)
        if budget is None:
            budget = budgets[key] = RatioBudget(0.0, 0)
        return budget


class _Walk[T]:
    """Sequential candidates with retries; one per side of a hedge."""

    def __init__(
        self,
        executor: Executor,
        run: Run,
        attempt: Attempt[T],
        close: Close[T],
        is_refusal: IsRefusal[T] | None,
    ) -> None:
        self._executor = executor
        self._run = run
        self._attempt = attempt
        self._close = close
        self._is_refusal = is_refusal

    async def run(  # noqa: PLR0912 (one branch per walk decision)
        self,
        candidates: Sequence[ModelTarget],
        *,
        after: ExhaustedError | None = None,
        held: Committed[T] | None = None,
    ) -> Committed[T]:
        """Try ``candidates`` in order. ``after`` is the failure of an earlier walk (it
        decides which candidates may follow); ``held`` is an earlier refusal."""
        error = after.error if after is not None else None
        failed = after.target if after is not None else None
        best = error if error is not None and error.kind is not ErrorKind.CIRCUIT_OPEN else None
        if held is not None:
            error, failed = _refusal(held.target), held.target
        try:
            for target in candidates:
                if error is not None and failed is not None:
                    if stops_the_walk(error):
                        break
                    if not may_fall_back(error, failed, target):
                        self._run.log(target, "skipped")
                        continue
                    FALLBACKS.labels(
                        self._run.plan.route_label, failed.name, target.name, error.kind.value
                    ).inc()
                try:
                    committed = await self._try_target(target)
                except _TargetFailedError as exc:
                    error, failed = exc.error, target
                    if error.kind is not ErrorKind.CIRCUIT_OPEN:
                        best = error
                    continue
                if self._is_refusal is not None and self._is_refusal(committed.value):
                    self._run.log(target, "refusal")
                    committed.refused = True
                    if held is None:
                        held = committed
                    else:
                        await self._close(committed.value)
                    error, failed = _refusal(target), target
                    continue
                if held is not None:
                    await self._close(held.value)
                    held = None
                return committed
        except BaseException:
            if held is not None:
                await self._close(held.value)
            raise
        if held is not None:
            return held
        final = best or error or ProviderError(ErrorKind.CIRCUIT_OPEN, "no candidate")
        raise ExhaustedError(final, failed)

    async def _try_target(self, target: ModelTarget) -> Committed[T]:
        executor, run = self._executor, self._run
        retries = timeout_retries = 0
        while True:
            admission = await executor.breakers.admit(target)
            if admission is Admission.DENY:
                run.log(target, ErrorKind.CIRCUIT_OPEN.value)
                message = f"circuit open for {target.name}"
                raise _TargetFailedError(ProviderError(ErrorKind.CIRCUIT_OPEN, message))
            probe = admission is Admission.PROBE
            if retries == 0:
                executor.retry_budget(target).note_request()
            ctx = run.new_ctx(target)
            try:
                value = await self._attempt(target, ctx)
            except ProviderError as error:
                run.log(target, error.kind.value, error.status)
                counts = disposition(error).breaker
                if counts or probe:  # a client-fault answer still shows the target is up
                    executor.breakers.record(target, ok=not counts, probe=probe)
                if error.kind is ErrorKind.AUTH:
                    _log.error("upstream.auth_failed", model=target.name, detail=error.message)
                delay = executor.retry_delay(
                    run,
                    target,
                    error,
                    retries=retries,
                    timeout_retries=timeout_retries,
                    probe=probe,
                )
                if delay is None:
                    raise _TargetFailedError(error) from None
                RETRIES.labels(target.provider, target.name, error.kind.value).inc()
                retries += 1
                timeout_retries += error.kind is ErrorKind.TIMEOUT
                await asyncio.sleep(delay)
                continue
            except asyncio.CancelledError:
                run.log(target, "cancelled")
                raise
            executor.breakers.record(target, ok=True, probe=probe)
            run.log(target, "success")
            return Committed(target, ctx, value)


def _is_refusal(stream: OpenStream) -> bool:
    return stream.refused


def _refusal(target: ModelTarget) -> ProviderError:
    return ProviderError(ErrorKind.REFUSAL, f"{target.name} refused before the first token")


def _pick(failures: Sequence[ExhaustedError]) -> ExhaustedError:
    """The failure to report after a hedge race: a request-level error if there is one,
    else the last provider error, else whatever there is."""
    for failure in failures:
        if stops_the_walk(failure.error):
            return failure
    for failure in reversed(failures):
        if failure.error.kind is not ErrorKind.CIRCUIT_OPEN:
            return failure
    return failures[-1]


async def open_stream(
    adapter: ProviderAdapter, chat: ChatRequest, target: ModelTarget, ctx: CallContext
) -> OpenStream:
    """Open the upstream stream and read up to the first content event (the commit
    point), within ``first_token``. The stream is closed on any failure."""
    events = adapter.stream_chat(chat, target, ctx)
    head: list[StreamEvent] = []
    try:
        async with asyncio.timeout(target.timeouts.first_token):
            async for event in events:
                head.append(event)
                if is_content(event):
                    break
    except TimeoutError:
        await events.aclose()
        message = f"no first token within {target.timeouts.first_token:g}s"
        raise ProviderError(ErrorKind.TIMEOUT, message) from None
    except BaseException:
        await events.aclose()
        raise
    return OpenStream(events, head)
