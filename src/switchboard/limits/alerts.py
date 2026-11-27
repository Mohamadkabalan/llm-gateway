"""Soft budget alerts: one per scope and month (the ``SET NX`` in settlement).

An alert is a warning log line (``budget.soft_limit_reached``), the metric
``switchboard_budget_alerts_total{scope}``, a ``budget_alerts`` row in Postgres, and,
if ``SWITCHBOARD_ALERT_WEBHOOK_URL`` is set, a Slack-compatible JSON POST
(``{"text": ...}``). Delivery runs in the background; a failing webhook or database is
logged and never affects requests.
"""

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import aiohttp
import asyncpg
import structlog

from switchboard.accounting.pricing import NANO
from switchboard.auth.context import KeyContext
from switchboard.limits.limiter import Settled, month_bounds
from switchboard.observability.metrics import BUDGET_ALERTS

_log = structlog.get_logger("switchboard.limits.alerts")
_WEBHOOK_TIMEOUT = aiohttp.ClientTimeout(total=5)


class Alerts:
    def __init__(
        self, *, pool: asyncpg.Pool[asyncpg.Record] | None, webhook_url: str | None
    ) -> None:
        self._pool = pool
        self._webhook_url = webhook_url
        self._session: aiohttp.ClientSession | None = None
        self._pending: set[asyncio.Task[None]] = set()

    def __call__(self, settled: Settled, key: KeyContext, period: str) -> None:
        budget = settled.budget
        limit = budget.limits.soft_usd or Decimal(0)
        spent = Decimal(settled.spent_nano) / NANO
        who = f"key {key.key_name} ({key.key_prefix}…)" if budget.scope == "key" else "team"
        text = (
            f"Switchboard: {who} of team {key.team_name} has spent ${spent:.4f} this month, "
            f"past its soft budget of ${limit:.2f}."
        )
        BUDGET_ALERTS.labels(budget.scope).inc()
        _log.warning(
            "budget.soft_limit_reached",
            scope=budget.scope,
            scope_id=str(budget.scope_id),
            team=key.team_name,
            key=key.key_prefix,
            period=period,
            spent_usd=str(spent),
            soft_limit_usd=str(limit),
        )
        self._spawn(self._record(settled, period, spent, limit))
        if self._webhook_url:
            self._spawn(self._post(text, settled, key, period))

    async def aclose(self) -> None:
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=5.0)
        if self._session is not None:
            await self._session.close()

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _record(self, settled: Settled, period: str, spent: Decimal, limit: Decimal) -> None:
        if self._pool is None:
            return
        start, _ = month_bounds(period)
        try:
            await self._pool.execute(
                "INSERT INTO budget_alerts (scope, scope_id, period, threshold, spent_usd, "
                "limit_usd) VALUES ($1, $2, $3, 'soft', $4, $5) ON CONFLICT DO NOTHING",
                settled.budget.scope,
                settled.budget.scope_id,
                start.date(),
                spent,
                limit,
            )
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _log.warning("budget.alert_not_recorded", detail=str(exc))

    async def _post(self, text: str, settled: Settled, key: KeyContext, period: str) -> None:
        assert self._webhook_url is not None  # noqa: S101
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=_WEBHOOK_TIMEOUT)
        payload = {
            "text": text,
            "switchboard": {
                "event": "budget.soft_limit_reached",
                "scope": settled.budget.scope,
                "scope_id": str(settled.budget.scope_id),
                "team": key.team_name,
                "period": period,
                "spent_usd": str(Decimal(settled.spent_nano) / NANO),
                "at": datetime.now(UTC).isoformat(),
            },
        }
        try:
            async with self._session.post(self._webhook_url, json=payload) as response:
                if response.status >= 300:
                    _log.warning("budget.webhook_failed", status=response.status)
        except (aiohttp.ClientError, TimeoutError) as exc:
            _log.warning("budget.webhook_failed", detail=str(exc))
