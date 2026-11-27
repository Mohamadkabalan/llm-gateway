"""Liveness and readiness probes.

* ``/health``: the process is up and its event loop answers. It never checks
  dependencies, so an orchestrator doesn't restart the gateway during a Redis blip.
* ``/ready``: this instance should receive traffic. That means it is not draining,
  its config is loaded and Redis and Postgres answer. Returns 503 otherwise.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from switchboard.api.errors import ORJSONResponse
from switchboard.state import AppState, get_state

router = APIRouter(include_in_schema=False)


@router.get("/health")
async def health() -> ORJSONResponse:
    return ORJSONResponse({"status": "ok"})


@router.get("/ready")
async def ready(state: Annotated[AppState, Depends(get_state)]) -> ORJSONResponse:
    body: dict[str, object] = {"config_version": state.config.current.version}
    if state.lifecycle.draining:
        return ORJSONResponse({"status": "draining", **body}, status_code=503)

    results = await state.readiness.check()
    ok = all(result.ok for result in results.values())
    body["status"] = "ready" if ok else "not_ready"
    body["checks"] = {
        name: {"ok": r.ok, "detail": r.detail, "latency_ms": r.latency_ms}
        for name, r in results.items()
    }
    return ORJSONResponse(body, status_code=200 if ok else 503)
