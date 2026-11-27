"""Run the gateway under uvicorn with graceful shutdown.

Shutdown sequence on SIGTERM/SIGINT:

1. ``Lifecycle.start_draining()``: ``/ready`` starts answering 503.
2. uvicorn stops accepting connections and waits up to ``shutdown_grace_seconds``
   for open requests (including streams) to finish.
3. Remaining tasks are cancelled, then the app lifespan closes pools and flushes state.
"""

import math
from types import FrameType

import uvicorn

from switchboard.app import create_app
from switchboard.lifecycle import Lifecycle
from switchboard.routing import LoadedConfig
from switchboard.settings import Settings


class GatewayServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, lifecycle: Lifecycle) -> None:
        super().__init__(config)
        self._lifecycle = lifecycle

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        # Runs inside a signal handler: only flip flags here (no logging, no I/O).
        self._lifecycle.start_draining()
        super().handle_exit(sig, frame)


def serve(settings: Settings, routing: LoadedConfig) -> None:
    lifecycle = Lifecycle()
    config = uvicorn.Config(
        create_app(settings, routing, lifecycle=lifecycle),
        host=settings.host,
        port=settings.port,
        loop="uvloop",
        http="httptools",
        lifespan="on",
        log_config=None,  # logging is configured by switchboard.observability.logging
        access_log=False,  # replaced by RequestContextMiddleware's access log
        server_header=False,
        timeout_graceful_shutdown=math.ceil(settings.shutdown_grace_seconds),
    )
    GatewayServer(config, lifecycle).run()
