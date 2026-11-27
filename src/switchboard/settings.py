"""Process settings, read from environment variables prefixed with ``SWITCHBOARD_``.

Settings cover how the process runs (ports, connections, logging). What the gateway
routes lives in the routing config file (see ``switchboard.routing``).
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


LogLevel = Literal["debug", "info", "warning", "error"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWITCHBOARD_",
        extra="ignore",
        frozen=True,
    )

    # Containers must listen on all interfaces; compose binds published ports to 127.0.0.1.
    host: str = "0.0.0.0"  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65535)
    metrics_port: int = Field(
        default=9100, ge=0, le=65535, description="Prometheus /metrics port; 0 disables it."
    )

    routing_config: Path = Path("config/routing.yaml")

    log_level: LogLevel = "info"
    log_format: LogFormat = LogFormat.JSON

    redis_url: RedisDsn | None = None
    database_url: PostgresDsn | None = None

    shutdown_grace_seconds: float = Field(
        default=25.0,
        gt=0,
        le=600,
        description="How long in-flight requests may run after SIGTERM before being cut off.",
    )
    ready_check_timeout_seconds: float = Field(default=1.0, gt=0, le=10)
    ready_cache_seconds: float = Field(
        default=1.0, ge=0, le=60, description="Reuse dependency check results for this long."
    )
