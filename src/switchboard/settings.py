"""Process settings, read from environment variables prefixed with ``SWITCHBOARD_``.

Settings cover how the process runs (ports, connections, logging). What the gateway
routes lives in the routing config file (see ``switchboard.routing``).
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, HttpUrl, PostgresDsn, RedisDsn, SecretStr, model_validator
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
    config_watch_seconds: float = Field(
        default=1.0,
        ge=0,
        le=3600,
        description="Check the routing config file for changes this often and reload it "
        "when it changes; 0 turns the watch off (SIGHUP still reloads).",
    )

    log_level: LogLevel = "info"
    log_format: LogFormat = LogFormat.JSON

    key_pepper: SecretStr | None = Field(
        default=None,
        description="Server secret mixed into every API key hash (HMAC-SHA256). Required "
        "when a database is configured; changing it invalidates every key.",
    )
    key_cache_ttl_seconds: float = Field(
        default=30.0,
        ge=0,
        le=3600,
        description="How long a looked-up key is reused; revocations are pushed sooner.",
    )
    budget_on_redis_failure: Literal["open", "closed"] = Field(
        default="open",
        description="With Redis unavailable at admission: serve requests anyway (open) or "
        "refuse those with a budget (closed). Rate limits always fail open.",
    )
    alert_webhook_url: HttpUrl | None = Field(
        default=None, description="Soft budget alerts are POSTed here (Slack-compatible JSON)."
    )
    usage_queue_size: int = Field(default=10_000, ge=1)
    usage_batch_rows: int = Field(default=500, ge=1)
    usage_flush_seconds: float = Field(default=0.25, gt=0, le=60)
    migrations_dir: Path = Path("migrations")
    max_request_bytes: int = Field(
        default=8 * 1024 * 1024, ge=1024, description="Larger request bodies get a 413."
    )

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

    @model_validator(mode="after")
    def _pepper_with_database(self) -> Self:
        if self.database_url is not None and self.key_pepper is None:
            raise ValueError(
                "KEY_PEPPER must be set when DATABASE_URL is: API keys are stored as "
                "HMAC hashes keyed by it (generate one with `make secrets`)"
            )
        if self.key_pepper is not None and len(self.key_pepper.get_secret_value()) < 32:
            raise ValueError("KEY_PEPPER must be at least 32 characters")
        return self
