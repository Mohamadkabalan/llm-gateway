"""Command-line entry point: ``switchboard <command>``.

Exit codes: 0 success, 1 invalid routing config, 2 invalid environment settings,
3 a database command failed.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import structlog
from pydantic import ValidationError

from switchboard import __version__, admin_cli
from switchboard.observability.logging import configure_logging
from switchboard.routing import ConfigError, load_routing_config
from switchboard.settings import Settings

EXIT_OK = 0
EXIT_INVALID_CONFIG = 1
EXIT_INVALID_SETTINGS = 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="switchboard", description="Switchboard LLM gateway")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="run the gateway data plane")
    validate = commands.add_parser("validate-config", help="validate a routing config and exit")
    validate.add_argument(
        "path", nargs="?", type=Path, help="config file (default: $SWITCHBOARD_ROUTING_CONFIG)"
    )
    admin_cli.add_commands(commands)
    args = parser.parse_args(argv)

    try:
        settings = Settings()
    except ValidationError as exc:
        print(format_settings_error(exc), file=sys.stderr)
        return EXIT_INVALID_SETTINGS

    if args.command == "validate-config":
        return _validate_config(args.path or settings.routing_config)
    if args.command in admin_cli.COMMANDS:
        return admin_cli.run(args, settings)
    return _serve(settings)


def _validate_config(path: Path) -> int:
    try:
        loaded = load_routing_config(path)
    except ConfigError as exc:
        print(exc.render(), file=sys.stderr)
        return EXIT_INVALID_CONFIG
    for warning in loaded.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    cfg = loaded.config
    pricing = f"pricing {loaded.pricing.version}" if loaded.pricing else "no pricing"
    print(
        f"OK {path} (version {loaded.version}, {pricing}): {len(cfg.providers)} providers, "
        f"{len(cfg.models)} models, {len(cfg.aliases)} aliases, "
        f"{len(cfg.rules)} rule{'s' * (len(cfg.rules) != 1)}"
    )
    return EXIT_OK


def _serve(settings: Settings) -> int:
    configure_logging(settings.log_level, settings.log_format)
    log = structlog.get_logger("switchboard.cli")
    try:
        routing = load_routing_config(settings.routing_config)
    except ConfigError as exc:
        log.error(
            "config.invalid",
            source=exc.source,
            problems=[str(issue) for issue in exc.issues],
        )
        print(exc.render(), file=sys.stderr)
        return EXIT_INVALID_CONFIG

    from switchboard.server import serve  # noqa: PLC0415 - keep uvicorn out of other commands

    serve(settings, routing)
    return EXIT_OK


def format_settings_error(exc: ValidationError) -> str:
    lines = ["Invalid environment settings:"]
    for err in exc.errors(include_url=False):
        field = str(err["loc"][0]) if err["loc"] else "?"
        lines.append(f"  - SWITCHBOARD_{field.upper()}: {err['msg']}")
    return "\n".join(lines)
