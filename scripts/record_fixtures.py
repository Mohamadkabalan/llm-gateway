"""Record real provider streams as test fixtures (PLAN.md §15.1).

    uv run --env-file .env python scripts/record_fixtures.py [--only anthropic|openai]

Calls Anthropic and OpenAI directly (not through the gateway) with small requests and
saves each raw SSE response to tests/fixtures/<provider>/recorded_<scenario>.sse, with
a .meta.json holding the date, model, request and what the replayed response should
contain. Message, call and tool ids are replaced with stable placeholders; nothing
else in the stream is changed. The whole set costs well under a cent at list prices
(PLAN.md §3.8), and runs only when invoked.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ANTHROPIC_VERSION = "2026-10-21"
WEATHER_ANTHROPIC: dict[str, Any] = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
WEATHER_OPENAI: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TWO_CITIES = "Use the get_weather tool for Paris and for Rome, in parallel."
THINK_THEN_CALL = (
    "Think about which city is warmer in winter, Oslo or Madrid, "
    "then use get_weather to check the warmer one."
)

ANTHROPIC: dict[str, dict[str, Any]] = {
    "text": {
        "model": "claude-haiku-5-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello in five words."}],
    },
    "tool_use": {
        "model": "claude-haiku-5-5",
        "max_tokens": 400,
        "tools": [WEATHER_ANTHROPIC],
        "messages": [{"role": "user", "content": "Use the get_weather tool for Paris."}],
    },
    "parallel_tool_use": {
        "model": "claude-haiku-5-5",
        "max_tokens": 400,
        "tools": [WEATHER_ANTHROPIC],
        "messages": [{"role": "user", "content": TWO_CITIES}],
    },
    "max_tokens": {
        "model": "claude-haiku-5-5",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Count from 1 to 100."}],
    },
    "thinking_tool_use": {
        "model": "claude-sonnet-5-5",
        "max_tokens": 2000,
        "output_config": {"effort": "high"},
        "tools": [WEATHER_ANTHROPIC],
        "messages": [
            {
                "role": "user",
                "content": THINK_THEN_CALL,
            }
        ],
    },
}
OPENAI: dict[str, dict[str, Any]] = {
    "text_with_usage": {
        "model": "gpt-5.4-mini",
        "messages": [{"role": "user", "content": "Say hello in five words."}],
        "stream_options": {"include_usage": True},
    },
    "tool_calls": {
        "model": "gpt-5.4-mini",
        "tools": [WEATHER_OPENAI],
        "messages": [{"role": "user", "content": "Use the get_weather tool for Paris."}],
        "stream_options": {"include_usage": True},
    },
    "parallel_tool_calls": {
        "model": "gpt-5.4-mini",
        "tools": [WEATHER_OPENAI],
        "messages": [{"role": "user", "content": TWO_CITIES}],
        "stream_options": {"include_usage": True},
    },
    "max_tokens": {
        "model": "gpt-5.4-mini",
        "max_completion_tokens": 16,
        "messages": [{"role": "user", "content": "Count from 1 to 100."}],
        "stream_options": {"include_usage": True},
    },
}
EXPECTED_FINISH = {
    "text": "stop",
    "tool_use": "tool_calls",
    "parallel_tool_use": "tool_calls",
    "max_tokens": "length",
    "thinking_tool_use": "tool_calls",
    "text_with_usage": "stop",
    "tool_calls": "tool_calls",
    "parallel_tool_calls": "tool_calls",
}
_IDS = re.compile(r"\b(msg|toolu|chatcmpl|call|req)_?-?[A-Za-z0-9]{8,}\b")


def scrub(data: bytes) -> bytes:
    """Replace provider ids with stable placeholders, keeping each distinct id distinct."""
    seen: dict[str, str] = {}

    def placeholder(match: re.Match[str]) -> str:
        value = match.group(0)
        prefix = match.group(1)
        return seen.setdefault(value, f"{prefix}_recorded{len(seen) + 1:02d}")

    return _IDS.sub(placeholder, data.decode()).encode()


def record(
    provider: str, name: str, url: str, headers: dict[str, str], body: dict[str, Any]
) -> None:
    body = {**body, "stream": True}
    with (
        httpx.Client(timeout=120) as client,
        client.stream("POST", url, json=body, headers=headers) as response,
    ):
        data = response.read()
    if response.status_code != 200:
        print(f"  {provider}/{name}: HTTP {response.status_code}: {data[:300]!r}", file=sys.stderr)
        return
    stem = ROOT / provider / f"recorded_{name}"
    stem.with_suffix(".sse").write_bytes(scrub(data))
    meta = {
        "source": "recorded",
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model": body["model"],
        "request": body,
        "expect": {"finish_reason": EXPECTED_FINISH[name]},
    }
    if provider == "anthropic":
        meta["api_version"] = ANTHROPIC_VERSION
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {provider}/{name}: {len(data)} bytes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=["anthropic", "openai"])
    args = parser.parse_args(argv)
    done = 0
    if args.only in (None, "anthropic") and (key := os.environ.get("ANTHROPIC_API_KEY")):
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        for name, body in ANTHROPIC.items():
            record("anthropic", name, "https://api.anthropic.com/v1/messages", headers, body)
            done += 1
    if args.only in (None, "openai") and (key := os.environ.get("OPENAI_API_KEY")):
        headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        for name, body in OPENAI.items():
            record("openai", name, "https://api.openai.com/v1/chat/completions", headers, body)
            done += 1
    if not done:
        print(
            "No provider keys set (ANTHROPIC_API_KEY, OPENAI_API_KEY); nothing recorded.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
