"""Shared test constants and helpers (importable, unlike conftest)."""

import uuid
from pathlib import Path

from switchboard.auth.context import KeyContext, Limits
from switchboard.auth.keys import display_prefix, hash_secret
from switchboard.auth.store import MemoryKeyStore

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL_CONFIG = """\
version: 1
providers:
  fake:
    type: openai
    base_url: http://fake-provider:8000/v1
models:
  fake/fake-chat:
    upstream_model: fake-chat
aliases:
  fake:
    targets: [{ model: fake/fake-chat }]
"""

# --- virtual keys for tests (no database) ----------------------------------------------

TEST_PEPPER = "test-pepper-not-a-secret-0123456789abcdef"
TEST_KEY = "sb_live_" + "T" * 43
TEST_TEAM_ID = uuid.UUID("00000000-0000-7000-8000-000000000001")
NO_LIMITS = Limits()


def key_context(
    secret: str = TEST_KEY,
    *,
    key_limits: Limits = NO_LIMITS,
    team_limits: Limits = NO_LIMITS,
    team_id: uuid.UUID = TEST_TEAM_ID,
    **fields: object,
) -> KeyContext:
    values: dict[str, object] = {
        "key_id": uuid.uuid5(uuid.NAMESPACE_URL, secret),
        "key_name": "test",
        "key_prefix": display_prefix(secret),
        "team_id": team_id,
        "team_name": "test-team",
        "key_limits": key_limits,
        "team_limits": team_limits,
    }
    values.update(fields)
    return KeyContext(**values)  # type: ignore[arg-type]


def key_store(*keys: tuple[str, KeyContext]) -> MemoryKeyStore:
    """A store holding ``(secret, context)`` pairs, hashed with ``TEST_PEPPER``."""
    pairs = keys or ((TEST_KEY, key_context()),)
    return MemoryKeyStore(
        (hash_secret(TEST_PEPPER.encode(), secret), context) for secret, context in pairs
    )
