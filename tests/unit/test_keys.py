"""Key format and hashing, and the caching ``KeyResolver``."""

import asyncio
import string
from datetime import UTC, datetime, timedelta

import pytest

from switchboard.auth.context import KeyContext, combine_allowed
from switchboard.auth.keys import (
    KEY_PREFIX,
    SECRET_LENGTH,
    display_prefix,
    generate_secret,
    hash_secret,
    plausible,
)
from switchboard.auth.resolver import AuthError, KeyResolver, invalidation_message
from switchboard.auth.store import MemoryKeyStore
from tests.support import TEST_KEY, TEST_PEPPER, TEST_TEAM_ID, key_context, key_store

PEPPER = TEST_PEPPER.encode()


def test_generated_keys_have_the_documented_shape() -> None:
    secrets = {generate_secret() for _ in range(200)}
    assert len(secrets) == 200
    for secret in secrets:
        assert secret.startswith(KEY_PREFIX)
        body = secret.removeprefix(KEY_PREFIX)
        assert len(body) == SECRET_LENGTH
        assert set(body) <= set(string.ascii_letters + string.digits)
        assert plausible(secret)
    assert display_prefix(TEST_KEY) == "sb_live_TTTT"


def test_hash_depends_on_the_pepper() -> None:
    assert hash_secret(PEPPER, TEST_KEY) == hash_secret(PEPPER, TEST_KEY)
    assert hash_secret(PEPPER, TEST_KEY) != hash_secret(b"another pepper", TEST_KEY)
    assert len(hash_secret(PEPPER, TEST_KEY)) == 32


@pytest.mark.parametrize("token", ["", "sk-abc", "sb_", "x" * 50, "sb_" + "x" * 300])
def test_implausible_tokens_are_rejected_without_a_lookup(token: str) -> None:
    assert not plausible(token)


def test_allowed_models_must_be_allowed_by_key_and_team() -> None:
    assert combine_allowed(None, None) is None
    assert combine_allowed(["a", "b"], None) == frozenset({"a", "b"})
    assert combine_allowed(None, ["b"]) == frozenset({"b"})
    assert combine_allowed(["a", "b"], ["b", "c"]) == frozenset({"b"})


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def resolver(store: MemoryKeyStore, clock: Clock | None = None) -> KeyResolver:
    return KeyResolver(store, PEPPER, ttl=30, negative_ttl=5, stale_ttl=300, clock=clock or Clock())


async def test_lookups_are_cached_for_the_ttl() -> None:
    store, clock = key_store(), Clock()
    keys = resolver(store, clock)
    for _ in range(5):
        assert (await keys.resolve(TEST_KEY)).key_name == "test"
    assert store.lookups == 1
    clock.now += 31
    await keys.resolve(TEST_KEY)
    assert store.lookups == 2


async def test_unknown_keys_are_cached_briefly() -> None:
    store, clock = MemoryKeyStore(), Clock()
    keys = resolver(store, clock)
    for _ in range(3):
        with pytest.raises(AuthError) as caught:
            await keys.resolve(TEST_KEY)
        assert (caught.value.status, caught.value.code) == (401, "invalid_api_key")
    assert store.lookups == 1
    clock.now += 6
    with pytest.raises(AuthError):
        await keys.resolve(TEST_KEY)
    assert store.lookups == 2


async def test_malformed_tokens_never_reach_the_store() -> None:
    store = key_store()
    with pytest.raises(AuthError):
        await resolver(store).resolve("not-a-key")
    assert store.lookups == 0


@pytest.mark.parametrize(
    ("context", "code", "message"),
    [
        (key_context(revoked=True), "invalid_api_key", "revoked"),
        (
            key_context(expires_at=datetime.now(UTC) - timedelta(seconds=1)),
            "expired_api_key",
            "expired",
        ),
    ],
)
async def test_revoked_and_expired_keys_are_401(
    context: KeyContext, code: str, message: str
) -> None:
    with pytest.raises(AuthError) as caught:
        await resolver(key_store((TEST_KEY, context))).resolve(TEST_KEY)
    assert caught.value.status == 401
    assert caught.value.code == code
    assert message in caught.value.message


class FlakyStore:
    """Answers once, then fails like a database that went away."""

    def __init__(self, context: KeyContext) -> None:
        self.context = context
        self.calls = 0

    async def lookup(self, key_hash: bytes) -> KeyContext | None:
        self.calls += 1
        if self.calls > 1:
            raise ConnectionError("database unreachable")
        return self.context


async def test_a_failing_store_serves_stale_entries_then_503() -> None:
    clock = Clock()
    keys = KeyResolver(FlakyStore(key_context()), PEPPER, ttl=30, stale_ttl=300, clock=clock)
    await keys.resolve(TEST_KEY)
    clock.now += 60  # expired, but within the stale window
    assert (await keys.resolve(TEST_KEY)).key_name == "test"
    clock.now += 400  # past it
    with pytest.raises(AuthError) as caught:
        await keys.resolve(TEST_KEY)
    assert (caught.value.status, caught.value.code) == (503, "auth_unavailable")


async def test_unseen_key_with_a_failing_store_is_503() -> None:
    store = FlakyStore(key_context())
    store.calls = 1  # the next call fails
    with pytest.raises(AuthError) as caught:
        await KeyResolver(store, PEPPER).resolve(TEST_KEY)
    assert caught.value.status == 503


async def test_invalidation_messages_drop_entries() -> None:
    store = key_store()
    keys = resolver(store)
    await keys.resolve(TEST_KEY)
    keys.apply(invalidation_message(key_hash=hash_secret(PEPPER, TEST_KEY)))
    await keys.resolve(TEST_KEY)
    assert store.lookups == 2
    keys.apply(invalidation_message(team_id=TEST_TEAM_ID))
    await keys.resolve(TEST_KEY)
    assert store.lookups == 3
    keys.apply(b"not json")  # unreadable: clear everything to be safe
    await keys.resolve(TEST_KEY)
    assert store.lookups == 4


async def test_revocation_applies_as_soon_as_the_message_arrives() -> None:
    context = key_context()
    store = key_store((TEST_KEY, context))
    keys = resolver(store)
    await keys.resolve(TEST_KEY)
    key_hash = hash_secret(PEPPER, TEST_KEY)
    store.keys[key_hash] = key_context(revoked=True)
    await keys.resolve(TEST_KEY)  # still cached: no message yet
    keys.apply(invalidation_message(key_hash=key_hash))
    with pytest.raises(AuthError, match="revoked"):
        await keys.resolve(TEST_KEY)


class SlowStore:
    def __init__(self) -> None:
        self.calls = 0

    async def lookup(self, key_hash: bytes) -> KeyContext | None:
        self.calls += 1
        await asyncio.sleep(0.05)
        return key_context()


async def test_concurrent_misses_share_one_lookup() -> None:
    store = SlowStore()
    keys = KeyResolver(store, PEPPER)
    results = await asyncio.gather(*(keys.resolve(TEST_KEY) for _ in range(20)))
    assert {r.key_name for r in results} == {"test"}
    assert store.calls == 1
