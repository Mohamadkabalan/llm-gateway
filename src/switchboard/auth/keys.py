"""Key format and hashing.

A key is ``sb_live_`` followed by 43 base62 characters drawn from ``secrets`` (about 256
bits). Only ``HMAC-SHA256(pepper, key)`` is stored. The pepper is a server secret
(``SWITCHBOARD_KEY_PEPPER``), so a copy of the database alone cannot be used to test
guesses. Keys are high-entropy, so a fast keyed hash is the right tool: a password
KDF such as argon2 would add tens of milliseconds to every request for no gain.
"""

import hashlib
import hmac
import secrets
import string

KEY_PREFIX = "sb_live_"
SECRET_LENGTH = 43  # base62 characters: log2(62) * 43 ≈ 256 bits
DISPLAY_LENGTH = len(KEY_PREFIX) + 4  # "sb_live_4f2a", safe to log and show
_ALPHABET = string.digits + string.ascii_letters
_MIN_TOKEN = 20
_MAX_TOKEN = 200


def generate_secret() -> str:
    return KEY_PREFIX + "".join(secrets.choice(_ALPHABET) for _ in range(SECRET_LENGTH))


def hash_secret(pepper: bytes, secret: str) -> bytes:
    return hmac.new(pepper, secret.encode(), hashlib.sha256).digest()


def display_prefix(secret: str) -> str:
    return secret[:DISPLAY_LENGTH]


def plausible(token: str) -> bool:
    """Cheap shape check before hashing: rejects obvious garbage without a lookup."""
    return token.startswith("sb_") and _MIN_TOKEN <= len(token) <= _MAX_TOKEN
