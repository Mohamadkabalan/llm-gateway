"""Make sure ``.env`` has the local stack's secrets: ``python3 scripts/ensure_secrets.py .env``.

Adds, only if missing or empty:

* ``SWITCHBOARD_KEY_PEPPER``: the server secret every API-key hash is keyed with;
* ``SWITCHBOARD_DEMO_KEY``: a virtual key (``sb_live_…``) that ``make up`` registers in
  team ``demo``.

Existing values are never changed (a new pepper would invalidate every key). Uses only
the standard library, so it runs before ``uv sync``.
"""

import secrets
import string
import sys
from pathlib import Path

_ALPHABET = string.digits + string.ascii_letters


def _generators() -> dict[str, str]:
    return {
        "SWITCHBOARD_KEY_PEPPER": secrets.token_urlsafe(48),
        "SWITCHBOARD_DEMO_KEY": "sb_live_" + "".join(secrets.choice(_ALPHABET) for _ in range(43)),
    }


def ensure(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    present = {
        line.split("=", 1)[0]: line.split("=", 1)[1].strip()
        for line in lines
        if "=" in line and not line.lstrip().startswith("#")
    }
    added: list[str] = []
    for name, value in _generators().items():
        if present.get(name):
            continue
        lines = [line for line in lines if not line.startswith(f"{name}=")]
        lines.append(f"{name}={value}")
        added.append(name)
    if added:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return added


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ".env")
    for name in ensure(path):
        print(f"Generated {name} in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
