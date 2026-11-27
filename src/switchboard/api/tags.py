"""Request tags from the ``X-Switchboard-Tags`` header, for routing rules (PLAN.md §4).

A comma-separated list, e.g. ``X-Switchboard-Tags: pii-sensitive, batch``, added to
the virtual key's own tags.
"""

from fastapi import Request

from switchboard.api.errors import GatewayError

HEADER = "x-switchboard-tags"
MAX_TAGS = 32
MAX_TAG_LENGTH = 64


def request_tags(request: Request) -> frozenset[str]:
    raw = request.headers.get(HEADER)
    if not raw:
        return frozenset()
    tags = [tag.strip() for tag in raw.split(",")]
    tags = [tag for tag in tags if tag]
    if len(tags) > MAX_TAGS or any(len(tag) > MAX_TAG_LENGTH for tag in tags):
        raise GatewayError(
            400,
            f"{HEADER} allows at most {MAX_TAGS} tags of up to {MAX_TAG_LENGTH} characters.",
            code="invalid_tags",
        )
    return frozenset(tags)
