"""The upstream HTTP transport, kept behind one small interface (ADR-0001, ADR-0003).

Adapters call ``UpstreamClient.stream`` and see ``UpstreamResponse`` and
``ProviderError``; no HTTP library leaks into them. The implementation uses aiohttp,
which used about a third of httpx's CPU per streamed chunk in an A/B run, and avoids
httpcore's pool, which does O(connections²) work per request (ADR-0012).

One pooled session per provider, HTTP/1.1. Upstream streams are long-lived, so each
in-flight request holds one connection: ``max_connections`` caps concurrent requests
to a provider per replica, and requests beyond it wait up to ``timeouts.connect``.
"""

import contextlib
import urllib.request
from collections.abc import AsyncIterator, Mapping
from urllib.parse import urlsplit

import aiohttp
from multidict import CIMultiDict

from switchboard import __version__
from switchboard.providers.errors import ErrorKind, ProviderError
from switchboard.routing.config import PoolConfig, Timeouts

USER_AGENT = f"switchboard/{__version__}"


class UpstreamResponse:
    def __init__(self, response: aiohttp.ClientResponse) -> None:
        self._response = response
        self.status = response.status
        self.headers: Mapping[str, str] = response.headers  # case-insensitive

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        """Body bytes as they arrive, in whatever pieces the network delivers."""
        try:
            async for chunk in self._response.content.iter_any():
                yield chunk
        except aiohttp.ClientError as exc:
            raise _classify(exc) from exc

    async def read_capped(self, limit: int) -> bytes:
        """Up to ``limit`` bytes of the body (for error responses)."""
        parts: list[bytes] = []
        size = 0
        async for chunk in self.iter_chunks():
            parts.append(chunk)
            size += len(chunk)
            if size >= limit:
                break
        return b"".join(parts)[:limit]


class UpstreamClient:
    def __init__(self, base_url: str, pool: PoolConfig, headers: Mapping[str, str]) -> None:
        """Must be created inside a running event loop (the app lifespan)."""
        self.base_url = base_url.rstrip("/")
        self.proxy = proxy_for(self.base_url)
        connector = aiohttp.TCPConnector(
            limit=pool.max_connections, limit_per_host=0, keepalive_timeout=pool.keepalive_expiry
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers={"user-agent": USER_AGENT, **headers},
            # Proxies are resolved once in proxy_for(); trust_env would re-read the whole
            # environment on every request.
            trust_env=False,
            raise_for_status=False,
        )

    @contextlib.asynccontextmanager
    async def stream(
        self, path: str, *, body: bytes, headers: Mapping[str, str], timeouts: Timeouts
    ) -> AsyncIterator[UpstreamResponse]:
        """POST ``body`` and yield the response with its body unread.

        Leaving the context before the body is fully read closes the connection, so
        the provider sees the client go away and stops generating.
        """
        try:
            async with self._session.post(
                self.base_url + path,
                data=body,
                headers=headers,
                timeout=client_timeout(timeouts),
                proxy=self.proxy,
            ) as response:
                yield UpstreamResponse(response)
        except aiohttp.ClientError as exc:
            raise _classify(exc) from exc

    async def request(
        self,
        path: str,
        *,
        body: bytes,
        headers: Mapping[str, str],
        timeouts: Timeouts,
        max_bytes: int,
    ) -> tuple[int, Mapping[str, str], bytes]:
        """POST and read the whole response (non-streaming calls such as embeddings)."""
        async with self.stream(path, body=body, headers=headers, timeouts=timeouts) as response:
            data = await response.read_capped(max_bytes + 1)
            if len(data) > max_bytes:
                raise ProviderError(ErrorKind.PROTOCOL, f"response larger than {max_bytes} bytes")
            return response.status, CIMultiDict(response.headers), data

    async def aclose(self) -> None:
        await self._session.close()


def proxy_for(base_url: str, env: Mapping[str, str] | None = None) -> str | None:
    """The proxy for this upstream from HTTP(S)_PROXY / NO_PROXY, as curl and httpx read them.

    Provider URLs and proxy settings are fixed for the life of the process, so this runs
    once per provider rather than once per request.
    """
    parts = urlsplit(base_url)
    proxies = (
        urllib.request.getproxies_environment()
        if env is None
        else {
            k.lower().removesuffix("_proxy"): v
            for k, v in env.items()
            if k.lower().endswith("_proxy")
        }
    )
    proxy = proxies.get(parts.scheme)
    if not proxy or not parts.hostname:
        return None
    no_proxy = proxies.get("no", "")
    if no_proxy and _bypassed(parts.hostname, no_proxy):
        return None
    return proxy


def _bypassed(host: str, no_proxy: str) -> bool:
    for raw in no_proxy.split(","):
        entry = raw.strip().lower()
        if entry == "*":
            return True
        domain = entry.removeprefix(".")
        if domain and (host == domain or host.endswith("." + domain)):
            return True
    return False


def client_timeout(timeouts: Timeouts) -> aiohttp.ClientTimeout:
    """aiohttp timeouts for one call.

    ``sock_read`` applies to every read, including the wait for the first byte, so it is
    the longer of ``first_token`` and ``idle``. The caller enforces the first-token and
    total deadlines precisely; ``connect`` covers waiting for a pooled connection too.
    """
    return aiohttp.ClientTimeout(
        total=None,
        connect=timeouts.connect,
        sock_connect=timeouts.connect,
        sock_read=max(timeouts.first_token, timeouts.idle),
    )


def _classify(exc: aiohttp.ClientError) -> ProviderError:
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, aiohttp.ConnectionTimeoutError | aiohttp.ClientConnectorError):
        kind = ErrorKind.CONNECT  # nothing reached the provider
    elif isinstance(exc, aiohttp.ServerTimeoutError):
        kind = ErrorKind.TIMEOUT
    else:  # dropped connection, truncated body, malformed HTTP
        kind = ErrorKind.PROTOCOL
    return ProviderError(kind, message)
