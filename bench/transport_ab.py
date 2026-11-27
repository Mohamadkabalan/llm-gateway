"""Client-side CPU per streamed read: httpx vs aiohttp against the same fake provider.

    uv run python -m bench.transport_ab --streams 20,200 --seconds 12 --rounds 2

It starts the fake provider (2 workers) on a free port, then for each concurrency
level runs the two clients alternately in fresh processes, so drift on the machine
affects both equally. Each client streams chat completions in a loop for
``--seconds`` and reports reads per second and process CPU per read. This is the
evidence for ADR-0012; it measures the HTTP client only, not the gateway.
"""

import argparse
import asyncio
import os
import resource
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable

import orjson
import uvloop

BODY = orjson.dumps(
    {
        "model": "fake-chat",
        "messages": [{"role": "user", "content": "x" * 2000}],
        "stream": True,
        "fake": {"ttft_ms": 300, "interval_ms": 20, "chunks": 100},
    }
)
HEADERS = {"content-type": "application/json"}
Worker = Callable[[float], Awaitable[int]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.transport_ab", description=__doc__)
    parser.add_argument("--streams", default="20,200")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--client", choices=["httpx", "aiohttp"], help=argparse.SUPPRESS)
    parser.add_argument("--url", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    levels = [int(n) for n in args.streams.split(",")]

    if args.client:  # child process: one measurement
        print(uvloop.run(_measure(args.client, args.url, levels[0], args.seconds)), flush=True)
        return 0

    port = _free_port()
    fake = subprocess.Popen(
        [sys.executable, "-m", "fakeprovider", "--host=127.0.0.1", f"--port={port}", "--workers=2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(port)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        print(f"load average at start: {', '.join(f'{x:.2f}' for x in os.getloadavg())}")
        for level in levels:
            for _ in range(args.rounds):
                for client in ("httpx", "aiohttp"):
                    command = [sys.executable, "-m", "bench.transport_ab", f"--client={client}"]
                    command += [f"--url={url}", f"--streams={level}", f"--seconds={args.seconds}"]
                    subprocess.run(command, check=True)
    finally:
        fake.terminate()
        fake.wait(timeout=10)
    return 0


async def _measure(client: str, url: str, streams: int, seconds: float) -> str:
    reads = 0
    if client == "httpx":
        import httpx  # noqa: PLC0415

        limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
        async with httpx.AsyncClient(limits=limits, timeout=60) as http:

            async def worker(deadline: float) -> int:
                count = 0
                while time.monotonic() < deadline:
                    async with http.stream("POST", url, content=BODY, headers=HEADERS) as response:
                        async for _ in response.aiter_bytes():
                            count += 1
                return count

            reads, cpu, elapsed = await _run(worker, streams, seconds)
    else:
        import aiohttp  # noqa: PLC0415

        connector = aiohttp.TCPConnector(limit=2000, keepalive_timeout=30)
        async with aiohttp.ClientSession(connector=connector) as session:

            async def worker(deadline: float) -> int:
                count = 0
                while time.monotonic() < deadline:
                    async with session.post(url, data=BODY, headers=HEADERS) as response:
                        async for _ in response.content.iter_any():
                            count += 1
                return count

            reads, cpu, elapsed = await _run(worker, streams, seconds)
    return (
        f"{client:8} streams={streams:<4} reads/s={reads / elapsed:7.0f}  "
        f"cpu={cpu / elapsed * 100:4.0f}%  {cpu / max(reads, 1) * 1e6:5.0f} us/read"
    )


async def _run(worker: Worker, streams: int, seconds: float) -> tuple[int, float, float]:
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.monotonic()
    counts = await asyncio.gather(*(worker(start + seconds) for _ in range(streams)))
    elapsed = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return sum(counts), cpu, elapsed


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"fake provider did not start on port {port}")


if __name__ == "__main__":
    sys.exit(main())
