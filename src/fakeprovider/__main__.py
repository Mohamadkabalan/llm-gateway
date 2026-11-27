"""``fakeprovider [--port 8000] [--workers 2] [--profile config/fake/bench.yaml]``"""

import argparse
import os
import sys

import uvicorn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fakeprovider", description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 (runs in a container)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", help="YAML file with default behavior")
    parser.add_argument("--seed", type=int, help="seed for fault-injection randomness")
    args = parser.parse_args(argv)

    # Worker processes build the app from the environment (see app_from_env).
    os.environ["FAKE_WORKERS"] = str(args.workers)
    if args.profile:
        os.environ["FAKE_PROFILE"] = args.profile
    if args.seed is not None:
        os.environ["FAKE_SEED"] = str(args.seed)
    uvicorn.run(
        "fakeprovider.app:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        loop="uvloop",
        http="httptools",
        log_level="warning",
        access_log=False,
        server_header=False,
        # Longer than the gateway pool's keepalive_expiry (30s), so the client side
        # always retires idle connections first and never reuses one we just closed.
        timeout_keep_alive=75,
        backlog=4096,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
