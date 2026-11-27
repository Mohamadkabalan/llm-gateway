"""Run the gateway-overhead benchmark matrix (PLAN.md §16.1).

    uv run python -m bench.run --levels 50,200,500 --reps 3 --warmup 30 --duration 180

For every concurrency level and repetition it runs the same k6 script twice, once
straight at the fake provider (the baseline) and once through the gateway,
alternating the order so slow drift on the machine affects both equally. Each run
directory gets the raw k6 summary, ``docker stats`` samples and gateway metrics
snapshots; ``manifest.json`` records the environment and every command. The report
(``bench/report.py``) is generated from those files only, so no number is typed by
hand.
"""

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bench import report

REPO = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "-f", str(REPO / "deploy/compose.bench.yaml")]
K6_IMAGE = "switchboard/k6:2.0.0-sse0.2.0"
GATEWAY_IMAGE = "switchboard/gateway:dev"
# Matches SWITCHBOARD_BENCH_KEY in deploy/compose.bench.yaml; local only, never leaves the box.
BENCH_KEY = "sb_live_bench0000000000000000000000000000000000000"
TARGETS = {
    "direct": {"TARGET_URL": "http://fake-provider:8000/v1/chat/completions", "MODEL": "fake-chat"},
    "gateway": {
        "TARGET_URL": "http://gateway:8080/v1/chat/completions",
        "MODEL": "fake",
        "API_KEY": BENCH_KEY,
    },
}
_METRICS_SCRIPT = (
    "import sys, urllib.request; "
    "sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:9100/metrics').read().decode())"
)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "started_at": _now(),
        "command": " ".join(["uv run python -m bench.run", *(argv or sys.argv[1:])]),
        "parameters": {
            "levels": args.levels,
            "reps": args.reps,
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "prompt_tokens": args.prompt_tokens,
            "targets": args.targets,
        },
        "environment": capture_environment(),
        "runs": [],
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"Results: {run_dir.relative_to(REPO)}")

    _sh([*COMPOSE, "build", "gateway", "k6"], quiet=True)
    _sh([*COMPOSE, "up", "-d", "--wait", "gateway", "fake-provider"])
    try:
        for level in args.levels:
            for rep in range(1, args.reps + 1):
                order = args.targets if rep % 2 else list(reversed(args.targets))
                for target in order:
                    entry = run_one(run_dir, target, level, rep, args)
                    manifest["runs"].append(entry)
                    _write_json(run_dir / "manifest.json", manifest)
                    time.sleep(args.cooldown)
    finally:
        manifest["finished_at"] = _now()
        _write_json(run_dir / "manifest.json", manifest)
        if not args.keep_up:
            _sh([*COMPOSE, "down"], quiet=True)

    markdown = report.render(run_dir)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def run_one(
    run_dir: Path, target: str, level: int, rep: int, args: argparse.Namespace
) -> dict[str, Any]:
    name = f"{target}-{level}-r{rep}"
    out = run_dir / "runs" / name
    out.mkdir(parents=True)
    print(f"[{_now()}] {name}: {level} streams, {args.warmup}s warm-up + {args.duration}s measured")

    env = {
        **TARGETS[target],
        "VUS": str(level),
        "WARMUP_S": str(args.warmup),
        "DURATION_S": str(args.duration),
        "PROMPT_TOKENS": str(args.prompt_tokens),
        "SUMMARY_PATH": "/results/k6-summary.json",
    }
    command = [*COMPOSE, "run", "--rm", "-T", "--name", f"switchboard-bench-k6-{name}"]
    command += ["--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{out}:/results"]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += ["k6", "run", "--quiet", "--no-color", "/scripts/chat_stream.js"]

    entry: dict[str, Any] = {
        "name": name,
        "target": target,
        "vus": level,
        "rep": rep,
        "command": _redact(" ".join(command)),
        "loadavg_before": os.getloadavg(),
    }
    _scrape_metrics(out / "gateway-metrics-start.txt")
    with _docker_stats(out / "docker-stats.jsonl"):
        started = time.time()
        k6 = subprocess.Popen(command, stdout=(out / "k6.log").open("w"), stderr=subprocess.STDOUT)
        # Snapshot gateway metrics when the warm-up ends (k6's own clock starts about a
        # second after the container does, so this window is approximate).
        time.sleep(args.warmup + 1)
        entry["measure_started_at"] = time.time()
        _scrape_metrics(out / "gateway-metrics-measure-start.txt")
        entry["k6_exit_code"] = k6.wait()
        entry["measure_ended_at"] = time.time()
    _scrape_metrics(out / "gateway-metrics-end.txt")
    entry["wall_s"] = round(time.time() - started, 1)
    entry["loadavg_after"] = os.getloadavg()
    if entry["k6_exit_code"] != 0:
        print(f"  k6 exited with {entry['k6_exit_code']}; see {out / 'k6.log'}")
    return entry


def capture_environment() -> dict[str, Any]:
    cpuinfo = (
        Path("/proc/cpuinfo").read_text(encoding="utf-8") if Path("/proc/cpuinfo").exists() else ""
    )
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    meminfo = (
        Path("/proc/meminfo").read_text(encoding="utf-8") if Path("/proc/meminfo").exists() else ""
    )
    mem_kb = next(
        (int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemTotal")), None
    )
    kernel_version = (
        Path("/proc/version").read_text(encoding="utf-8") if Path("/proc/version").exists() else ""
    )
    docker_info = json.loads(_out(["docker", "info", "--format", "{{json .}}"]) or "{}")
    others = [
        name
        for name in _out(["docker", "ps", "--format", "{{.Names}}"]).split()
        if not name.startswith("switchboard-bench")
    ]
    bench_files = ["config/routing.bench.yaml", "config/fake/bench.yaml", "bench/k6/chat_stream.js"]
    return {
        "host": {
            "cpu_model": model,
            "logical_cpus": os.cpu_count(),
            "memory_gib": round(mem_kb / 1024 / 1024, 1) if mem_kb else None,
            "kernel": platform.release(),
            "wsl": "microsoft" in kernel_version.lower(),
            "python": platform.python_version(),
            "loadavg": os.getloadavg(),
            "cpu_topology": _out(["lscpu", "-e=CPU,CORE,SOCKET"]),
        },
        "docker": {
            "server_version": docker_info.get("ServerVersion"),
            "operating_system": docker_info.get("OperatingSystem"),
            "ncpu": docker_info.get("NCPU"),
            "mem_total_gib": round(docker_info.get("MemTotal", 0) / 1024**3, 1),
            "other_running_containers": others,
        },
        "git": {
            "sha": _out(["git", "rev-parse", "HEAD"]).strip(),
            "dirty": bool(_out(["git", "status", "--porcelain"]).strip()),
        },
        "images": {
            "gateway": _out(
                ["docker", "image", "inspect", GATEWAY_IMAGE, "--format", "{{.Id}}"]
            ).strip(),
            "k6": _out(["docker", "run", "--rm", K6_IMAGE, "version"]).strip(),
        },
        "files_sha256": {
            path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in bench_files
        },
        "fake_profile": (REPO / "config/fake/bench.yaml").read_text(encoding="utf-8"),
        "cpusets": {
            "gateway": os.environ.get("SB_BENCH_CPUS_GATEWAY", "0,1"),
            "fake-provider": os.environ.get("SB_BENCH_CPUS_FAKE", "2,3"),
            "k6": os.environ.get("SB_BENCH_CPUS_K6", "4-7"),
            "fake_workers": os.environ.get("SB_BENCH_FAKE_WORKERS", "2"),
        },
    }


@contextlib.contextmanager
def _docker_stats(path: Path) -> Iterator[None]:
    """Sample ``docker stats`` (about once a second) for the bench containers."""
    proc = subprocess.Popen(
        ["docker", "stats", "--format", "{{json .}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    stop = threading.Event()

    def pump() -> None:
        assert proc.stdout is not None
        with path.open("w", encoding="utf-8") as sink:
            for raw in proc.stdout:
                if stop.is_set():
                    break
                line = _ANSI.sub("", raw).strip()
                if "switchboard-bench" not in line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    sample = json.loads(line)
                    sink.write(json.dumps({"t": time.time(), **sample}) + "\n")

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        proc.terminate()
        proc.wait(timeout=10)
        thread.join(timeout=10)


def _scrape_metrics(path: Path) -> None:
    text = _out([*COMPOSE, "exec", "-T", "gateway", "python", "-c", _METRICS_SCRIPT])
    path.write_text(text, encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--levels",
        type=_int_list,
        default=[50, 200, 500],
        help="concurrent streams, e.g. 50,200,500",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30, help="seconds, not recorded")
    parser.add_argument("--duration", type=int, default=180, help="measured seconds")
    parser.add_argument("--prompt-tokens", type=int, default=500)
    parser.add_argument("--targets", type=lambda s: s.split(","), default=["direct", "gateway"])
    parser.add_argument("--cooldown", type=float, default=10, help="seconds between runs")
    parser.add_argument(
        "--scratch", action="store_true", help="write to bench/results/scratch-* (git-ignored)"
    )
    parser.add_argument("--keep-up", action="store_true", help="leave the bench stack running")
    args = parser.parse_args(argv)
    unknown = set(args.targets) - set(TARGETS)
    if unknown:
        parser.error(f"unknown targets: {', '.join(sorted(unknown))}")
    return args


def _run_dir(args: argparse.Namespace) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = _out(["git", "rev-parse", "--short=7", "HEAD"]).strip() or "nogit"
    prefix = "scratch-" if args.scratch else ""
    return REPO / "bench" / "results" / f"{prefix}{stamp}-{sha}"


def _int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _redact(command: str) -> str:
    return command.replace(BENCH_KEY, "<bench-key>")


def _sh(command: list[str], *, quiet: bool = False) -> None:
    subprocess.run(command, check=True, cwd=REPO, stdout=subprocess.DEVNULL if quiet else None)


def _out(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, cwd=REPO, capture_output=True, text=True, timeout=120
        ).stdout
    except OSError, subprocess.TimeoutExpired:
        return ""


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
