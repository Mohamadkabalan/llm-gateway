"""Turn a bench run directory into Markdown tables.

    uv run python -m bench.report bench/results/<run-id>

Every number comes from files in the run directory: k6 summaries, gateway metrics
snapshots and ``docker stats`` samples. Nothing is typed in by hand.

* **Added latency** is the gateway's percentile minus the direct baseline's
  percentile at the same concurrency (medians over repetitions), as PLAN.md §16.1
  specifies. A difference of percentiles is not the percentile of per-request
  differences; it is what a client sees, which is the point.
* **Internal overhead** comes from the gateway's own histograms over the measured
  window. Percentiles are interpolated inside buckets the way Prometheus's
  ``histogram_quantile`` does, so they are approximate.
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$"
)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_UNITS = {
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
}
MISSING = "n/a"
GATE_LEVEL, GATE_TARGET_MS = 200, 25.0


def render(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    runs = [_load_run(run_dir / "runs" / entry["name"], entry) for entry in manifest["runs"]]
    levels = sorted({run["vus"] for run in runs})
    lines = _header(run_dir, manifest)
    lines += _added_latency(runs, levels)
    lines += _overhead(runs, levels)
    lines += _resources(runs, levels)
    lines += _every_run(runs)
    return "\n".join(lines) + "\n"


# --- loading -------------------------------------------------------------------------


def _load_run(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    run = dict(entry)
    summary_path = path / "k6-summary.json"
    metrics = (
        json.loads(summary_path.read_text(encoding="utf-8"))["metrics"]
        if summary_path.exists()
        else {}
    )
    run["ttft"] = _values(metrics, "ttft_ms")
    run["stream"] = _values(metrics, "stream_ms")
    run["completed"] = _values(metrics, "completed").get("count", 0)
    failed = _values(metrics, "failed")
    run["failed"] = failed.get("passes", 0)  # a Rate's "passes" are the true samples
    run["attempted"] = failed.get("passes", 0) + failed.get("fails", 0)
    run["overhead"] = _histogram_deltas(path)
    run["stats"] = _docker_stats(
        path / "docker-stats.jsonl", entry.get("measure_started_at"), entry.get("measure_ended_at")
    )
    return run


def _values(metrics: dict[str, Any], name: str) -> dict[str, float]:
    metric = metrics.get(name) or {}
    return dict(metric.get("values", metric))


def _histogram_deltas(path: Path) -> dict[str, dict[str, float | None]]:
    start, end = path / "gateway-metrics-measure-start.txt", path / "gateway-metrics-end.txt"
    if not (start.exists() and end.exists()):
        return {}
    before = _buckets(start.read_text(encoding="utf-8"))
    after = _buckets(end.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float | None]] = {}
    for key, buckets in after.items():
        delta = [
            (le, count - before.get(key, {}).get(le, 0.0)) for le, count in sorted(buckets.items())
        ]
        total = delta[-1][1] if delta else 0.0
        result[key] = {
            "count": total,
            **{f"p{int(q * 100)}": _quantile(q, delta) for q in (0.5, 0.95, 0.99)},
        }
    return result


def _buckets(text: str) -> dict[str, dict[float, float]]:
    """``{"gateway_overhead:pre_upstream": {le: cumulative_count}}`` plus TTFT."""
    out: dict[str, dict[float, float]] = defaultdict(dict)
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if not match or not match["name"].endswith("_bucket"):
            continue
        labels = dict(_LABEL.findall(match["labels"] or ""))
        le = float("inf") if labels.get("le") == "+Inf" else float(labels.get("le", "nan"))
        if match["name"] == "switchboard_gateway_overhead_seconds_bucket":
            key = f"overhead:{labels.get('phase')}"
        elif match["name"] == "switchboard_ttft_seconds_bucket":
            key = "ttft"
        else:
            continue
        out[key][le] = out[key].get(le, 0.0) + float(match["value"])
    return out


def _quantile(q: float, buckets: list[tuple[float, float]]) -> float | None:
    """Prometheus-style ``histogram_quantile`` over cumulative (le, count) pairs."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    rank = q * buckets[-1][1]
    lower_bound, lower_count = 0.0, 0.0
    for upper_bound, count in buckets:
        if count >= rank:
            if upper_bound == float("inf"):
                return lower_bound
            width = count - lower_count
            fraction = (rank - lower_count) / width if width > 0 else 1.0
            return lower_bound + (upper_bound - lower_bound) * fraction
        lower_bound, lower_count = upper_bound, count
    return lower_bound


def _docker_stats(
    path: Path, start: float | None, end: float | None
) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    cpu: dict[str, list[float]] = defaultdict(list)
    mem: dict[str, list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        if start is not None and end is not None and not start <= sample["t"] <= end:
            continue
        service = _service(sample.get("Name", ""))
        if service is None:
            continue
        try:  # a container that is starting or stopping reports "--"
            cpu_percent = float(sample["CPUPerc"].rstrip("%"))
        except ValueError:
            continue
        cpu[service].append(cpu_percent)
        mem[service].append(_bytes(sample["MemUsage"].split("/")[0]))
    return {
        service: {
            "cpu_mean": statistics.fmean(values),
            "cpu_max": max(values),
            "mem_max_mib": max(mem[service]) / 1024**2,
            "samples": len(values),
        }
        for service, values in cpu.items()
        if values
    }


def _service(container: str) -> str | None:
    for service in ("gateway", "fake-provider", "k6"):
        if container.startswith(f"switchboard-bench-{service}"):
            return service
    return None


def _bytes(text: str) -> float:
    match = re.match(r"\s*([\d.]+)\s*([a-zA-Z]+)", text)
    if not match:
        return 0.0
    return float(match[1]) * _UNITS.get(match[2].lower(), 1)


# --- tables --------------------------------------------------------------------------


def _header(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    env, params = manifest["environment"], manifest["parameters"]
    host, docker, git = env["host"], env["docker"], env["git"]
    dirty = " (uncommitted changes)" if git["dirty"] else ""
    others = docker["other_running_containers"]
    return [
        f"# Gateway overhead: {run_dir.name}",
        "",
        f"- **Command:** `{manifest['command']}`",
        f"- **Code:** git `{git['sha'][:12]}`{dirty}; started {manifest['started_at']}",
        f"- **Machine:** {host['cpu_model']}, {host['logical_cpus']} logical CPUs, "
        f"{host['memory_gib']} GiB; kernel {host['kernel']}{' (WSL2)' if host['wsl'] else ''}; "
        f"Docker {docker['server_version']} ({docker['operating_system']})",
        f"- **Load average at start:** {', '.join(f'{x:.2f}' for x in host['loadavg'])}; "
        f"{len(others)} unrelated containers running",
        f"- **CPU pinning:** {', '.join(f'{k} {v}' for k, v in env['cpusets'].items())}",
        f"- **Upstream (fake provider profile):** `{_profile(env['fake_profile'])}`",
        f"- **Load:** closed model, {params['warmup_s']} s warm-up (discarded) + "
        f"{params['duration_s']} s measured, {params['reps']} repetition(s) per level, "
        f"prompt ≈ {params['prompt_tokens']} tokens, unique per request",
        "",
    ]


def _profile(text: str) -> str:
    settings = [line for line in text.splitlines() if line and not line.startswith("#")]
    return ", ".join(settings)


def _table(*columns: str) -> list[str]:
    """A Markdown table header; columns starting with '<' are left-aligned."""
    names = [column.lstrip("<") for column in columns]
    rule = ["---" if column.startswith("<") else "---:" for column in columns]
    return ["| " + " | ".join(names) + " |", "|" + "|".join(rule) + "|"]


def _median(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _by(runs: list[dict[str, Any]], target: str, level: int) -> list[dict[str, Any]]:
    return [run for run in runs if run["target"] == target and run["vus"] == level]


def _ms(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return MISSING
    return f"{value:+.0f} ms" if signed else f"{value:.0f} ms"


def _added_latency(runs: list[dict[str, Any]], levels: list[int]) -> list[str]:
    lines = [
        "## Added latency (gateway minus direct, client side)",
        "",
        "Median over repetitions of each percentile, gateway minus direct. k6 measures whole "
        "milliseconds.",
        "",
        *_table(
            "Streams",
            "TTFT p50",
            "TTFT p95",
            "TTFT p99",
            "Stream p50",
            "Stream p99",
            "Gateway failed",
            "Direct failed",
        ),
    ]
    gate: float | None = None
    for level in levels:
        gateway, direct = _by(runs, "gateway", level), _by(runs, "direct", level)
        cells = []
        for metric, stat in (
            ("ttft", "med"),
            ("ttft", "p(95)"),
            ("ttft", "p(99)"),
            ("stream", "med"),
            ("stream", "p(99)"),
        ):
            g = _median(run[metric].get(stat) for run in gateway)
            d = _median(run[metric].get(stat) for run in direct)
            added = g - d if g is not None and d is not None else None
            cells.append(_ms(added, signed=True))
            if level == GATE_LEVEL and metric == "ttft" and stat == "p(99)":
                gate = added
        cells += [_failures(gateway), _failures(direct)]
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    if gate is not None:
        verdict = "within" if gate <= GATE_TARGET_MS else "over"
        lines += [
            "",
            f"**Decision gate (ADR-0001):** the target is p99 added TTFT ≤ "
            f"{GATE_TARGET_MS:.0f} ms at {GATE_LEVEL} streams. This run measured "
            f"{gate:+.0f} ms, {verdict} the target.",
        ]
    return [*lines, ""]


def _failures(runs: list[dict[str, Any]]) -> str:
    attempted = sum(run["attempted"] for run in runs)
    failed = sum(run["failed"] for run in runs)
    return f"{failed}/{attempted}" if attempted else MISSING


def _overhead(runs: list[dict[str, Any]], levels: list[int]) -> list[str]:
    lines = [
        "## Gateway internal overhead (server side)",
        "",
        "From the gateway's `switchboard_gateway_overhead_seconds` histogram over the measured "
        "window, bucket-interpolated. `pre_upstream`: request received → upstream request sent. "
        "`first_chunk_forward`: first content bytes read from upstream → first chunk written to "
        "the client.",
        "",
        *_table("Streams", "pre_upstream p50", "p99", "first_chunk_forward p50", "p99", "Samples"),
    ]
    for level in levels:
        gateway = _by(runs, "gateway", level)
        pre = [run["overhead"].get("overhead:pre_upstream", {}) for run in gateway]
        fwd = [run["overhead"].get("overhead:first_chunk_forward", {}) for run in gateway]
        cells = [
            _sub_ms(_median(h.get("p50") for h in pre)),
            _sub_ms(_median(h.get("p99") for h in pre)),
            _sub_ms(_median(h.get("p50") for h in fwd)),
            _sub_ms(_median(h.get("p99") for h in fwd)),
            f"{sum(h.get('count') or 0 for h in pre):.0f}",
        ]
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _sub_ms(seconds: float | None) -> str:
    return MISSING if seconds is None else f"{seconds * 1000:.2f} ms"


def _resources(runs: list[dict[str, Any]], levels: list[int]) -> list[str]:
    lines = [
        "## Throughput and resources (measured window)",
        "",
        "CPU is `docker stats` percent of one CPU (100% = one logical CPU busy); memory is "
        "the container's peak.",
        "",
        *_table(
            "Streams",
            "<Target",
            "Completed/s",
            "Gateway CPU mean / max",
            "Gateway mem max",
            "Fake CPU max",
            "k6 CPU max",
        ),
    ]
    for level in levels:
        for target in ("direct", "gateway"):
            group = _by(runs, target, level)
            if not group:
                continue
            duration = _duration(group[0])
            rate = _median(run["completed"] / duration for run in group) if duration else None
            gw = [run["stats"].get("gateway", {}) for run in group]
            cpu = mem = MISSING
            if target == "gateway" and any(gw):
                cpu_mean = _median(stats.get("cpu_mean") for stats in gw)
                cpu_max = _median(stats.get("cpu_max") for stats in gw)
                cpu = f"{_pct(cpu_mean)} / {_pct(cpu_max)}"
                mem_max = _median(stats.get("mem_max_mib") for stats in gw)
                mem = MISSING if mem_max is None else f"{mem_max:.0f} MiB"
            fake = _median(run["stats"].get("fake-provider", {}).get("cpu_max") for run in group)
            k6 = _median(run["stats"].get("k6", {}).get("cpu_max") for run in group)
            completed = MISSING if rate is None else f"{rate:.1f}"
            lines.append(
                f"| {level} | {target} | {completed} | {cpu} | {mem} | {_pct(fake)} | {_pct(k6)} |"
            )
    return [*lines, ""]


def _duration(run: dict[str, Any]) -> float | None:
    start, end = run.get("measure_started_at"), run.get("measure_ended_at")
    return end - start if start and end else None


def _pct(value: float | None) -> str:
    return MISSING if value is None else f"{value:.0f}%"


def _every_run(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Every run",
        "",
        *_table(
            "<Run",
            "Completed",
            "Failed",
            "TTFT p50",
            "p95",
            "p99",
            "max",
            "Stream p50",
            "p99",
            "<Load avg before → after",
        ),
    ]
    for run in runs:
        t, s = run["ttft"], run["stream"]
        load = f"{run['loadavg_before'][0]:.1f} → {run['loadavg_after'][0]:.1f}"
        cells = [
            run["name"],
            f"{run['completed']:.0f}",
            f"{run['failed']:.0f}",
            *(_ms(t.get(stat)) for stat in ("med", "p(95)", "p(99)", "max")),
            *(_ms(s.get(stat)) for stat in ("med", "p(99)")),
            load,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m bench.report bench/results/<run-id>", file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    markdown = render(run_dir)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
