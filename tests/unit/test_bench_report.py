"""The report generator's arithmetic, on synthetic run files."""

import json
from pathlib import Path

import pytest
from bench import report


def test_quantile_interpolates_inside_buckets_like_prometheus() -> None:
    # 100 observations: 50 in (0, 1], 40 in (1, 2], 10 in (2, 4].
    buckets = [(1.0, 50.0), (2.0, 90.0), (4.0, 100.0), (float("inf"), 100.0)]
    assert report._quantile(0.5, buckets) == pytest.approx(1.0)
    assert report._quantile(0.75, buckets) == pytest.approx(1.625)
    assert report._quantile(0.99, buckets) == pytest.approx(3.8)
    assert report._quantile(0.5, [(1.0, 0.0), (float("inf"), 0.0)]) is None


def test_quantile_in_the_overflow_bucket_returns_the_last_bound() -> None:
    assert report._quantile(0.99, [(1.0, 10.0), (float("inf"), 100.0)]) == 1.0


METRICS = """\
# HELP switchboard_gateway_overhead_seconds x
switchboard_gateway_overhead_seconds_bucket{{le="0.001",phase="pre_upstream"}} {a}
switchboard_gateway_overhead_seconds_bucket{{le="0.002",phase="pre_upstream"}} {b}
switchboard_gateway_overhead_seconds_bucket{{le="+Inf",phase="pre_upstream"}} {b}
switchboard_gateway_overhead_seconds_count{{phase="pre_upstream"}} {b}
"""


def _write_run(root: Path, name: str, target: str, ttft_p99: float) -> dict[str, object]:
    path = root / "runs" / name
    path.mkdir(parents=True)
    summary = {
        "metrics": {
            "ttft_ms": {"values": {"med": 300.0, "p(95)": 310.0, "p(99)": ttft_p99, "max": 400.0}},
            "stream_ms": {"values": {"med": 2300.0, "p(99)": 2400.0}},
            "completed": {"values": {"count": 600}},
            "failed": {"values": {"passes": 1, "fails": 599}},
        }
    }
    (path / "k6-summary.json").write_text(json.dumps(summary))
    (path / "gateway-metrics-measure-start.txt").write_text(METRICS.format(a=10, b=10))
    (path / "gateway-metrics-end.txt").write_text(METRICS.format(a=60, b=110))
    stats = [
        {
            "t": 15.0,
            "Name": "switchboard-bench-gateway-1",
            "CPUPerc": "40.0%",
            "MemUsage": "100MiB / 15GiB",
        },
        {
            "t": 16.0,
            "Name": "switchboard-bench-gateway-1",
            "CPUPerc": "60.0%",
            "MemUsage": "120MiB / 15GiB",
        },
        {"t": 16.5, "Name": "switchboard-bench-gateway-1", "CPUPerc": "--", "MemUsage": "0B / 0B"},
        {
            "t": 99.0,
            "Name": "switchboard-bench-gateway-1",
            "CPUPerc": "99.0%",
            "MemUsage": "999MiB / 15GiB",
        },
    ]
    (path / "docker-stats.jsonl").write_text("\n".join(json.dumps(s) for s in stats))
    return {
        "name": name,
        "target": target,
        "vus": 200,
        "rep": 1,
        "loadavg_before": [1.0, 1.0, 1.0],
        "loadavg_after": [2.0, 2.0, 2.0],
        "measure_started_at": 10.0,
        "measure_ended_at": 70.0,
    }


def test_render_computes_added_latency_overhead_and_resources(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "direct-200-r1", "direct", 320.0),
        _write_run(tmp_path, "gateway-200-r1", "gateway", 333.0),
    ]
    manifest = {
        "command": "uv run python -m bench.run --levels 200",
        "started_at": "2026-11-27T00:00:00+00:00",
        "parameters": {
            "levels": [200],
            "reps": 1,
            "warmup_s": 30,
            "duration_s": 60,
            "prompt_tokens": 500,
        },
        "environment": {
            "host": {
                "cpu_model": "CPU",
                "logical_cpus": 8,
                "memory_gib": 16,
                "kernel": "6.6",
                "wsl": True,
                "loadavg": [1, 1, 1],
            },
            "docker": {
                "server_version": "29",
                "operating_system": "Docker Desktop",
                "other_running_containers": ["x"],
            },
            "git": {"sha": "abcdef1234567890", "dirty": False},
            "cpusets": {"gateway": "0,1"},
            "fake_profile": "# comment\nttft_ms: 300\nchunks: 100\n",
        },
        "runs": runs,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    text = report.render(tmp_path)

    # p99 TTFT: gateway 333 - direct 320 = +13 ms; within the 25 ms gate.
    assert "| 200 | +0 ms | +0 ms | +13 ms |" in text
    assert "This run measured +13 ms, within the target." in text
    # 100 observations in the window: 50 in (0, 1ms], 50 in (1, 2ms] -> p50 = 1.00 ms.
    assert "| 200 | 1.00 ms | 1.98 ms |" in text
    # CPU from the two in-window numeric samples only (40%, 60%); mem peak 120 MiB.
    assert "50% / 60% | 120 MiB" in text
    assert "1/600" in text
    assert "ttft_ms: 300, chunks: 100" in text
