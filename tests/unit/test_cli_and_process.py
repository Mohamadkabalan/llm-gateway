"""CLI exit codes, and a real ``switchboard serve`` process shutting down on SIGTERM."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from switchboard.cli import EXIT_INVALID_CONFIG, EXIT_INVALID_SETTINGS, EXIT_OK, main
from tests.support import MINIMAL_CONFIG, REPO_ROOT


def test_validate_config_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "routing.yaml"
    path.write_text(MINIMAL_CONFIG)
    assert main(["validate-config", str(path)]) == EXIT_OK
    assert capsys.readouterr().out.startswith(f"OK {path} (version ")


def test_validate_config_reports_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "routing.yaml"
    path.write_text(MINIMAL_CONFIG.replace("targets:", "target:"))
    assert main(["validate-config", str(path)]) == EXIT_INVALID_CONFIG
    err = capsys.readouterr().err
    assert "unknown field 'target' (did you mean 'targets'?)" in err
    assert "required field is missing" in err


def test_invalid_settings_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SWITCHBOARD_PORT", "99999")
    assert main(["validate-config"]) == EXIT_INVALID_SETTINGS
    assert "SWITCHBOARD_PORT" in capsys.readouterr().err


def test_serve_with_invalid_config_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "routing.yaml"
    bad.write_text(MINIMAL_CONFIG + "bogus: true\n")
    proc = subprocess.run(
        [sys.executable, "-m", "switchboard", "serve"],
        env={**os.environ, "SWITCHBOARD_ROUTING_CONFIG": str(bad), "SWITCHBOARD_METRICS_PORT": "0"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == EXIT_INVALID_CONFIG
    assert "line 12, bogus: unknown field 'bogus'" in proc.stderr
    event = json.loads(proc.stdout.splitlines()[-1])
    assert event["event"] == "config.invalid"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def test_sigterm_shuts_down_gracefully(tmp_path: Path) -> None:
    config = tmp_path / "routing.yaml"
    config.write_text(MINIMAL_CONFIG)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "switchboard", "serve"],
        env={
            **os.environ,
            "SWITCHBOARD_HOST": "127.0.0.1",
            "SWITCHBOARD_PORT": str(port),
            "SWITCHBOARD_METRICS_PORT": str(_free_port()),
            "SWITCHBOARD_ROUTING_CONFIG": str(config),
        },
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/ready", timeout=0.5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "gateway never became ready"
            time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    # uvicorn re-raises the signal after a clean shutdown, so the process ends "by SIGTERM".
    assert proc.returncode in (0, -signal.SIGTERM)
    events = [json.loads(line)["event"] for line in output.splitlines() if line.startswith("{")]
    assert events.index("gateway.started") < events.index("gateway.stopping")
    assert events.index("gateway.stopping") < events.index("gateway.stopped")
