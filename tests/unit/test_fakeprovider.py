"""The fake provider's behavior controls and control API (its streaming is covered by the
component tests, which drive it through the gateway)."""

import base64
import struct
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import ValidationError

from fakeprovider.app import FakeProvider
from fakeprovider.behavior import Behavior, Fault, header_overrides, load_profile
from fakeprovider.openai_dialect import embedding


def test_header_overrides_map_to_fields() -> None:
    headers = [(b"x-fake-ttft-ms", b"300"), (b"x-fake-fault", b"hang"), (b"accept", b"*/*")]
    assert header_overrides(headers) == {"ttft_ms": "300", "fault": "hang"}
    behavior = Behavior().merged(header_overrides(headers))
    assert behavior.ttft_ms == 300
    assert behavior.fault is Fault.HANG


def test_unknown_or_invalid_overrides_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Behavior().merged({"ttft": 1})
    with pytest.raises(ValidationError):
        Behavior().merged({"fault_rate": 2})


def test_profile_file(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("ttft_ms: 300\ninterval_ms: 20\nchunks: 100\n")
    assert load_profile(path) == Behavior(ttft_ms=300, interval_ms=20, chunks=100)
    assert load_profile(None) == Behavior()


def test_embeddings_are_deterministic_unit_vectors() -> None:
    a, b = embedding("hello", 64), embedding("hello", 64)
    assert a == b
    assert embedding("other", 64) != a
    assert abs(sum(x * x for x in a) - 1) < 1e-9


@pytest.fixture
async def fake_client() -> AsyncIterator[httpx.AsyncClient]:
    app = FakeProvider(seed=1)
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://fake") as client:
            yield client


async def test_embeddings_endpoint_float_and_base64(fake_client: httpx.AsyncClient) -> None:
    floats = await fake_client.post(
        "/v1/embeddings", json={"model": "e", "input": ["a", "b"], "dimensions": 8}
    )
    assert floats.status_code == 200
    data = floats.json()["data"]
    assert len(data) == 2
    assert len(data[0]["embedding"]) == 8

    encoded = await fake_client.post(
        "/v1/embeddings",
        json={"model": "e", "input": "a", "dimensions": 8, "encoding_format": "base64"},
    )
    raw = base64.b64decode(encoded.json()["data"][0]["embedding"])
    decoded = struct.unpack("<8f", raw)
    assert decoded == pytest.approx(data[0]["embedding"], rel=1e-6)


async def test_control_api_sets_runtime_behavior_and_reports_stats(
    fake_client: httpx.AsyncClient,
) -> None:
    response = await fake_client.put("/_control/behavior", json={"fault": "status", "status": 529})
    assert response.json()["status"] == 529
    failed = await fake_client.post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    assert failed.status_code == 529

    await fake_client.delete("/_control/behavior")
    ok = await fake_client.post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    assert ok.status_code == 200
    assert ok.json()["choices"][0]["message"]["content"]

    stats = (await fake_client.get("/_control/stats")).json()
    assert stats["requests"]["/v1/chat/completions"] == 2
    assert stats["faults"] == {"status": 1}
    assert stats["recent"][-1]["body"]["model"] == "m"

    await fake_client.post("/_control/reset")
    assert (await fake_client.get("/_control/stats")).json()["requests"] == {}


async def test_invalid_runtime_behavior_is_400(fake_client: httpx.AsyncClient) -> None:
    response = await fake_client.put("/_control/behavior", json={"chunks": -1})
    assert response.status_code == 400


async def test_control_api_refuses_with_several_workers() -> None:
    app = FakeProvider(workers=2)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://fake") as client:
        response = await client.get("/_control/stats")
    assert response.status_code == 409


async def test_unknown_path_is_404(fake_client: httpx.AsyncClient) -> None:
    assert (await fake_client.get("/nope")).status_code == 404
