"""/v1/models, /v1/embeddings, the Ollama profile and capability checks, end to end."""

import base64
import struct
from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessageParam

from tests.component.conftest import Stack, StackFactory

USER: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]
FAST = {"fake": {"ttft_ms": 0, "interval_ms": 0, "chunks": 3}}


async def test_models_lists_aliases_and_enabled_models(make_stack: StackFactory) -> None:
    providers = (
        "  gone: { type: openai, base_url: 'http://gone/v1', api_key_env: GONE_API_KEY, "
        "optional: true }"
    )
    async with make_stack(providers=providers, models="  gone/m: { upstream_model: m }") as stack:
        page = await stack.sdk().models.list()
        owners = {model.id: model.owned_by for model in page.data}
        assert owners == {
            "fake": "switchboard",
            "embed": "switchboard",
            "fake/fake-chat": "fake",
            "fake/fake-embed": "fake",
        }  # gone/m is left out: its provider has no key
        model = await stack.sdk().models.retrieve("fake/fake-chat")
        assert model.owned_by == "fake"
        async with stack.http() as http:
            missing = await http.get("/v1/models/gone/m")
            anonymous = await http.get("/v1/models", headers={"authorization": ""})
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "model_not_found"
        assert anonymous.status_code == 401


async def test_embeddings_through_the_sdk(stack: Stack) -> None:
    # The SDK asks for base64 by default and decodes it; the gateway passes it through.
    response = await stack.sdk().embeddings.create(model="embed", input=["hello", "world"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) == 256
    assert response.usage.prompt_tokens > 0
    sent = stack.fake.stats.recent[-1]
    assert sent["path"] == "/v1/embeddings"
    assert sent["body"]["model"] == "fake-embed"


async def test_embeddings_float_format_and_headers(stack: Stack) -> None:
    async with stack.http() as http:
        response = await http.post(
            "/v1/embeddings",
            json={"model": "embed", "input": "hi", "encoding_format": "base64", "dimensions": 4},
        )
    assert response.status_code == 200
    assert response.headers["x-switchboard-model"] == "fake/fake-embed"
    raw = base64.b64decode(response.json()["data"][0]["embedding"])
    assert len(struct.unpack("<4f", raw)) == 4


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"model": "fake", "input": "x"}, 400, "model_kind_mismatch"),
        ({"model": "embed", "messages": []}, 400, None),
        ({"model": "nope", "input": "x"}, 404, "model_not_found"),
    ],
)
async def test_embedding_request_errors(
    stack: Stack, body: dict[str, Any], status: int, code: str | None
) -> None:
    async with stack.http() as http:
        response = await http.post("/v1/embeddings", json=body)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


async def test_chat_model_cannot_be_used_for_embeddings_and_vice_versa(stack: Stack) -> None:
    async with stack.http() as http:
        response = await http.post(
            "/v1/chat/completions", json={"model": "embed", "messages": USER}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_kind_mismatch"


OLLAMA = "  local: { type: ollama, base_url: '{fake_url}' }"
OLLAMA_MODELS = (
    "  local/llama: { upstream_model: 'llama3.2:3b', "
    "capabilities: { tools: true, forced_tool_choice: false } }"
)


async def test_ollama_profile_rewrites_the_request(make_stack: StackFactory) -> None:
    async with make_stack(providers=OLLAMA, models=OLLAMA_MODELS) as stack:
        async with stack.http() as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "local/llama",
                    "messages": USER,
                    "stream": True,
                    "max_completion_tokens": 20,
                    "tool_choice": "none",
                    "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
                    "user": "u-1",
                    **FAST,
                },
            )
        assert response.status_code == 200
        assert '"content":"lorem "' in response.text
        assert response.headers["x-switchboard-dropped-params"] == "user"
        body = stack.fake.stats.recent[-1]["body"]
        assert body["model"] == "llama3.2:3b"
        assert body["max_tokens"] == 20
        assert "max_completion_tokens" not in body
        assert "tools" not in body  # tool_choice none, said the way Ollama supports
        assert "tool_choice" not in body
        assert "user" not in body


async def test_ollama_rejects_linked_images(make_stack: StackFactory) -> None:
    models = "  local/vision: { upstream_model: llava, capabilities: { vision: true } }"
    async with make_stack(providers=OLLAMA, models=models) as stack, stack.http() as http:
        image = {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
        response = await http.post(
            "/v1/chat/completions",
            json={"model": "local/vision", "messages": [{"role": "user", "content": [image]}]},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert stack.fake.stats.requests["/v1/chat/completions"] == 0  # never sent


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        (
            {"tool_choice": "required", "tools": [{"type": "function", "function": {"name": "f"}}]},
            "forced tool_choice",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            }
                        ],
                    }
                ]
            },
            "image input",
        ),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "x", "schema": {}},
                }
            },
            "json_schema",
        ),
        ({"max_tokens": 5000}, "more than 4096 output tokens"),
    ],
)
async def test_capability_gaps_are_400s_naming_the_gap(
    make_stack: StackFactory, extra: dict[str, Any], fragment: str
) -> None:
    models = (
        "  fake/small: { upstream_model: small, capabilities: { forced_tool_choice: false, "
        "max_output_tokens: 4096 } }"
    )
    async with make_stack(models=models) as stack, stack.http() as http:
        response = await http.post(
            "/v1/chat/completions", json={"model": "fake/small", "messages": USER, **extra}
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "model_capability_missing"
    assert fragment in error["message"]
