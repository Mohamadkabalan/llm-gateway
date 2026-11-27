import pytest

from switchboard.providers.http import proxy_for


@pytest.mark.parametrize(
    ("url", "env", "expected"),
    [
        ("https://api.openai.com/v1", {}, None),
        ("https://api.openai.com/v1", {"HTTPS_PROXY": "http://proxy:3128"}, "http://proxy:3128"),
        ("http://fake-provider:8000/v1", {"HTTPS_PROXY": "http://proxy:3128"}, None),
        ("http://fake-provider:8000/v1", {"http_proxy": "http://p:1"}, "http://p:1"),
        (
            "https://api.openai.com/v1",
            {"HTTPS_PROXY": "http://p:1", "NO_PROXY": "localhost,.openai.com"},
            None,
        ),
        (
            "https://api.anthropic.com",
            {"HTTPS_PROXY": "http://p:1", "NO_PROXY": "openai.com"},
            "http://p:1",
        ),
        ("https://api.anthropic.com", {"HTTPS_PROXY": "http://p:1", "no_proxy": "*"}, None),
    ],
)
def test_proxy_resolution(url: str, env: dict[str, str], expected: str | None) -> None:
    assert proxy_for(url, env) == expected
