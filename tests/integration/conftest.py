import pytest


@pytest.fixture(autouse=True)
def _isolate_env() -> None:
    """Integration tests run against a real stack and read its settings from the
    environment, so they opt out of the root conftest's environment isolation."""
