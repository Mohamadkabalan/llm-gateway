"""Shared test constants and helpers (importable, unlike conftest)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL_CONFIG = """\
version: 1
providers:
  fake:
    type: openai
    base_url: http://fake-provider:8000/v1
models:
  fake/fake-chat:
    upstream_model: fake-chat
aliases:
  fake:
    targets: [{ model: fake/fake-chat }]
"""
