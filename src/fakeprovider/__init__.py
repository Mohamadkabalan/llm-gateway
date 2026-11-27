"""A configurable fake LLM provider for tests, benchmarks and chaos runs (PLAN.md §15.2).

It speaks the OpenAI Chat Completions and Embeddings APIs. Timing (time to first
token, chunk interval, chunk count) and faults (HTTP errors, mid-stream errors,
dropped connections, hangs) come from a startup profile, per-request ``x-fake-*``
headers, a ``fake`` object in the request body, or the runtime control API.
"""
