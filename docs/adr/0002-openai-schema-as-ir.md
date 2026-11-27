# 0002: The OpenAI Chat Completions schema is the internal format

- **Status:** accepted
- **Date:** 2026-11-27

## Context

Clients speak OpenAI Chat Completions; that is the product. Upstreams speak OpenAI
(and OpenAI-compatible servers such as vLLM and Ollama's `/v1`), the Anthropic Messages
API, and Ollama's native API. A gateway can either invent a neutral internal format and
translate on both sides, or adopt the ingress format and translate only towards
providers that differ.

## Decision

- Requests stay in the OpenAI shape inside the gateway (`schemas.chat.ChatRequest`).
  Only the fields the gateway reads are declared; unknown fields are kept
  (`extra="allow"`) and passed through to OpenAI-compatible upstreams, so a new OpenAI
  parameter works the day it ships.
- Responses are normalized into small stream events (`schemas.events`: text, refusal,
  reasoning, tool-call start, tool-call argument fragment, finish, usage), and one
  encoder turns them back into OpenAI chunks or a whole `chat.completion`.
- Usage is normalized into non-overlapping buckets (uncached input, cache reads, cache
  writes, output, with reasoning as part of output), because OpenAI counts cached
  tokens inside `prompt_tokens` and Anthropic does not.

## Consequences

- The OpenAI adapter is near pass-through, which keeps the common path cheap and
  low-risk. Only Anthropic and Ollama translations need deep tests.
- The event model carries one choice and no logprobs, so `n > 1` and `logprobs` are
  refused with a clear 400 instead of being silently wrong. Supporting them later means
  widening the event model, not redesigning it.
- Response ids are the gateway's own (`chatcmpl-<uuid7>`), so cached replays and
  responses from different providers look alike to clients.
- Revisit if OpenAI's Responses API becomes the dominant client format; it would be a
  second ingress, still translated onto the same events.
