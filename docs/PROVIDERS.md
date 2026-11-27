# Providers

Switchboard speaks the OpenAI Chat Completions and Embeddings APIs to clients and
translates for each upstream. This page lists exactly what is translated, what is
dropped or refused, and the known gaps. Provider facts were checked against the
official docs on 2026-11-27; source links and the full fact list are in
[PLAN.md §3.8](../PLAN.md).

| Provider type | Adapter | Chat | Embeddings | Notes |
|---|---|---|---|---|
| `openai` | `providers/openai.py` | ✓ | ✓ | Near pass-through. Also any OpenAI-compatible server (vLLM, LM Studio, the fake provider). |
| `anthropic` | `providers/anthropic.py`, `anthropic_request.py` | ✓ | — | Native Messages API, API version `2026-10-21` by default. Anthropic has no embeddings API. |
| `ollama` | `providers/ollama.py` | ✓ | ✓ | Ollama's OpenAI-compatible `/v1`, with its limits applied. `base_url` is the server root. |

Every upstream call streams; non-streaming clients get an aggregated response
([ADR-0003](adr/0003-upstream-calls-always-stream.md)). Retries, fallback between
providers and circuit breakers are described in [ROUTING.md](ROUTING.md). Responses carry
`x-switchboard-provider`, `x-switchboard-model` and, when request fields were dropped,
`x-switchboard-dropped-params`.

## OpenAI

- `model` becomes the configured `upstream_model`. `stream: true` and
  `stream_options.include_usage: true` are always set upstream; the final usage chunk
  reaches the client only if it asked for one.
- `max_tokens` (deprecated by OpenAI) becomes `max_completion_tokens`. With neither,
  the model's `defaults.max_tokens` applies, if configured.
- Every other field is forwarded unchanged, including fields Switchboard does not know,
  except `provider_options`: its `openai` entry (`ollama` for Ollama) is merged into the
  body before the gateway sets `model` and streaming, and the field itself is never
  forwarded. Entries for other providers are dropped, so a request written for Anthropic
  can fall back to OpenAI.
- Usage: OpenAI's `prompt_tokens` includes cached tokens; they are reported separately
  internally and recombined for the client, so clients see OpenAI's numbers.

## Anthropic

### Request (OpenAI → Anthropic Messages)

| OpenAI field | Anthropic | Notes |
|---|---|---|
| Leading `system` / `developer` messages | `system` (text blocks) | |
| Later `system` / `developer` messages | `role: "system"` message, in place | Allowed by the API right after a user turn and before an assistant turn or the end. Kept in place, because moving it into `system` invalidates earlier thinking. Elsewhere it becomes a `[system] …` text block in the user turn. |
| `user` text and `image_url` parts | `text` and `image` blocks | Data URLs become `base64` sources (jpeg, png, gif, webp); http(s) URLs become `url` sources. Other part types (audio, files) are refused with a 400. |
| `assistant` content and `tool_calls` | `text` and `tool_use` blocks | `arguments` must be a JSON object; anything else is a 400 naming the call. |
| `tool` messages | `tool_result` blocks in one user turn | Consecutive tool messages are merged. |
| Consecutive same-role turns | Merged | The API would merge them too; merging here keeps the translation deterministic. |
| `tools[].function` | `tools[]` with `input_schema` | `strict` is passed through. |
| `tool_choice` `auto` / `none` / `required` / named | `auto` / `none` / `any` / `tool` | Models with `forced_tool_choice: false` (Opus 5.5, Sonnet 5.5 and Fable 5.2 reject forced choices) are skipped for such requests, so a fallback that supports it serves them; with none, a 400. |
| `parallel_tool_calls: false` | `disable_parallel_tool_use: true` | |
| `max_completion_tokens`, `max_tokens` | `max_tokens` | Required by Anthropic. Fallbacks: the model's `defaults.max_tokens`, then `max_output_tokens`, then 4096. |
| `reasoning_effort` | `output_config.effort` | `none` and `minimal` → `low`; `low`, `medium`, `high`, `xhigh` map directly. `max` is reachable through `provider_options`. |
| `response_format` `json_schema` | `output_config.format` | `json_object` has no equivalent (`json_object: false` in the model's capabilities), so such requests skip Claude models: a fallback serves them, or the client gets a 400. |
| `stop` | `stop_sequences` | |
| `user` | `metadata.user_id` | |
| `temperature`, `top_p` | Dropped | API version `2026-10-21` rejects them with a 400. A provider pinned to an older `api_version` gets them, with `temperature` clamped to 1. |
| Anything else (`seed`, `presence_penalty`, `logit_bias`, …) | Dropped | Listed in `x-switchboard-dropped-params`. With `unsupported_params: error` on the provider, not sent: the next candidate is tried, and with none the client gets a 400. |
| `extra_body.provider_options.anthropic` | Merged into the request last | Escape hatch, e.g. `{"thinking": {"type": "adaptive", "display": "summarized"}}`. |
| A conversation ending with an assistant message | Not sent | Current Claude models reject prefill. The next candidate is tried; with none, a 400. |

### Response (Anthropic stream → OpenAI chunks)

| Anthropic | OpenAI |
|---|---|
| `message_start` | The first chunk (`role: assistant`); `model` is the served model |
| `text_delta` | `delta.content` |
| `tool_use` block start, `input_json_delta` | `delta.tool_calls[k]` with id and name, then argument fragments. `k` counts tool calls only, not Anthropic block indexes. |
| `thinking`, `redacted_thinking` | Not forwarded (kept for the thinking sidecar) |
| `stop_reason` `end_turn`, `stop_sequence`, `pause_turn` | `finish_reason: stop` |
| `max_tokens`, `model_context_window_exceeded` | `length` |
| `tool_use` | `tool_calls` |
| `refusal` | `content_filter`; the refusal category is logged |
| Usage | `prompt_tokens` = input + cache reads + cache writes; `prompt_tokens_details.cached_tokens` = cache reads; `completion_tokens` = output (thinking included); `completion_tokens_details.reasoning_tokens` = `output_tokens_details.thinking_tokens` |
| `ping`, unknown event types | Ignored, as Anthropic's versioning policy asks |
| `error` event before any content | An HTTP error response |
| `error` event after content | An OpenAI-shaped SSE error event (the SDK raises `APIError`) |

### Errors

| Anthropic | Client sees |
|---|---|
| 401 `authentication_error`, 403 `permission_error` | 502 `upstream_auth_failed`: the gateway's key is the problem, not the client's |
| 402 `billing_error`; 429 with `details.error_code` `enforced_spend_limit_reached` or `contract_spend_limit_reached`; 400 "You have reached your specified API usage limits" | 503 `upstream_quota_exhausted` |
| 429 `rate_limit_error` (including `ramp_limit_exceeded`) | 429 `upstream_rate_limited`, with `retry-after` |
| 400 "prompt is too long" | 400 `context_length_exceeded` |
| Other 400, 404, 413 | 400 `upstream_bad_request`, with Anthropic's message |
| 409, 5xx, 529 `overloaded_error` | 502 `upstream_error` |

### Thinking blocks in tool loops

Claude models think during tool use, and Anthropic expects a tool-use turn's thinking
blocks back, unchanged, with the tool results. OpenAI-format clients cannot carry
them. Switchboard stores the turn's full content (Redis, or in-process memory without
Redis) keyed by its tool-call ids for an hour, and swaps it back in when the follow-up
request's echo of that turn matches. Otherwise the turn is sent without thinking,
which the API accepts. Outcomes are counted in
`switchboard_thinking_passback_total{result}`. Set `thinking_passback: false` on the
provider to turn it off. See [ADR-0009](adr/0009-thinking-sidecar.md).

## Ollama

Per Ollama's OpenAI-compatibility docs (2026-11-27):

- `max_completion_tokens` becomes `max_tokens` (the only one Ollama lists).
- `tool_choice` is not supported. `none` is expressed by removing `tools`; `auto` is
  the default anyway; forced choices are refused (configure `forced_tool_choice: false`).
- `user` and `parallel_tool_calls` are dropped and reported.
- Images must be base64 data URLs; http(s) image links are refused with a 400.
- Embeddings accept text only, not token ids. Local models are priced at zero
  (`config/pricing/`).

## Known gaps

- `n > 1` and `logprobs` are refused for every provider: the internal event model
  carries one choice and no log probabilities.
- Anthropic: `json_object`, audio and file inputs are refused; citations are kept for
  the sidecar but not exposed; server tools (web search) are not translated.
- Anthropic reasoning text is not exposed by default (current models default to
  `display: "omitted"`, so there is usually none to expose).
- Ollama: whether the streamed usage chunk is always present is not stated in its
  docs; if it is missing, the request's cost is estimated and marked `cost_estimated`
  ([ACCOUNTING.md](ACCOUNTING.md)).
- Fixtures under `tests/fixtures/` are built from the documented formats
  (`"source": "docs"`) until `make record-fixtures` records real ones.
