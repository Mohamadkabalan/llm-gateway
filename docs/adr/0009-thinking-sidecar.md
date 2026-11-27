# 0009: A thinking sidecar for Anthropic tool loops

- **Status:** accepted
- **Date:** 2026-11-27

## Context

Current Claude models think, and within a tool-use turn Anthropic requires the
assistant turn's thinking blocks to be passed back complete and unmodified with the
tool results. If they are all left out, the API accepts the request, but the model
loses the reasoning it did before calling the tool. If they are partly left out or
edited, the request fails with a 400. Changing the conversation's earlier prefix also
invalidates the thinking after the change.

OpenAI-format clients have no field to carry these blocks, so a straightforward
translation always drops them, and agent loops through the gateway would quietly get
worse than the same loops against Anthropic directly.

## Decision

- When a streamed Anthropic response stops for tool use and contains thinking, the
  adapter stores the turn's content blocks verbatim (thinking with its signature,
  redacted thinking, text, tool_use with parsed input), keyed by (scope, the sorted
  tool_use ids), for one hour. Storing runs in the background and never delays or
  fails the response.
- When a request's last assistant turn called tools and only tool results follow it,
  the adapter loads the stored blocks for those ids. They replace the translated turn
  only if the text and tool calls the client echoed match what was stored. Otherwise
  the turn is sent without thinking, which the API accepts.
- The store is Redis when configured (shared by every replica), else an in-process LRU
  with the same TTL. Redis errors count as misses.
- The request translation is deterministic, so a history the client echoes back
  produces the same Anthropic prefix byte for byte. A property test checks this.
- `switchboard_thinking_passback_total{result=hit|miss|mismatch|stored|error}` makes
  the effect visible. `thinking_passback: false` on a provider turns it off.

## Consequences

- Agent loops through the gateway keep their reasoning across tool calls, at the cost
  of one Redis GET on requests that end with tool results, and one SET per tool-calling
  turn that thought.
- The scope is a single shared value until virtual keys exist (Phase 4), when it
  becomes the team id, so teams never see each other's stored turns.
- Stored blocks contain the encrypted thinking signature and, with
  `display: "omitted"` (the default on current models), no readable reasoning text.
  They are still model output and expire after an hour.
- If a client rewrites the assistant turn, the swap is skipped and the turn loses its
  thinking, exactly as without the sidecar. A mismatch is counted, not an error.
