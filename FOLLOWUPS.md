# Follow-ups

Open work deferred from prior sweeps. Items here are intentionally
skipped, not forgotten. Delete sections as they land.

## Testing gaps

The lib's request/response surface is now fully wired for all the
features we've enumerated. The remaining gaps are coverage — real
captured traces against each provider's APIs for surfaces that
exist in code but haven't been exercised end-to-end.

- **Anthropic real captures**: project needs Claude enabled in Vertex
  Model Garden. Replay/snapshot tests no-op for Anthropic until then.
  Anthropic integration coverage today is `function_calling_e2e`
  against hand-authored fixtures + provider unit tests only.
- **OpenAI typed-refusal channel**: the `refusal` scenario captures
  a real refusal from `gpt-4o-mini`, but the model emits it via the
  ordinary `output_text` channel (graceful plain-text refusal),
  *not* via the typed `response.refusal.delta` channel. The lib's
  `AssistantPart::Refusal` surface still has no captured wire-shape
  pinning — that fires only when OpenAI's safety system intercepts
  a request, which doesn't happen for the polite "please refuse"
  prompt we're using. Real coverage would require an actually-
  policy-violating prompt; deferred.
- **Anthropic multi-region captures**: the `us` / `eu` Vertex
  multi-region URL handling has unit-test coverage but no captured
  trace pins it against the real provider. Blocked on the same Model
  Garden access as the regional Anthropic captures.
- **Audio + Document captures**: `UserPart::Image` is now exercised
  by the `multi_modal_image` scenario against real OpenAI + Google.
  Audio and Document scenarios still missing — would need a small
  base64 mp3 / PDF asset (or reference files via path) in the
  scenarios schema.
- **Provider-builtin captures (partial)**: `WebSearch` is exercised
  by the `web_search` scenario against OpenAI. `GoogleSearch` /
  `CodeExecution` / `ComputerUse` still missing.
- **ProviderContinuation captures**: `OpenAI { response_id }` and
  `Gemini { cached_content }` wire through but no captured scenario
  exercises a real chained turn (would also need OpenAI's `store:
  true` so the response is retained server-side).
- **Header-config captures**: `OpenAIProvider::with_organization` /
  `with_project` and `AnthropicViaVertexProvider::with_beta` are
  unit-testable via the transport but no real captures verify the
  servers accept the headers.
