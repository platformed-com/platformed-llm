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
- **Anthropic multi-region captures**: the `us` / `eu` Vertex
  multi-region URL handling has unit-test coverage but no captured
  trace pins it against the real provider. Blocked on the same Model
  Garden access as the regional Anthropic captures.
- **Audio + Document captures**: `UserPart::Image` is now exercised
  by the `multi_modal_image` scenario against real OpenAI + Google.
  Audio and Document scenarios still missing — would need a small
  base64 mp3 / PDF asset (or reference files via path) in the
  scenarios schema.
- **Provider-builtin captures (partial)**: `WebSearch` (OpenAI),
  `GoogleSearch` (Gemini), and `CodeExecution` (Gemini) are now
  captured *and* surface through typed
  `AssistantPart::BuiltinToolCall` + `Annotation` events.
  `ComputerUse` still missing — would need an actual computer-use
  session against OpenAI or Anthropic and isn't a one-shot prompt.
- **ProviderContinuation captures**: `OpenAI { response_id }` and
  `Gemini { cached_content }` wire through but no captured scenario
  exercises a real chained turn (would also need OpenAI's `store:
  true` so the response is retained server-side).
- **Header-config captures**: `OpenAIProvider::with_organization` /
  `with_project` and `AnthropicViaVertexProvider::with_beta` are
  unit-testable via the transport but no real captures verify the
  servers accept the headers.

## Response-side dropped metadata (observability only)

The lib is now lossless for content-shaped surfaces (text, tool calls,
reasoning summaries, citations, builtin tool invocations). Remaining
drops are pure metadata:

- **OpenAI**: `logprobs` on output_text deltas, `obfuscation` on
  deltas (anti-debug, intentional), `usage.input_tokens_details` /
  `output_tokens_details` sub-fields beyond `cached_tokens` and
  `reasoning_tokens`.
- **Gemini**: `safetyRatings` on candidates; `groundingMetadata`
  fields beyond chunks/supports (`webSearchQueries`,
  `searchEntryPoint`, `retrievalMetadata`); per-modality token
  breakdowns (`promptTokensDetails`, `candidatesTokensDetails`,
  `toolUsePromptTokenCount`).
- **Anthropic**: `stop_sequence` and `pause_turn` finish reasons
  collapse to `FinishReason::Stop`.

None of these block a caller from reading the model's emissions; lift
them onto the typed surface when a concrete consumer needs them.
