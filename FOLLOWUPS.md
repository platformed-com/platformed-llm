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
- **OpenAI refusal coverage**: `AssistantPart::Refusal` is the typed
  surface, but no captured trace exercises a real refusal — needs a
  prompt that reliably elicits one without poking policy boundaries.
- **Anthropic multi-region captures**: the `us` / `eu` Vertex
  multi-region URL handling has unit-test coverage but no captured
  trace pins it against the real provider. Blocked on the same Model
  Garden access as the regional Anthropic captures.
- **Real multi-modal captures**: `UserPart::Image/Audio/Document` are
  wired through all three providers but no captured scenario
  exercises them against real APIs yet. The capture binary
  (`examples/capture_traces.rs`) and `tests/scenarios.json` schema
  would need a way to express inline base64 bytes for a scenario
  message.
- **Provider-builtin captures**: `Tool::Builtin(WebSearch /
  GoogleSearch / CodeExecution / ComputerUse)` produce the right
  wire shape per provider but no captured scenario exercises a real
  web-search / code-execution round-trip.
- **Structured-output captures**: `ResponseFormat::JsonObject` /
  `JsonSchema` wired on OpenAI and Gemini but no captured scenario
  exercises them. JSON mode is a high-value feature to pin against
  real APIs.
- **ProviderContinuation captures**: `OpenAI { response_id }` and
  `Gemini { cached_content }` wire through but no captured scenario
  exercises a real chained turn (would also need OpenAI's `store:
  true` so the response is retained server-side).
- **Header-config captures**: `OpenAIProvider::with_organization` /
  `with_project` and `AnthropicViaVertexProvider::with_beta` are
  unit-testable via the transport but no real captures verify the
  servers accept the headers.
