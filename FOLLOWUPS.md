# Follow-ups

Open work deferred from prior sweeps. Items here are intentionally
skipped, not forgotten. Delete sections as they land.

## Remaining provider features

### Gemini cachedContent

Gemini exposes prompt caching via a separate `cachedContent` API that
must be created server-side ahead of time. The lib's `CacheBreakpoint`
is currently a no-op on Gemini. Wiring would mean adding a
`cached_content` field on `LLMRequest` (or a Gemini-specific config)
that references a pre-created CachedContent resource name. Different
surface from `CacheBreakpoint`; both could coexist if we want
per-block hints AND server-side caches.

### Gemini structured output

`responseMimeType` / `responseSchema` (JSON-mode equivalent) — no
surface today. Adding `LLMRequest::response_format: Option<…>` and
threading it through each provider would cover OpenAI's
`response_format` and Gemini's `responseSchema` at once.

### OpenAI multi-org / project headers

`OpenAI-Organization` and `OpenAI-Project` headers are optional but
required for multi-org keys / project-scoped keys. Add an optional
field on `OpenAIProvider` (or a new constructor) that injects these.

### Anthropic beta headers

Several Anthropic features (computer use beta, fine-grained tool
streaming) require an `anthropic-beta` header. Similar treatment to
OpenAI org headers — an optional list of beta IDs on
`AnthropicViaVertexProvider`.

### Computer-use parameters

`ProviderBuiltin::ComputerUse` currently emits the bare type marker.
The real wire shape on OpenAI takes `display_width`, `display_height`,
`environment`; Anthropic takes similar fields. Either expand the
variant to carry a config struct or add a separate
`ComputerUseConfig` parameter.

## Testing gaps

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
  (`examples/capture_traces.rs`) and scenarios.json schema would need
  a way to express inline base64 bytes for a scenario message.
- **Provider-builtin captures**: `Tool::Builtin(WebSearch)` etc.
  produce the right wire shape per provider but no captured scenario
  exercises a real web-search round-trip.
