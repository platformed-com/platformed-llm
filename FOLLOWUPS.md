# Follow-ups

Open work deferred from prior sweeps. Items here are intentionally
skipped, not forgotten. Delete sections as they land.

## Phase 5 — remaining items

The Phase 5 redesign (canonical message model, part-indexed
StreamEvent, ProviderContinuation, Tool::Builtin, Error redesign with
typed status + retryable, Response::collect(), explicit re-exports,
from_env simplification, image multi-modal across all three providers,
Anthropic cache_control) has landed. The remaining Phase-5-adjacent
items are smaller and orthogonal:

### Multi-modal beyond images

`UserPart::Audio` and `UserPart::Document` are typed on the surface
but each provider silently drops them today. Provider wiring:

- **OpenAI**: `input_audio` (mp3/wav) and `input_file` (PDFs) content
  parts. Same `OpenAIMessageContent::Parts` machinery; just add the
  variants.
- **Anthropic**: `document` content blocks for PDFs; no audio support.
- **Gemini**: `inlineData` with the right mime types covers audio /
  documents already (binary just rides under a different content-type).

### OpenAI prompt_cache_key

`UserPart::CacheBreakpoint` currently no-ops on OpenAI. To map it
onto the per-request `prompt_cache_key`, derive a stable hash of the
prefix bytes up to the breakpoint and set it on the
`ResponsesRequest` once. Single-breakpoint only on OpenAI's side.

### Gemini cachedContent

Gemini exposes prompt caching via a separate `cachedContent` API
that must be created server-side ahead of time. Wiring would mean
adding a `cached_content` field on `LLMRequest` (or a Gemini-specific
config) referencing a pre-created CachedContent name. Different
surface from `CacheBreakpoint`; both could coexist.

### Provider-builtin wire shapes

`Tool::Builtin(ProviderBuiltin)` is enumerated and silently dropped
from the tools array on providers that don't offer that builtin, but
the providers that *do* offer them aren't emitting the right wire
shape yet. Each provider has its own:

- OpenAI: `{"type": "web_search_preview"}` / `{"type":
  "computer_use_preview", "display_width": ...}` entries in the tools
  array.
- Anthropic: `{"type": "web_search_20250305"}` / `{"type":
  "computer_20250124", ...}` entries.
- Gemini: `googleSearch` and `codeExecution` are *separate keys* on
  the request rather than entries in `tools`; needs a different
  conversion path.

### Gemini request-side thinkingConfig

Phase 3.5 added response-side `thoughtsTokenCount` parsing. The
request-side `generationConfig.thinkingConfig: { thinking_budget: N }`
that lets callers ask Gemini 2.5 to think is still unwired. Maps from
`ReasoningConfig.effort` like the Anthropic budget does.

## Unwired request fields

These reach `LLMRequest` but no provider serializes them:

- `LLMRequest::stop` — Gemini's `stopSequences`, OpenAI's `stop`,
  Anthropic's `stop_sequences`. None of the three providers thread it
  through.
- `LLMRequest::presence_penalty` / `frequency_penalty` — supported by
  OpenAI (non-reasoning models) and Gemini, ignored today.

## Other provider feature gaps

- **OpenAI `OpenAI-Organization` / `OpenAI-Project` headers**:
  optional, but multi-org / project-scoped keys won't work without
  them.
- **Anthropic `anthropic-beta` headers**: required for several beta
  features (computer use, fine-grained tool streaming).
- **Gemini `toolConfig`**: forced tool mode (`AUTO` / `ANY` / `NONE`)
  isn't exposed. The `ToolChoice` enum on `LLMRequest` is wired only
  on OpenAI; Anthropic and Gemini still ignore it.
- **Gemini structured output**: `responseMimeType` / `responseSchema`
  (JSON-mode equivalent) — no surface.

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
- **Real multi-modal captures**: nothing exercises image / audio /
  document input against real APIs yet — only synthetic round-trip
  via `Tool::Function` paths.
