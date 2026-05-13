# Follow-ups

Open work deferred from prior sweeps. Items here are intentionally
skipped, not forgotten. Delete sections as they land.

## Phase 5 — breaking API redesign

These each touch every consumer of the crate, so they deserve their own
dedicated commit/PR rather than being mixed into bug fixes.

### Guiding principles

The redesign aims to support **mid-conversation provider switching** as
a first-class capability. Concretely:

1. **Lossless canonical message model.** Every content type from every
   supported provider has a representation in the internal model.
   Round-tripping through the *same* provider preserves everything.
2. **Silent, lossy egress when the target provider doesn't support a
   variant.** No errors — switching providers in the middle of a
   conversation must always *work*, even if quality degrades (less
   context, no caching, no reasoning continuation). The lib drops or
   best-effort-translates per-target.
3. **Provider-specific optimizations are hints, not requirements.**
   `previous_response_id`, cache breakpoints, provider-builtin tools —
   all opt-in. The lib always supports running with none of them, and
   silently ignores hints that don't apply to the target provider.
4. **Opaque continuation tokens.** Responses carry provider-specific
   state that the caller can pass back. The issuing provider uses it as
   an optimization; other providers ignore it and fall back to sending
   full message history.

### Canonical message model

The single biggest piece. `Message::content: String` and
`OutputItem::Text` can't represent:
- Multi-modal user content (images, audio, documents).
- Reasoning blocks with `signature` (Anthropic). Currently
  `OutputItem::Reasoning { content, signature }` carries the signature,
  but `OutputItem::to_input_item` drops it on the way back into a
  request — so multi-turn thinking can't actually round-trip.
- Anthropic `redacted_thinking` blocks (opaque blob; pass-through only).
- Interleaved text + thinking + tool_use within a single assistant turn
  on Anthropic (we currently split them into separate `InputItem`s,
  which loses ordering).
- OpenAI typed `refusal` content parts.
- Per-text-part citations / annotations.
- Anthropic per-block `cache_control` hints (90% cost reduction on
  cached prefixes — major lever).

**Plan**:
```rust
pub enum InputItem {
    System(String),
    User { content: Vec<UserPart> },
    Assistant { content: Vec<AssistantPart> }, // includes ToolCall, in emit order
}

pub enum UserPart {
    Text(String),
    Image(ImageSource),
    Audio(AudioSource),
    Document(DocumentSource),
    /// Tool execution result. `call_id` correlates with a prior
    /// `AssistantPart::ToolCall` in the same conversation. The content is
    /// itself a `Vec<UserPart>` so tool results can carry images / docs
    /// (e.g. a tool that returns a chart).
    ToolResult { call_id: String, content: Vec<UserPart> },
    /// Anthropic-only: marks the end of a cacheable prefix. Up to 4
    /// breakpoints per request. On OpenAI we derive a stable
    /// `prompt_cache_key` from a hash of the cached prefix (best-effort
    /// equivalent). Dropped on Gemini.
    CacheBreakpoint,
}

pub enum AssistantPart {
    Text {
        content: String,
        /// Per-text citations, web-search results, file references.
        /// Provider-specific shapes flatten into a uniform Annotation list.
        annotations: Vec<Annotation>,
    },
    Reasoning {
        content: String,
        /// Anthropic's `thinking` signature. Dropped on cross-provider
        /// conversion — the signature is meaningful only to the model
        /// that produced it.
        signature: Option<String>,
    },
    /// Anthropic `redacted_thinking` — opaque blob passed back to
    /// Anthropic unchanged. Dropped on cross-provider conversion.
    RedactedReasoning { data: String },
    /// OpenAI typed refusal channel. Translated to plain text on
    /// providers that don't surface refusals separately (so the caller
    /// still sees the content).
    Refusal(String),
    ToolCall(FunctionCall),
    /// Anthropic cache breakpoint marker. Same semantics as on `UserPart`.
    CacheBreakpoint,
}

pub struct Annotation {
    pub kind: AnnotationKind, // UrlCitation | FileCitation | WebSearch | …
    pub start: usize, // byte offset in the containing text
    pub end: usize,
    pub source: String,
}

pub enum ImageSource {
    Url(String),
    Base64 { data: String, media_type: String },
}
// AudioSource and DocumentSource follow the same shape.

pub enum OutputItem {
    /// The model's full assistant turn, parts in emit order. Tool calls,
    /// reasoning, refusals, and text are all parts of one turn.
    Assistant { content: Vec<AssistantPart> },
}
```

Each provider's `convert_request` walks the parts and:
- **Anthropic**: emits each `AssistantPart` as a content block in the
  assistant message — natural 1:1 mapping. Cache breakpoints become
  `cache_control: {type: "ephemeral"}` on the preceding block.
- **OpenAI**: flattens. Text → message content part. Reasoning → top-
  level `reasoning` input item. ToolCall → top-level `function_call`
  item. Refusal → message content part of type `refusal`. Cache
  breakpoints → derive `prompt_cache_key` from a stable hash of bytes
  up to that point.
- **Gemini**: emits each `AssistantPart` as a part inside `role: model`
  content. Reasoning text dropped (Gemini doesn't accept thinking back
  as input). ToolCall → `functionCall` part. Refusal → plain text part.
  Cache breakpoints dropped.

The accumulator coalesces adjacent text deltas into a single
`AssistantPart::Text` per logical text span.

### Provider continuation tokens

OpenAI's `previous_response_id` (and any future analogue from other
providers) is a request-time optimization that lets the provider elide
the message history when the previous response is still in
server-side state.

```rust
pub struct LLMRequest {
    // existing fields…
    /// Optimization hint. If the target provider matches the issuer, use
    /// it to skip re-sending history. Otherwise ignored — the lib falls
    /// back to sending the full conversation.
    pub continuation: Option<ProviderContinuation>,
}

pub enum ProviderContinuation {
    OpenAI { response_id: String },
    // Anthropic / Gemini have no analogous concept today. Adding one
    // later is non-breaking.
}

pub struct CompleteResponse {
    pub output: Vec<OutputItem>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    /// Opaque token the caller can pass to the next request as
    /// `LLMRequest::continuation`. Only meaningful when the next call
    /// goes to the same provider.
    pub continuation: Option<ProviderContinuation>,
}
```

This is purely additive. Callers who don't care never see it; callers
who do see meaningful token-cost savings on multi-turn reasoning.

### Provider-builtin tools

Each provider has its own pre-baked tools (`web_search`, `computer_use`,
Gemini code execution, Google search retrieval). These are different
from user-defined function tools — they're configured by name and the
model invokes them via wire-shape that's provider-specific.

```rust
pub enum Tool {
    Function(FunctionTool),
    /// Provider-builtin: dropped from the tools array on providers that
    /// don't offer it.
    Builtin(ProviderBuiltin),
}

pub enum ProviderBuiltin {
    WebSearch,         // OpenAI, Anthropic (different invocations)
    GoogleSearch,      // Gemini
    CodeExecution,     // Gemini
    ComputerUse(ComputerConfig), // OpenAI, Anthropic
}
```

Like everything else, unsupported builtins are silently dropped on
egress. The model still gets a valid tools array, just smaller.

### Error redesign

Currently `Error::Provider { provider, message }` swallows the HTTP
status and the structured provider error body, and `Error::Auth(String)`
/ `Error::Streaming(String)` are stringly typed. After the OpenAI
typed-error work in Phase 2.6, the variant shapes are inconsistent
across providers — and the Vertex providers (Google, Anthropic) still
collapse every non-2xx into `Error::Provider("Google", "API error: …")`
even though the captured 401 body contains `"status": "UNAUTHENTICATED"`
and the 404 contains `"status": "NOT_FOUND"` — easy mappings to
`Error::Auth` / `Error::ModelNotAvailable` once the shape allows it.

**Plan**:
```rust
pub enum Error {
    Auth { provider: ProviderType, status: Option<u16>, message: String },
    InvalidRequest { status: u16, message: String, body: Option<Box<RawValue>> },
    RateLimit { status: u16, retry_after: Option<Duration>, message: String },
    Provider { status: u16, retryable: bool, message: String, body: Option<Box<RawValue>> },
    Streaming { source: Box<dyn std::error::Error + Send + Sync> },
    Transport(reqwest::Error),
    Serialization(serde_json::Error),
    Cancelled,
}
```

All three provider error mappers route into these variants; consumers
get `retry_after` / `retryable` for free. Once landed, `error_traces_e2e`
should tighten to assert `Error::Auth` on Vertex 401 instead of accepting
any `Error::Provider`.

### Response::collect()

`Response::text()` and `Response::buffer()` each consume the response
once, so callers who want both the streamed events AND the final
buffered text have to call `provider.generate(...)` twice. The examples
work around this exact problem.

**Plan**: add `Response::collect() -> Result<(Vec<StreamEvent>,
CompleteResponse), Error>` that returns both. Deprecate `text()` if
collect makes it redundant.

### Drop `pub use providers::*; pub use types::*;` globs

`src/lib.rs:20-23`. Glob re-exports make the public API surface
implicit and turn an accidentally-leaked `pub(crate)` helper into a
breaking change. Replace with explicit re-exports of named items.

### `from_env()` simplification

`src/factory.rs::ProviderConfig::from_env` has ~60 lines of
credential-sniffing fallbacks that try to guess the provider type from
which env var happens to be set. The behaviour is now pinned by 9 unit
tests but it's still ambiguous from the user side. Drop the fallback;
require `PROVIDER_TYPE` to be set explicitly. Document the four envs
each provider reads. Tests update with the simplification.

## Reasoning / thinking gaps

Mostly blocked on the `Vec<ContentPart>` redesign above:

- `OutputItem::Reasoning::to_input_item` drops the content + signature
  on round-trip. Once messages carry parts, fold reasoning back into the
  assistant message so multi-turn extended thinking works.
- Anthropic `redacted_thinking` blocks are parsed but discarded. They
  must be echoed back unchanged to preserve continuity for sensitive
  thinking turns.
- Gemini request-side `generationConfig.thinkingConfig` is **not** wired.
  Phase 3.5 added response-side `thoughtsTokenCount` parsing; the
  request-side budget setting (so callers can ask Gemini 2.5 to think)
  still needs `thinkingConfig: { thinking_budget: N }` on the
  `GoogleGenerationConfig`.

## Unwired request fields

These reach `LLMRequest` but no provider serializes them:

- `LLMRequest::stop` — Gemini's `stopSequences`, OpenAI's `stop`,
  Anthropic's `stop_sequences`. None of the three providers thread it
  through.
- `LLMRequest::presence_penalty` / `frequency_penalty` — supported by
  OpenAI (non-reasoning models) and Gemini, ignored today.

Easy fix once decided whether to wire them or remove them from
`LLMRequest` (Phase 5's API redesign is a natural moment to decide).

## Other provider feature gaps

Some of these are subsumed by Phase 5 once the canonical message model
and the provider-builtin tool surface land — flagged below.

- **OpenAI refusal handling**: `response.refusal.delta` / `.done` events
  are silently dropped. *Phase 5 subsumes* — `AssistantPart::Refusal`
  is the typed surface; this becomes wiring it up.
- **OpenAI `OpenAI-Organization` / `OpenAI-Project` headers**: optional,
  but multi-org / project-scoped keys won't work without them.
- **Anthropic `cache_control: Ephemeral`** on tools / system / message
  blocks: prompt caching gate. Big cost lever. *Phase 5 subsumes* via
  `UserPart::CacheBreakpoint` / `AssistantPart::CacheBreakpoint`.
- **Anthropic `anthropic-beta` headers**: required for several beta
  features (computer use, fine-grained tool streaming).
- **Gemini `toolConfig`**: forced tool mode (`AUTO` / `ANY` / `NONE`)
  isn't exposed. The `ToolChoice` enum on `LLMRequest` is wired only on
  OpenAI; Anthropic and Gemini still ignore it.
- **Gemini structured output**: `responseMimeType` /
  `responseSchema` (JSON mode equivalent) — no surface.
- **Provider-builtin tools** (OpenAI web_search, Anthropic computer_use,
  Gemini google_search / code execution). *Phase 5 subsumes* via the
  `Tool::Builtin(ProviderBuiltin)` variant.

## Testing gaps

- **Anthropic real captures**: project needs Claude enabled in Vertex
  Model Garden. Replay/snapshot tests no-op for Anthropic until then.
  Anthropic integration coverage today is `function_calling_e2e` against
  hand-authored fixtures + provider unit tests only.
- **Vertex typed-error mapping**: `error_traces_e2e` accepts any
  `Error::Provider` for Vertex 4xx because the provider doesn't map
  the structured `google.rpc.Status` envelope. Tightens to assert
  `Error::Auth` / `Error::ModelNotAvailable` once the Phase 5 Error
  redesign lands.
- **OpenAI refusal coverage**: still untested; needs a prompt that
  reliably elicits a refusal without poking policy boundaries.
- **Anthropic multi-region captures**: the `us` / `eu` Vertex
  multi-region URL handling has unit-test coverage but no captured
  trace pins it against the real provider. Blocked on the same Model
  Garden access as the regional Anthropic captures.
