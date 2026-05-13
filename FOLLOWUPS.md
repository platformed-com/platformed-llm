# Follow-ups

Open work deferred from prior sweeps. Items here are intentionally
skipped, not forgotten. Delete sections as they land.

## Phase 5 — breaking API redesign

These each touch every consumer of the crate, so they deserve their own
dedicated commit/PR rather than being mixed into bug fixes.

### Vec<ContentPart> message model

The single biggest gap. `Message::content: String` and `OutputItem::Text`
can't represent:
- Multi-modal (images / audio in user messages).
- Reasoning blocks with `signature` (Anthropic). Currently
  `OutputItem::Reasoning { content, signature }` carries the signature,
  but `OutputItem::to_input_item` drops it on the way back into a
  request — so multi-turn thinking can't actually round-trip.
- Anthropic `redacted_thinking` blocks (opaque blob; pass-through only).
- Mixed text + tool_use within a single assistant turn (we currently
  split them into separate `InputItem`s, which loses ordering when
  reading back).

**Plan**:
```rust
pub enum ContentPart {
    Text(String),
    Image { source: ImageSource },
    Reasoning { content: String, signature: Option<String> },
    RedactedReasoning { data: String },
}

pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

pub enum OutputItem {
    Message { content: Vec<ContentPart> },
    FunctionCall { call: FunctionCall },
}
```

Each provider's `convert_request` walks parts and emits the right wire
shape; the accumulator coalesces adjacent text deltas into a single
`ContentPart::Text`. Anthropic's signature now round-trips correctly.

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

- **OpenAI refusal handling**: `response.refusal.delta` / `.done` events
  are silently dropped. Should surface as a typed `StreamEvent::Refusal`
  (or fold into `Error::ContentFilter`). No captured trace currently
  exercises a refusal — needs a prompt that elicits one reliably.
- **OpenAI `OpenAI-Organization` / `OpenAI-Project` headers**: optional,
  but multi-org / project-scoped keys won't work without them.
- **Anthropic `cache_control: Ephemeral`** on tools / system / message
  blocks: prompt caching gate. Big cost lever.
- **Anthropic `anthropic-beta` headers**: required for several beta
  features (computer use, fine-grained tool streaming).
- **Gemini `toolConfig`**: forced tool mode (`AUTO` / `ANY` / `NONE`)
  isn't exposed. The `ToolChoice` enum on `LLMRequest` is wired only on
  OpenAI; Anthropic and Gemini still ignore it.
- **Gemini structured output**: `responseMimeType` /
  `responseSchema` (JSON mode equivalent) — no surface.

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
