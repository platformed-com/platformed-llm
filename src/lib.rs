//! A unified abstraction over multiple LLM providers.
//!
//! This library provides a consistent API for interacting with OpenAI,
//! Google Gemini (via Vertex AI), and Anthropic Claude (via Vertex AI),
//! with support for streaming responses and function calling.

#![deny(missing_docs)]

/// Manual stream-event accumulation. Most callers consume
/// [`Response`] / [`CompleteResponse`] and never touch this directly;
/// expose it for advanced users that drive the event stream themselves
/// (e.g. running the accumulator alongside a live UI handler).
pub mod accumulator;
/// Per-model capability table consulted by middleware to decide which
/// features can be requested natively vs. need a polyfill or drop.
pub mod capabilities;
/// Conversation compaction — summarize-and-rebuild support for
/// long-running sessions that would otherwise blow past the model's
/// context window. See [`compaction::Compactor`].
pub mod compaction;
/// Request/response middleware applied above the provider layer —
/// polyfills, validation, and the top-level [`generate`] entry point.
pub mod middleware;
/// Concrete provider implementations. Browse this module to see what
/// backends the lib supports and how to construct each one.
pub mod providers;
/// Server-Sent Events parser used by the default streaming response
/// path. Exposed for callers plugging a custom [`transport`] into a
/// non-default backend.
pub mod sse_stream;
/// HTTP transport abstraction. The default implementation is
/// `reqwest`-backed; callers can supply their own (recording,
/// retrying, replaying) [`transport::TransportImpl`] for testing or
/// fault injection.
pub mod transport;

// Test-only helpers for locating/downloading the integration suite's
// GGUF models, for reuse by downstream crates. Documented via its own
// module-level docs (`//!` in `test_util.rs`) so intra-doc links there
// resolve in the module's scope.
#[cfg(feature = "test-util")]
pub mod test_util;

// Internal modules — every public item below is re-exported at the
// crate root, so there's no value in users importing through the
// submodule path. Keep them private to keep the rustdoc table of
// contents focused on the canonical name.
mod error;
mod factory;
mod provider;
mod response;
mod types;

// Top-level re-exports — the everyday API surface. Concrete provider
// types live under the [`providers`] module so users can discover what
// backends are supported in one place. Lower-level surfaces (custom
// transports, manual stream-event handling) live in their submodules
// and are reachable via the fully-qualified path. No globs — adding a
// `pub` item to an internal module must not leak it.

pub use capabilities::Capabilities;
pub use compaction::Compactor;
pub use error::Error;
pub use factory::{ProviderConfig, ProviderFactory, ProviderType};
pub use middleware::{generate, JsonCoercionMiddleware, Middleware};
pub use provider::Provider;
pub use response::{CompleteResponse, Response};
pub use types::{
    Annotation, AnnotationKind, AssistantPart, AudioSource, ComputerUseConfig, Config,
    ConfigBuilder, DocumentSource, FinishReason, Function, FunctionCall, ImageSource, InputItem,
    PartKind, PartUpdate, Prompt, ProviderBuiltin, ProviderContinuation, RawConfig,
    ReasoningConfig, ReasoningEffort, ReasoningSummary, ResponseFormat, StreamEvent, Tool,
    ToolChoice, Usage, UserPart,
};
