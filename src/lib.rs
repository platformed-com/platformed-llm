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

pub use error::Error;
pub use factory::{ProviderConfig, ProviderFactory, ProviderType};
pub use provider::Provider;
pub use response::{CompleteResponse, Response};
pub use types::{
    Annotation, AnnotationKind, AssistantPart, AudioSource, ComputerUseConfig, Config,
    DocumentSource, FinishReason, Function, FunctionCall, ImageSource, InputItem, PartKind,
    PartUpdate, Prompt, ProviderBuiltin, ProviderContinuation, ReasoningConfig, ReasoningEffort,
    ReasoningSummary, ResponseFormat, StreamEvent, Tool, ToolChoice, Usage, UserPart,
};
