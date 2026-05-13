//! A unified abstraction over multiple LLM providers.
//!
//! This library provides a consistent API for interacting with OpenAI,
//! Google Gemini (via Vertex AI), and Anthropic Claude (via Vertex AI),
//! with support for streaming responses and function calling.

pub mod accumulator;
pub mod error;
pub mod factory;
pub mod provider;
pub mod providers;
pub mod response;
pub mod sse_stream;
pub mod transport;
pub mod types;

// Explicit re-exports of the crate's public API surface. No globs —
// adding a `pub` item to an internal module must not leak it.

pub use accumulator::ResponseAccumulator;
pub use error::Error;
pub use factory::{ProviderConfig, ProviderFactory, ProviderType};
pub use provider::LLMProvider;
pub use providers::{
    AnthropicViaVertexProvider, GoogleProvider, OpenAIProvider, VertexAuth, VertexEndpoint,
};
pub use response::{CompleteResponse, OutputItem, Response};
pub use sse_stream::SseEvent;
pub use transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
pub use types::{
    Annotation, AnnotationKind, AssistantPart, AudioSource, DocumentSource, FinishReason,
    Function, FunctionCall, ImageSource, InputItem, LLMRequest, PartKind, PartUpdate, Prompt,
    ProviderBuiltin, ProviderContinuation, ReasoningConfig, ReasoningEffort, ReasoningSummary,
    ResponseFormat, StreamEvent, Tool, ToolChoice, ToolType, Usage, UserPart,
};
