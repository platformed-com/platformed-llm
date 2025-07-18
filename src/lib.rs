//! A unified abstraction over multiple LLM providers.
//!
//! This library provides a consistent API for interacting with OpenAI, Google Gemini (via Vertex AI),
//! and Anthropic Claude (via Vertex AI), with support for streaming responses and function calling.

pub mod error;
pub mod types;
pub mod providers;
pub mod response;
pub mod streaming;
pub mod accumulator;

// Re-export core types for easy usage
pub use error::Error;
pub use types::*;
pub use providers::*;
pub use response::*;
pub use streaming::*;
pub use accumulator::*;
