//! Google Vertex AI hosted providers — Gemini natively, plus Anthropic
//! Claude via Vertex's Model Garden. Both share a single endpoint /
//! auth surface ([`VertexEndpoint`]).

#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "anthropic")]
pub(crate) mod anthropic_types;
mod endpoint;
#[cfg(feature = "google")]
mod google;
#[cfg(feature = "google")]
pub(crate) mod google_types;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicViaVertexProvider;
pub use endpoint::VertexEndpoint;
#[cfg(feature = "google")]
pub use google::GoogleProvider;
