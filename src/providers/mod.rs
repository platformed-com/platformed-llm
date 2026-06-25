//! Provider implementations for different LLM services.
//!
//! Each provider is gated behind a Cargo feature so a leaner build can
//! drop unused HTTP / auth dependencies:
//!
//! - `openai` — OpenAI Responses API (`OpenAIProvider`).
//! - `google` — Google Gemini via Vertex AI (`GoogleProvider`).
//! - `anthropic-vertex` — Anthropic Claude via Vertex AI
//!   (`AnthropicViaVertexProvider`).
//! - `llama-gguf` — Local GGUF inference (`LlamaGgufProvider`).
//! - `mock` — In-process canned responses for testing (`MockProvider`).
//!
//! No features are enabled by default — opt in per provider.

#[cfg(feature = "mock")]
pub mod mock;
#[cfg(feature = "openai")]
mod openai;
#[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
pub(crate) mod part_tracker;
#[cfg(feature = "vertex")]
mod vertex;

#[cfg(feature = "llama-gguf")]
pub mod local;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;
#[cfg(feature = "anthropic-vertex")]
pub use vertex::AnthropicViaVertexProvider;
#[cfg(feature = "google")]
pub use vertex::GoogleProvider;
#[cfg(feature = "vertex")]
pub use vertex::VertexEndpoint;

#[cfg(feature = "llama-gguf")]
pub use local::LlamaGgufProvider;

#[cfg(feature = "mock")]
pub use mock::{CallLog, Chunking, MockProvider, MockProviderBuilder, MockResponse, RecordedCall};

/// Standard base64 (RFC 4648) encode with padding. The hosted
/// providers carry inline file bytes as base64 in their wire formats
/// (OpenAI data-URLs, Gemini `inlineData.data`, Anthropic
/// `source.data`); the canonical [`crate::FileSource::Bytes`] holds raw
/// bytes, so each provider encodes here. Kept dependency-free — a small
/// encoder beats pulling in the `base64` crate for one call site per
/// provider.
#[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        // Pad the final partial chunk: 1 byte → "==", 2 bytes → "=".
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Best-effort flatten of a tool-result content array into a single
/// string. Tool-result wire shapes accept only plain text on OpenAI's
/// `function_call_output`, Gemini's `functionResponse`, and the
/// Anthropic `tool_result` block — non-text parts (images, audio,
/// documents) have nowhere to land, so they're dropped with a
/// `tracing::debug!` so the loss is visible in logs.
#[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
pub(crate) fn flatten_user_parts_to_text(parts: &[crate::types::UserPart]) -> String {
    use crate::types::UserPart;
    let mut out = String::new();
    for part in parts {
        match part {
            UserPart::Text(s) => {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(s);
            }
            _ => {
                tracing::debug!("dropping non-text tool result part during request flatten");
            }
        }
    }
    out
}
