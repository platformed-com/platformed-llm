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

#[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
pub(crate) mod file_resolve;
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

/// Reject a prompt that carries an input modality the target provider can't
/// accept. Run at the top of `generate()` so the caller gets a clear
/// [`Error::UnsupportedInput`](crate::Error::UnsupportedInput) instead of the
/// model silently answering without the (otherwise dropped) content.
///
/// Image and document inputs are accepted by every provider, so only audio /
/// video need gating; pass `false` for a modality the provider doesn't
/// support. Walks nested tool-result content too.
///
/// Only OpenAI and Anthropic gate modalities today (Gemini accepts all), so
/// the helper is gated to those features.
#[cfg(any(feature = "openai", feature = "anthropic-vertex"))]
pub(crate) fn reject_unsupported_modalities(
    items: &[crate::types::InputItem],
    provider: &'static str,
    supports_audio: bool,
    supports_video: bool,
) -> Result<(), crate::Error> {
    use crate::types::{InputItem, UserPart};

    fn check(
        parts: &[UserPart],
        provider: &'static str,
        audio: bool,
        video: bool,
    ) -> Result<(), crate::Error> {
        for part in parts {
            match part {
                UserPart::Audio(_) if !audio => {
                    return Err(crate::Error::unsupported_input(provider, "audio"));
                }
                UserPart::Video(_) if !video => {
                    return Err(crate::Error::unsupported_input(provider, "video"));
                }
                UserPart::ToolResult { content, .. } => check(content, provider, audio, video)?,
                _ => {}
            }
        }
        Ok(())
    }

    for item in items {
        if let InputItem::User { content } = item {
            check(content, provider, supports_audio, supports_video)?;
        }
    }
    Ok(())
}

#[cfg(all(test, any(feature = "openai", feature = "anthropic-vertex")))]
mod modality_tests {
    use super::reject_unsupported_modalities;
    use crate::types::{FileSource, InputItem, UserPart};
    use crate::Error;

    fn user(parts: Vec<UserPart>) -> InputItem {
        InputItem::User { content: parts }
    }

    #[test]
    fn audio_and_video_rejected_when_unsupported() {
        let audio = vec![user(vec![UserPart::Audio(FileSource::Url("a".into()))])];
        let err = reject_unsupported_modalities(&audio, "OpenAI", false, false)
            .expect_err("audio should be rejected");
        assert!(matches!(
            err,
            Error::UnsupportedInput {
                provider: "OpenAI",
                modality: "audio"
            }
        ));

        let video = vec![user(vec![UserPart::Video(FileSource::Url("v".into()))])];
        let err = reject_unsupported_modalities(&video, "Anthropic", false, false)
            .expect_err("video should be rejected");
        assert!(matches!(
            err,
            Error::UnsupportedInput {
                modality: "video",
                ..
            }
        ));
    }

    #[test]
    fn accepted_when_supported_or_other_modalities() {
        // Supported provider: no error.
        let media = vec![user(vec![
            UserPart::Audio(FileSource::Url("a".into())),
            UserPart::Video(FileSource::Url("v".into())),
        ])];
        assert!(reject_unsupported_modalities(&media, "Google", true, true).is_ok());

        // Image/document/text never trip the check.
        let other = vec![user(vec![
            UserPart::Text("hi".into()),
            UserPart::Image(FileSource::Url("i".into())),
            UserPart::Document(FileSource::Url("d".into())),
        ])];
        assert!(reject_unsupported_modalities(&other, "OpenAI", false, false).is_ok());
    }

    #[test]
    fn nested_tool_result_audio_rejected() {
        let nested = vec![user(vec![UserPart::ToolResult {
            call_id: "c1".into(),
            content: vec![UserPart::Audio(FileSource::Url("a".into()))],
        }])];
        assert!(reject_unsupported_modalities(&nested, "OpenAI", false, false).is_err());
    }
}
