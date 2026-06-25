use crate::{Capabilities, Error, Prompt, ProviderFileRef, RawConfig, Response};

/// A trait for LLM providers that can generate text responses.
///
/// Implementors translate the unified [`Prompt`] + [`RawConfig`] shape
/// into whatever wire format the upstream API expects, and stream
/// events back. The `Provider` trait itself is intentionally
/// "dumb" about polyfills and validation — that logic lives above the
/// provider in [`crate::middleware`] and is applied by the top-level
/// [`crate::generate`] function.
///
/// Most callers should use [`crate::generate`] rather than calling
/// `Provider::generate` directly — the free function applies the
/// middleware pipeline first so the provider sees an already-rewritten
/// request. Calling `Provider::generate` directly bypasses middleware.
///
/// All responses are internally streamed — use `response.stream()` for
/// streaming or `response.text().await` for buffered text.
#[async_trait::async_trait]
pub trait Provider: Send + Sync + 'static {
    /// Generate a chat completion from `prompt` and `config`.
    /// (Internally always streams; the result is a [`Response`] you can
    /// consume token-by-token or buffer.)
    async fn generate(&self, prompt: &Prompt, config: &RawConfig) -> Result<Response, Error>;

    /// Upload `bytes` to this provider's Files API and return a
    /// [`ProviderFileRef`] stamped with this provider's
    /// [`ProviderType`](crate::ProviderType). The returned reference can
    /// then be sent as a [`crate::FileSource::Uploaded`] file input.
    ///
    /// `file_name` is an optional hint some Files APIs accept for the
    /// stored object's filename.
    ///
    /// The default implementation returns [`Error::config`] — providers
    /// without a usable Files API (the in-process mock, local GGUF, and
    /// the Vertex-hosted Gemini / Claude providers, whose upstreams
    /// expose no Files API on Vertex) inherit it. Hosted providers with
    /// a real Files API override this.
    async fn upload(
        &self,
        bytes: bytes::Bytes,
        mime_type: &str,
        file_name: Option<&str>,
    ) -> Result<ProviderFileRef, Error> {
        let _ = (bytes, mime_type, file_name);
        Err(Error::config(
            "this provider does not support file uploads (no Files API)",
        ))
    }

    /// Report the [`Capabilities`] of `model` as understood by this
    /// provider. Called by [`crate::generate`] at request time to
    /// decide which middleware should run.
    ///
    /// Default impl delegates to [`Capabilities::for_model`] — that's
    /// correct for hosted providers whose model names share a globally
    /// unique namespace (`gpt-*`, `gemini-*`, `claude-*`). Providers
    /// whose models don't follow such a namespace — local inference,
    /// custom fine-tunes, novel backends — should override.
    fn capabilities(&self, model: &str) -> Capabilities {
        Capabilities::for_model(model)
    }
}
