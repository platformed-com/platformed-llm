use crate::{Config, Error, Prompt, Response};

/// A trait for LLM providers that can generate text responses.
///
/// All responses are internally streamed — use `response.stream()` for
/// streaming or `response.text().await` for buffered text.
#[async_trait::async_trait]
pub trait Provider: Send + Sync + 'static {
    /// Generate a chat completion from `prompt` and `config`.
    /// (Internally always streams; the result is a [`Response`] you can
    /// consume token-by-token or buffer.)
    async fn generate(&self, prompt: &Prompt, config: &Config) -> Result<Response, Error>;
}
