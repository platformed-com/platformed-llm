use crate::{Error, LLMRequest, Response};

/// A trait for LLM providers that can generate text responses.
/// All responses are internally streamed - use `response.stream()` for streaming
/// or `response.text().await` for buffered text.
#[async_trait::async_trait]
pub trait LLMProvider: Send + Sync + 'static {
    /// Generate a chat completion (internally always streams).
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error>;
}
