use thiserror::Error;

/// Errors that can occur when using the platformed-llm library.
#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Provider error: {provider} - {message}")]
    Provider { provider: String, message: String },

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Streaming error: {0}")]
    Streaming(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),
}

impl Error {
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Error::Provider {
            provider: provider.into(),
            message: message.into(),
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Error::Config(message.into())
    }

    pub fn auth(message: impl Into<String>) -> Self {
        Error::Auth(message.into())
    }

    pub fn streaming(message: impl Into<String>) -> Self {
        Error::Streaming(message.into())
    }
}
