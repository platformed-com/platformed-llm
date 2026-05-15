use std::time::Duration;

use thiserror::Error;

/// Errors that can occur when using the platformed-llm library.
#[derive(Error, Debug)]
pub enum Error {
    /// Low-level HTTP transport failure (connect timeout, TLS error, etc.).
    ///
    /// Only present when the `reqwest` feature is enabled — that's the
    /// only path through which a `reqwest::Error` reaches us.
    #[cfg(feature = "reqwest")]
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),

    /// Authentication failure (typically a 401). The optional `status`
    /// is the HTTP status code we observed, if any.
    #[error("authentication failed{}{}", status_suffix(*status), .message)]
    Auth {
        /// HTTP status observed (typically 401 or 403), if available.
        status: Option<u16>,
        /// Provider-supplied error description.
        message: String,
    },

    /// JSON (de)serialization failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Provider-side error. Carries HTTP `status` plus a `retryable`
    /// hint: 5xx and 429 are retryable, 4xx generally isn't.
    #[error("provider error ({provider}{}{}", status_suffix(*status), .message)]
    Provider {
        /// Short identifier of the provider that raised the error
        /// (e.g. `"OpenAI"`, `"Google"`, `"Anthropic"`).
        provider: &'static str,
        /// HTTP status if the failure was an HTTP response.
        status: Option<u16>,
        /// `true` when the operation is safe to retry (5xx or 429).
        retryable: bool,
        /// Provider-supplied error description.
        message: String,
    },

    /// Caller misconfiguration (wrong env, invalid value).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Mid-stream / SSE parsing failure.
    #[error("streaming error: {0}")]
    Streaming(String),

    /// Rate limit hit (HTTP 429). `retry_after` is the parsed
    /// `Retry-After` header or equivalent, if any.
    #[error("rate limit exceeded{}{}", retry_after_suffix(*retry_after), .message)]
    RateLimit {
        /// Suggested wait duration from a `Retry-After` header, if the
        /// provider supplied one.
        retry_after: Option<Duration>,
        /// Provider-supplied error description.
        message: String,
    },

    /// Model not available (typically a 404 on the model name).
    #[error("model not available: {0}")]
    ModelNotAvailable(String),
}

impl Error {
    /// Build a provider error with sensible defaults. `status` and
    /// `retryable` default to `None` / `false` — callers that have the
    /// HTTP status should use [`Error::provider_with_status`].
    pub fn provider(provider: &'static str, message: impl Into<String>) -> Self {
        Error::Provider {
            provider,
            status: None,
            retryable: false,
            message: message.into(),
        }
    }

    /// Build a provider error with an HTTP status. `retryable` is
    /// inferred from the status (5xx + 429 → retryable).
    pub fn provider_with_status(
        provider: &'static str,
        status: u16,
        message: impl Into<String>,
    ) -> Self {
        let retryable = status == 429 || (500..=599).contains(&status);
        Error::Provider {
            provider,
            status: Some(status),
            retryable,
            message: message.into(),
        }
    }

    /// Build a configuration error (invalid env, missing required field, etc.).
    pub fn config(message: impl Into<String>) -> Self {
        Error::Config(message.into())
    }

    /// Build an auth error with no observed HTTP status.
    pub fn auth(message: impl Into<String>) -> Self {
        Error::Auth {
            status: None,
            message: message.into(),
        }
    }

    /// Build an auth error with an HTTP status (typically 401 or 403).
    pub fn auth_with_status(status: u16, message: impl Into<String>) -> Self {
        Error::Auth {
            status: Some(status),
            message: message.into(),
        }
    }

    /// Build a streaming error (SSE parse failure, out-of-order events, etc.).
    pub fn streaming(message: impl Into<String>) -> Self {
        Error::Streaming(message.into())
    }

    /// Build a rate-limit error. `retry_after_seconds` is parsed from
    /// the provider's `Retry-After` header or equivalent.
    pub fn rate_limit(retry_after_seconds: Option<u64>, message: impl Into<String>) -> Self {
        Error::RateLimit {
            retry_after: retry_after_seconds.map(Duration::from_secs),
            message: message.into(),
        }
    }
}

fn status_suffix(status: Option<u16>) -> String {
    match status {
        Some(s) => format!(", status {s}): "),
        None => "): ".to_string(),
    }
}

fn retry_after_suffix(retry_after: Option<Duration>) -> String {
    match retry_after {
        Some(d) => format!(" (retry after {}s): ", d.as_secs()),
        None => ": ".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_constructor_preserves_provider_and_message() {
        let err = Error::provider("OpenAI", "Test error");
        let msg = err.to_string();
        assert!(msg.contains("OpenAI"));
        assert!(msg.contains("Test error"));
        match err {
            Error::Provider {
                status, retryable, ..
            } => {
                assert_eq!(status, None);
                assert!(!retryable);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn provider_with_status_infers_retryable_for_5xx() {
        let err = Error::provider_with_status("OpenAI", 503, "service unavailable");
        match err {
            Error::Provider {
                status, retryable, ..
            } => {
                assert_eq!(status, Some(503));
                assert!(retryable, "5xx should be retryable");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn provider_with_status_marks_429_retryable() {
        let err = Error::provider_with_status("OpenAI", 429, "rate limited");
        match err {
            Error::Provider { retryable, .. } => assert!(retryable),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn provider_with_status_marks_4xx_non_retryable() {
        let err = Error::provider_with_status("OpenAI", 400, "bad request");
        match err {
            Error::Provider { retryable, .. } => assert!(!retryable),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn auth_constructor_renders_message() {
        let err = Error::auth("bad key");
        assert!(err.to_string().contains("bad key"));
        match err {
            Error::Auth { status, .. } => assert_eq!(status, None),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn auth_with_status_carries_code() {
        let err = Error::auth_with_status(401, "bad key");
        match err {
            Error::Auth { status, .. } => assert_eq!(status, Some(401)),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn rate_limit_converts_seconds_to_duration() {
        let err = Error::rate_limit(Some(42), "slow down");
        match err {
            Error::RateLimit { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(42)));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn config_constructor_renders_invalid_configuration_prefix() {
        let err = Error::config("Invalid model name");
        assert!(err.to_string().contains("invalid configuration"));
        assert!(err.to_string().contains("Invalid model name"));
    }
}
