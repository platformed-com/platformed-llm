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
    #[error("provider error ({provider}{}): {message}", status_suffix(*status))]
    Provider {
        /// Short identifier of the provider that raised the error
        /// (e.g. `"OpenAI"`, `"Google"`, `"Anthropic"`).
        provider: &'static str,
        /// HTTP status if the failure was an HTTP response.
        status: Option<u16>,
        /// `true` when the operation is safe to retry (5xx or 429).
        retryable: bool,
        /// Provider-supplied wait hint from a `Retry-After` header (or
        /// equivalent). RFC 7231 explicitly defines `Retry-After` on
        /// 503s as well as 429s, so 5xx responses can carry one too —
        /// the retry helper surfaces it via [`Self::retry_after`].
        retry_after: Option<Duration>,
        /// Provider-supplied error description.
        message: String,
    },

    /// Caller misconfiguration (wrong env, invalid value).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// The prompt is structurally invalid for the target provider, caught
    /// client-side before the HTTP round trip. Lets the caller learn the
    /// rule from a typed error instead of decoding an opaque provider 400.
    ///
    /// Raised for provider-specific prompt-shape requirements that aren't
    /// expressible in the type system — e.g. Gemini requires at least one
    /// user/assistant turn, and requires each assistant turn's
    /// `functionCall` count to match the following user turn's
    /// `functionResponse` count.
    #[error("invalid prompt: {0}")]
    InvalidPrompt(String),

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

    /// Request rejected because the prompt exceeded the model's
    /// context window. Distinct from a generic
    /// [`Self::Provider`] error so callers running long-lived
    /// conversations can detect it cheaply and trigger compaction.
    ///
    /// Detection is best-effort: OpenAI reports this reliably via
    /// `code: "context_length_exceeded"`; Anthropic and Google are
    /// detected via message-string matching (their schemas don't
    /// expose a stable typed code), so some context-exceeded errors
    /// may still arrive as [`Self::Provider`] when the upstream
    /// wording changes.
    #[error("context window exceeded ({provider}): {message}")]
    ContextWindowExceeded {
        /// Short identifier of the provider that raised the error
        /// (e.g. `"OpenAI"`, `"Anthropic"`, `"Google"`).
        provider: &'static str,
        /// Provider-supplied error description.
        message: String,
    },

    /// Compaction couldn't produce a usable memo — the
    /// summarisation model returned no usable text (empty, refusal,
    /// pure tool-call, content-filtered) or was truncated by an
    /// output-token budget mid-memo. The lib propagates rather
    /// than committing a degenerate memo because the only
    /// alternative is silently destroying history; the caller can
    /// retry with a larger summarisation budget, switch the
    /// summarisation model, surface to the user, etc. The original
    /// prompt is untouched — `compact()` is transactional in the
    /// failure case.
    #[error("compaction failed: {reason}")]
    Compaction {
        /// Human-readable description of what went wrong (empty
        /// summary, truncation, etc.).
        reason: String,
    },

    /// The prompt carries an input modality the target provider cannot
    /// accept (e.g. audio or video on OpenAI / Anthropic). Surfaced as a
    /// hard error rather than silently dropping the content — otherwise the
    /// model would answer without input the caller explicitly provided.
    /// Distinct variant so callers orchestrating provider fallback can branch
    /// on it (e.g. route audio/video to Gemini).
    #[error("{provider} does not support {modality} input")]
    UnsupportedInput {
        /// Short identifier of the provider (e.g. `"OpenAI"`, `"Anthropic"`).
        provider: &'static str,
        /// The unsupported modality (`"audio"`, `"video"`).
        modality: &'static str,
    },
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
            retry_after: None,
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
            retry_after: None,
            message: message.into(),
        }
    }

    /// Build a provider error with an HTTP status and a
    /// `Retry-After`-derived wait hint. Use this when a 5xx (or any
    /// non-429 status that still carried a `Retry-After`) provides a
    /// server-suggested backoff — surfaced via [`Self::retry_after`]
    /// so retry policies honour it the same way they do for 429s.
    /// For 429s use [`Self::rate_limit`] instead — that variant
    /// preserves the distinct rate-limit shape.
    pub fn provider_with_retry_after(
        provider: &'static str,
        status: u16,
        retry_after_seconds: Option<u64>,
        message: impl Into<String>,
    ) -> Self {
        let retryable = status == 429 || (500..=599).contains(&status);
        Error::Provider {
            provider,
            status: Some(status),
            retryable,
            retry_after: retry_after_seconds.map(Duration::from_secs),
            message: message.into(),
        }
    }

    /// Build a configuration error (invalid env, missing required field, etc.).
    pub fn config(message: impl Into<String>) -> Self {
        Error::Config(message.into())
    }

    /// Build an invalid-prompt error — the prompt's structure violates a
    /// provider requirement that can be detected before sending.
    pub fn invalid_prompt(message: impl Into<String>) -> Self {
        Error::InvalidPrompt(message.into())
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

    /// Build a rate-limit error. `retry_after_seconds` is parsed from
    /// the provider's `Retry-After` header or equivalent.
    pub fn rate_limit(retry_after_seconds: Option<u64>, message: impl Into<String>) -> Self {
        Error::RateLimit {
            retry_after: retry_after_seconds.map(Duration::from_secs),
            message: message.into(),
        }
    }

    /// Build a context-window-exceeded error. Use this when a provider
    /// 400 carries an unambiguous "too many tokens" signal.
    pub fn context_window_exceeded(provider: &'static str, message: impl Into<String>) -> Self {
        Error::ContextWindowExceeded {
            provider,
            message: message.into(),
        }
    }

    /// Build a compaction failure error. Use when
    /// `Compactor::compact` couldn't produce a usable memo (empty
    /// summary, refusal, truncated).
    pub fn compaction(reason: impl Into<String>) -> Self {
        Error::Compaction {
            reason: reason.into(),
        }
    }

    /// Build an unsupported-input error for a modality the target provider
    /// can't accept (e.g. `("OpenAI", "audio")`).
    pub fn unsupported_input(provider: &'static str, modality: &'static str) -> Self {
        Error::UnsupportedInput { provider, modality }
    }

    /// Whether this error represents a transient failure where
    /// re-issuing the same request is likely to behave differently
    /// next time.
    ///
    /// Returns `true` for [`Self::RateLimit`], for [`Self::Transport`]
    /// **only when** the wrapped `reqwest::Error` is a connect or
    /// timeout failure (the unambiguously transient network shapes
    /// — request-build, body-read, decode, and startup errors stay
    /// terminal because they could equally be deterministic bugs),
    /// and for [`Self::Provider`] when its `retryable` flag is set
    /// (5xx / 429, mid-stream connection-drop errors that we
    /// classified as transient at their site). All other variants
    /// are terminal — re-issuing the same request won't change the
    /// outcome (bad auth, malformed prompt, model unavailable,
    /// context-window-exceeded, etc.).
    ///
    /// **"Retryable" is not the same as "safe to retry without
    /// thought."** Every retry is a fresh request — the model's
    /// reply may diverge from whatever the first attempt streamed.
    /// If the caller has already shown partial output to a user or
    /// committed it downstream, the new attempt's output won't
    /// stitch cleanly with what was shown. That's a caller-policy
    /// concern — pre-stream vs mid-stream timing isn't a variant
    /// property.
    ///
    /// Pair this with [`Self::retry_after`] when building a manual
    /// retry loop, or hand off to [`crate::retry()`] /
    /// [`crate::RetryPolicy`] which package both into a closure-based
    /// helper. See the [`mod@crate::retry`] module docs for the
    /// buffered vs streaming patterns.
    pub fn is_retryable(&self) -> bool {
        match self {
            #[cfg(feature = "reqwest")]
            Error::Transport(e) => {
                // `reqwest::Error` is a grab bag, and the network-
                // shape predicates overlap. The bucket covers the
                // failure modes that are *most often* transient even
                // though a minority of each can be deterministic:
                //
                // - `is_connect()` / `is_timeout()` / `is_request()` —
                //   pre-flight network shapes (TLS handshake reset,
                //   connect timeout, DNS hiccup, body-build failure
                //   that surfaces at `send()` time).
                // - `is_body()` — mid-body failures. These include
                //   legitimate connection drops (the common case)
                //   *and* deterministic decode failures (gzip
                //   corruption, broken UTF-8). We deliberately err on
                //   the side of over-retrying: a connection drop
                //   mid-stream is by far the more common shape, and
                //   silently giving up on it (the false-negative the
                //   prior carve-out preferred) is more painful in
                //   production than burning 4× latency on the
                //   genuinely-corrupt-decode case.
                //
                // What stays terminal: `is_builder()` (startup config
                // error), `is_decode()` for JSON/wire-format decode
                // failures (we surface those as `Serialization`
                // anyway), and anything else not in the above set.
                e.is_connect() || e.is_timeout() || e.is_request() || e.is_body()
            }
            Error::RateLimit { .. } => true,
            Error::Provider { retryable, .. } => *retryable,
            Error::Auth { .. }
            | Error::Serialization(_)
            | Error::Config(_)
            | Error::InvalidPrompt(_)
            | Error::ModelNotAvailable(_)
            | Error::ContextWindowExceeded { .. }
            | Error::UnsupportedInput { .. }
            | Error::Compaction { .. } => false,
        }
    }

    /// Provider-suggested wait duration before retrying, parsed from a
    /// `Retry-After` header (or equivalent).
    ///
    /// Returns `Some(d)` for [`Self::RateLimit`] *and* for
    /// [`Self::Provider`] when the upstream supplied a hint — RFC
    /// 7231 defines `Retry-After` on 503 as well as 429, and several
    /// providers send it on transient 5xx. `None` for every other
    /// variant *and* for retryable errors with no header. Callers
    /// that get `None` from a retryable error should fall back to
    /// their own backoff policy.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Error::RateLimit { retry_after, .. } => *retry_after,
            Error::Provider { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

/// Status fragment for the `Provider` Display. Returns only the
/// `, status NNN` part (or empty) — the surrounding `(…)`: and
/// message live in the format string itself, so editing this helper
/// can't silently unbalance the parens.
fn status_suffix(status: Option<u16>) -> String {
    match status {
        Some(s) => format!(", status {s}"),
        None => String::new(),
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
    fn provider_display_is_exactly_balanced_with_and_without_status() {
        // Pins the exact rendered string so a future edit to
        // `status_suffix` can't silently unbalance the parens.
        assert_eq!(
            Error::provider("OpenAI", "boom").to_string(),
            "provider error (OpenAI): boom",
        );
        assert_eq!(
            Error::provider_with_status("Google", 503, "down").to_string(),
            "provider error (Google, status 503): down",
        );
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
    fn invalid_prompt_constructor_renders_prefix_and_message() {
        let err = Error::invalid_prompt("only system items");
        assert!(matches!(err, Error::InvalidPrompt(_)));
        let msg = err.to_string();
        assert!(msg.contains("invalid prompt"));
        assert!(msg.contains("only system items"));
    }

    #[test]
    fn config_constructor_renders_invalid_configuration_prefix() {
        let err = Error::config("Invalid model name");
        assert!(err.to_string().contains("invalid configuration"));
        assert!(err.to_string().contains("Invalid model name"));
    }

    #[test]
    fn is_retryable_covers_transient_variants() {
        assert!(Error::rate_limit(Some(5), "slow down").is_retryable());
        assert!(Error::rate_limit(None, "slow down").is_retryable());
        assert!(Error::provider_with_status("OpenAI", 503, "down").is_retryable());
        assert!(Error::provider_with_status("OpenAI", 429, "slow").is_retryable());
    }

    #[test]
    fn is_retryable_rejects_terminal_variants() {
        // Status-bearing 4xx that aren't 429 are explicitly non-retryable.
        assert!(!Error::provider_with_status("OpenAI", 400, "bad").is_retryable());
        assert!(!Error::provider("OpenAI", "default non-retryable").is_retryable());
        assert!(!Error::auth("bad key").is_retryable());
        assert!(!Error::auth_with_status(401, "bad key").is_retryable());
        assert!(!Error::config("nope").is_retryable());
        assert!(!Error::invalid_prompt("nope").is_retryable());
        assert!(!Error::ModelNotAvailable("gpt-x".into()).is_retryable());
        assert!(!Error::context_window_exceeded("OpenAI", "too long").is_retryable());
        assert!(!Error::compaction("empty memo").is_retryable());
    }

    #[test]
    fn retry_after_surfaces_rate_limit_hint() {
        let with_hint = Error::rate_limit(Some(42), "slow down");
        assert_eq!(with_hint.retry_after(), Some(Duration::from_secs(42)));

        let without_hint = Error::rate_limit(None, "slow down");
        assert_eq!(without_hint.retry_after(), None);
    }

    /// A 503 (or other retryable 5xx) that carries a `Retry-After`
    /// must surface the hint via `retry_after()` so the helper
    /// honours the server's instruction rather than falling back to
    /// blind exponential backoff. RFC 7231 explicitly allows
    /// `Retry-After` on 503, and OpenAI / Anthropic / Vertex all
    /// send it in practice.
    #[test]
    fn retry_after_surfaces_provider_5xx_hint() {
        let err = Error::provider_with_retry_after("OpenAI", 503, Some(30), "down");
        assert_eq!(err.retry_after(), Some(Duration::from_secs(30)));
        assert!(err.is_retryable());
        // Without a hint, retryability is preserved but no hint is reported.
        let no_hint = Error::provider_with_status("OpenAI", 503, "down");
        assert!(no_hint.is_retryable());
        assert_eq!(no_hint.retry_after(), None);
    }
}
