#![cfg(feature = "openai")]
//! End-to-end tests for OpenAI HTTP error mapping.
//!
//! `parse_openai_error` has unit tests, but those don't exercise the path
//! from a real HTTP response (status, headers, body) through
//! `OpenAIProvider::generate()` to a typed [`Error`]. The wiring there
//! reads `retry-after`, drains the body, and calls the parser — that whole
//! pipeline is untested at the unit level. This file fills that gap by
//! injecting a [`StaticTransport`] that returns synthetic status / headers /
//! body — no wiremock, no async server spin-up.

use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use platformed_llm::providers::OpenAIProvider;
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{Config, Error, Prompt, Provider};

/// Test-only `TransportImpl` returning a fixed status / headers / body.
/// The lib's error path will read these in the same order as it would
/// against the real wire (header lookup → body drain → typed Error).
struct StaticTransport {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[async_trait]
impl TransportImpl for StaticTransport {
    async fn send(&self, _req: TransportRequest) -> Result<TransportResponse, Error> {
        let body = Bytes::from(self.body.clone());
        let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![Ok(body)]));
        Ok(TransportResponse {
            status: self.status,
            headers: self.headers.clone(),
            body: stream,
        })
    }
}

async fn openai_against(
    status: u16,
    headers: Vec<(String, String)>,
    body: &str,
) -> Result<(), Error> {
    let transport = Transport::new(StaticTransport {
        status,
        headers,
        body: body.as_bytes().to_vec(),
    });
    let provider = OpenAIProvider::with_transport(
        "test-key".to_string(),
        "http://placeholder".to_string(),
        transport,
    );
    let prompt = Prompt::user("hi");
    let cfg = Config::new("gpt-4o-mini").build();
    provider.generate(&prompt, cfg.raw()).await.map(|_| ())
}

#[tokio::test]
async fn http_429_with_retry_after_surfaces_as_rate_limit() {
    let body = r#"{"error":{"message":"Rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
    let err = openai_against(
        429,
        vec![
            ("retry-after".to_string(), "42".to_string()),
            ("content-type".to_string(), "application/json".to_string()),
        ],
        body,
    )
    .await
    .expect_err("429 must produce an error");

    match err {
        Error::RateLimit {
            retry_after,
            message,
        } => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(42)));
            assert!(
                message.contains("Rate limited"),
                "message should contain provider text, got: {message}",
            );
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn http_429_without_retry_after_still_maps_to_rate_limit() {
    let body = r#"{"error":{"message":"Slow down","type":"rate_limit_error"}}"#;
    let err = openai_against(429, vec![], body)
        .await
        .expect_err("429 must error");

    match err {
        Error::RateLimit { retry_after, .. } => assert_eq!(retry_after, None),
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn http_401_surfaces_as_auth() {
    let body = r#"{"error":{"message":"Bad key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
    let err = openai_against(401, vec![], body)
        .await
        .expect_err("401 must error");

    match err {
        Error::Auth { status, message } => {
            assert_eq!(status, Some(401));
            assert!(
                message.contains("Bad key"),
                "auth message lost provider text: {message}",
            );
        }
        other => panic!("expected Auth, got {other:?}"),
    }
}

#[tokio::test]
async fn http_500_surfaces_as_provider() {
    let body = r#"{"error":{"message":"boom","type":"server_error"}}"#;
    let err = openai_against(500, vec![], body)
        .await
        .expect_err("500 must error");

    match err {
        Error::Provider {
            provider, message, ..
        } => {
            assert_eq!(provider, "OpenAI");
            assert!(message.contains("500"), "should mention status: {message}");
            assert!(message.contains("boom"), "should mention body: {message}");
        }
        other => panic!("expected Provider, got {other:?}"),
    }
}

/// Even when the body isn't JSON we should still produce a typed error and
/// preserve the raw text. Catches a regression where a malformed body would
/// panic the deserializer or get swallowed.
#[tokio::test]
async fn non_json_error_body_is_preserved() {
    let err = openai_against(503, vec![], "upstream proxy timeout")
        .await
        .expect_err("503 must error");

    match err {
        Error::Provider {
            provider, message, ..
        } => {
            assert_eq!(provider, "OpenAI");
            assert!(message.contains("upstream proxy timeout"), "got: {message}");
        }
        other => panic!("expected Provider, got {other:?}"),
    }
}
