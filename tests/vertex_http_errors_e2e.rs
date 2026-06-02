#![cfg(all(feature = "google", feature = "anthropic-vertex"))]
//! End-to-end tests for Vertex (Gemini + Anthropic-via-Vertex) HTTP
//! error mapping.
//!
//! Mirrors `http_errors_e2e.rs` (OpenAI): a [`StaticTransport`] returns
//! a synthetic status / headers / body and we assert the lib maps it to
//! the right typed [`Error`] through the real `generate()` path —
//! specifically that 429 / RESOURCE_EXHAUSTED becomes
//! [`Error::RateLimit`] (carrying `Retry-After`), not the generic
//! [`Error::Provider`] that backoff code would miss.

use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use platformed_llm::providers::{AnthropicViaVertexProvider, GoogleProvider, VertexEndpoint};
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{generate, Config, Error, Prompt};

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

fn endpoint() -> VertexEndpoint {
    VertexEndpoint::with_access_token(
        "proj".to_string(),
        "us-east1".to_string(),
        "tok".to_string(),
    )
    .with_base_url("http://placeholder")
}

fn transport(status: u16, headers: Vec<(String, String)>, body: &str) -> Transport {
    Transport::new(StaticTransport {
        status,
        headers,
        body: body.as_bytes().to_vec(),
    })
}

async fn google_err(status: u16, headers: Vec<(String, String)>, body: &str) -> Error {
    let provider = GoogleProvider::with_transport(endpoint(), transport(status, headers, body));
    let cfg = Config::builder("gemini-2.5-flash").build();
    generate(&provider, &Prompt::user("hi"), &cfg)
        .await
        .map(|_| ())
        .expect_err("non-2xx should error")
}

async fn anthropic_err(status: u16, headers: Vec<(String, String)>, body: &str) -> Error {
    let provider =
        AnthropicViaVertexProvider::with_transport(endpoint(), transport(status, headers, body));
    let cfg = Config::builder("claude-sonnet-4").build();
    generate(&provider, &Prompt::user("hi"), &cfg)
        .await
        .map(|_| ())
        .expect_err("non-2xx should error")
}

fn assert_rate_limited(err: Error, want_secs: Option<u64>) {
    match err {
        Error::RateLimit { retry_after, .. } => {
            assert_eq!(
                retry_after,
                want_secs.map(std::time::Duration::from_secs),
                "retry_after mismatch",
            );
        }
        other => panic!("expected Error::RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn google_429_with_retry_after_is_rate_limit() {
    let err = google_err(
        429,
        vec![("retry-after".to_string(), "42".to_string())],
        r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota"}}"#,
    )
    .await;
    assert_rate_limited(err, Some(42));
}

#[tokio::test]
async fn google_429_without_retry_after_still_rate_limit() {
    let err = google_err(429, vec![], r#"{"error":{"status":"RESOURCE_EXHAUSTED"}}"#).await;
    assert_rate_limited(err, None);
}

#[tokio::test]
async fn anthropic_429_with_retry_after_is_rate_limit() {
    let err = anthropic_err(
        429,
        vec![("Retry-After".to_string(), "7".to_string())],
        r#"{"type":"error","error":{"type":"rate_limit_error"}}"#,
    )
    .await;
    assert_rate_limited(err, Some(7));
}

#[tokio::test]
async fn google_500_is_generic_provider_error() {
    let err = google_err(500, vec![], "boom").await;
    assert!(
        matches!(err, Error::Provider { .. }),
        "500 should be a generic provider error, got {err:?}"
    );
}
