#![cfg(feature = "google")]
//! Offline check that the Gemini (Vertex) generate call carries the
//! `X-Vertex-AI-LLM-Request-Type` header the caller asked for.
//!
//! The header decides which capacity pool Vertex serves the request from, and
//! therefore which billing line it lands on. Vertex ignores an unrecognised
//! header name rather than rejecting the request, so a wrong or missing header
//! produces no error anywhere — only a billing discrepancy. These tests pin the
//! exact wire strings against a mock transport that records what was sent.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use platformed_llm::providers::{GoogleProvider, VertexEndpoint};
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{generate, Config, Error, InputItem, Prompt, UserPart, VertexRequestType};

/// Headers captured from the one request the provider issues. `None` until the
/// transport is called, which distinguishes "no header sent" from "no request
/// sent at all".
type CapturedHeaders = Arc<Mutex<Option<Vec<(String, String)>>>>;

/// Records the headers of the request it is handed, then fails the call with a
/// terminal 400 so the provider returns without needing a parseable response
/// body. The assertions only care about what went out.
struct RecordingTransport {
    headers: CapturedHeaders,
}

#[async_trait]
impl TransportImpl for RecordingTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        *self.headers.lock().unwrap() = Some(req.headers.clone());

        let body =
            br#"{"error":{"code":400,"message":"stop here","status":"INVALID_ARGUMENT"}}"#.to_vec();
        let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(body))]));
        Ok(TransportResponse {
            status: 400,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: stream,
        })
    }
}

/// Drive one generate call through a `GoogleProvider` configured with the given
/// request type and hand back the headers it put on the wire.
async fn headers_for(request_type: Option<VertexRequestType>) -> Vec<(String, String)> {
    let captured = Arc::new(Mutex::new(None));
    let transport = Transport::new(RecordingTransport {
        headers: captured.clone(),
    });

    let endpoint = VertexEndpoint::with_access_token(
        "proj-1".to_string(),
        "us-east1".to_string(),
        "tok".to_string(),
    );
    let mut provider = GoogleProvider::with_transport(endpoint, transport);
    if let Some(request_type) = request_type {
        provider = provider.with_request_type(request_type);
    }

    let prompt = Prompt::new().with_item(InputItem::User {
        content: vec![UserPart::Text("Hello.".to_string())],
    });
    let cfg = Config::builder("gemini-2.5-flash").max_tokens(16).build();

    // The 400 from `RecordingTransport` makes this fail; the request has
    // already been recorded by then.
    let _ = generate(&provider, &prompt, &cfg).await;

    let headers = captured.lock().unwrap().take();
    headers.expect("transport was never called")
}

/// Look up a header by name, matching case-insensitively as HTTP does.
fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Hardcoded to keep this assertion independent of the constants the provider
/// renders from — reading them back would pass even if they were wrong.
#[tokio::test]
async fn shared_request_type_sends_shared_header() {
    let headers = headers_for(Some(VertexRequestType::Shared)).await;
    assert_eq!(
        header(&headers, "X-Vertex-AI-LLM-Request-Type"),
        Some("shared"),
        "got: {headers:?}"
    );
}

#[tokio::test]
async fn dedicated_request_type_sends_dedicated_header() {
    let headers = headers_for(Some(VertexRequestType::Dedicated)).await;
    assert_eq!(
        header(&headers, "X-Vertex-AI-LLM-Request-Type"),
        Some("dedicated"),
        "got: {headers:?}"
    );
}

/// An absent header is a third Vertex behaviour: provisioned throughput with
/// spillover to on-demand. Sending `shared` in its place would silently move
/// every request off the provisioned-throughput commitment.
#[tokio::test]
async fn no_request_type_omits_the_header() {
    let headers = headers_for(None).await;
    assert_eq!(
        header(&headers, "X-Vertex-AI-LLM-Request-Type"),
        None,
        "got: {headers:?}"
    );
}
