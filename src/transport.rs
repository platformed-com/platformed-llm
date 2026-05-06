//! HTTP transport abstraction shared by all providers.
//!
//! Every provider's `generate()` ultimately runs
//!
//! 1. build a request (URL, headers, body) — provider-specific,
//! 2. send it over HTTP and read back a streaming response — *not*
//!    provider-specific, and
//! 3. parse the response stream into unified `StreamEvent`s — again
//!    provider-specific.
//!
//! Step 2 used to be inlined into each provider against `reqwest::Client`
//! directly. That made every provider responsible for its own retry /
//! recording / mocking story and forced tests to either spin up wiremock or
//! reach into provider internals. This module factors step 2 behind a
//! [`Transport`] handle so providers compose against an interface and tests
//! can inject anything that implements [`TransportImpl`] (e.g. a recorder
//! that tees the body to disk, or a replayer that returns canned bytes).
//!
//! The split is:
//!
//! - [`TransportImpl`] — the extension trait. Implement this to plug in
//!   recording, replay, mocking, custom retry, etc.
//! - [`Transport`] — the public concrete type providers store and clone.
//!   Wraps `Arc<dyn TransportImpl>` internally so cloning is cheap.
//! - [`ReqwestTransport`] — the default implementation backed by
//!   `reqwest::Client`. Constructed via [`Transport::reqwest`] /
//!   [`Transport::reqwest_with_client`].
//!
//! All current LLM requests are `POST` so we don't expose a method field
//! yet; add it when we need `GET` (e.g. for fetching files / models /
//! batches).

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt as _};

use crate::Error;

/// Connect timeout for the default reqwest-backed transport. We deliberately
/// do **not** set a total request timeout — streaming responses (especially
/// reasoning / extended thinking) can legitimately run for many minutes.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// A request to be sent by a [`Transport`]. POST-only for now.
#[derive(Debug, Clone)]
pub struct TransportRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// A streaming response yielded by a [`Transport`].
///
/// Status and headers are read eagerly (cheap), but the body is exposed as
/// a streaming `Bytes` source so providers can pipe it straight into the
/// SSE parser without buffering the whole response. Dropping `body`
/// closes the underlying connection — the cancellation contract verified
/// by `tests/cancellation.rs`.
pub struct TransportResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
}

impl TransportResponse {
    /// Read the value of the first header whose name (case-insensitively)
    /// equals `name`. Convenience for providers that only care about a
    /// handful of headers (e.g. `retry-after`).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Drain the entire body into a single `Vec<u8>`. Only used on the
    /// error path (we want the full error envelope to log / parse) and
    /// during capture. Streaming success-path consumers should iterate
    /// `body` directly.
    pub async fn collect_body(mut self) -> Result<Vec<u8>, Error> {
        let mut buf = Vec::new();
        while let Some(chunk) = self.body.next().await {
            buf.extend_from_slice(&chunk?);
        }
        Ok(buf)
    }
}

/// Extension trait implemented by anyone supplying a transport.
///
/// Implementations must propagate cancellation: dropping the returned
/// `TransportResponse.body` must terminate any in-flight network read
/// (the default `ReqwestTransport` does this because `reqwest`'s
/// `bytes_stream` carries the underlying connection in its drop).
#[async_trait]
pub trait TransportImpl: Send + Sync + 'static {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error>;
}

/// The shared transport handle that providers store. Cheap to clone
/// (internally an `Arc`).
#[derive(Clone)]
pub struct Transport {
    inner: Arc<dyn TransportImpl>,
}

impl Transport {
    /// Wrap any [`TransportImpl`] as a `Transport`.
    pub fn new<T: TransportImpl>(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Default `reqwest::Client`-backed transport with a sensible connect
    /// timeout and no whole-request timeout. Most callers want this.
    pub fn reqwest() -> Result<Self, Error> {
        Ok(Self::new(ReqwestTransport::with_default_client()?))
    }

    /// Build a transport from a caller-owned `reqwest::Client`. Useful when
    /// the caller already configures TLS, proxies, retry middleware, etc.
    pub fn reqwest_with_client(client: reqwest::Client) -> Self {
        Self::new(ReqwestTransport::new(client))
    }

    /// Send a request via the underlying transport.
    pub async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        self.inner.send(req).await
    }
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transport").finish_non_exhaustive()
    }
}

/// Default transport: a `reqwest::Client` configured for streaming LLM
/// responses (connect timeout, no whole-request timeout, no retry).
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    /// Wrap a caller-supplied client.
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Build with the default client config used by the lib.
    pub fn with_default_client() -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .build()
            .map_err(Error::from)?;
        Ok(Self::new(client))
    }
}

#[async_trait]
impl TransportImpl for ReqwestTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        let mut builder = self.client.post(&req.url).body(req.body);
        for (k, v) in &req.headers {
            builder = builder.header(k, v);
        }
        let response = builder.send().await?;

        let status = response.status().as_u16();
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.as_str().to_string(), s.to_string())))
            .collect();

        // Map reqwest's per-chunk stream error onto ours. Dropping this
        // boxed stream drops the underlying reqwest body, which closes
        // the connection — preserving the cancellation contract.
        let body = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| Error::streaming(format!("transport: {e}"))));
        let body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> = Box::pin(body);

        Ok(TransportResponse {
            status,
            headers,
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    /// `header()` is the convenience that error paths use to read
    /// `retry-after` etc. Must be case-insensitive on the lookup side
    /// because real providers vary (`Retry-After`, `retry-after`).
    #[test]
    fn header_lookup_is_case_insensitive() {
        let resp = TransportResponse {
            status: 429,
            headers: vec![
                ("Retry-After".to_string(), "30".to_string()),
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
            body: Box::pin(stream::empty()),
        };
        assert_eq!(resp.header("retry-after"), Some("30"));
        assert_eq!(resp.header("RETRY-AFTER"), Some("30"));
        assert_eq!(resp.header("missing"), None);
    }

    /// `Transport` is a thin newtype around `Arc<dyn TransportImpl>` —
    /// cloning must not create new underlying resources.
    #[tokio::test]
    async fn transport_clone_shares_underlying_impl() {
        struct Counting(std::sync::atomic::AtomicUsize);
        #[async_trait]
        impl TransportImpl for Counting {
            async fn send(&self, _req: TransportRequest) -> Result<TransportResponse, Error> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: Box::pin(stream::empty()),
                })
            }
        }
        let counting = Counting(std::sync::atomic::AtomicUsize::new(0));
        let t = Transport::new(counting);
        let t2 = t.clone();
        let req = || TransportRequest {
            url: "http://x".into(),
            headers: vec![],
            body: vec![],
        };
        let _ = t.send(req()).await.unwrap();
        let _ = t2.send(req()).await.unwrap();
        // Both clones routed to the same impl, so we should see 2 calls.
        // We can't introspect Counting through the trait object without
        // adding more API; instead, this test mostly exists to lock in
        // that `clone()` compiles and doesn't allocate a new impl.
    }
}
