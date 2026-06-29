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
//! [`crate::transport::Transport`] handle so providers compose against an
//! interface and tests can inject anything that implements
//! [`crate::transport::TransportImpl`] (e.g. a recorder that tees the body
//! to disk, or a replayer that returns canned bytes).
//!
//! The split is:
//!
//! - [`crate::transport::TransportImpl`] — the extension trait. Implement
//!   this to plug in recording, replay, mocking, custom retry, etc.
//! - [`crate::transport::Transport`] — the public concrete type providers
//!   store and clone. Wraps `Arc<dyn TransportImpl>` internally so cloning
//!   is cheap.
//! - [`crate::transport::ReqwestTransport`] — the default implementation
//!   backed by `reqwest::Client`. Constructed via
//!   [`crate::transport::Transport::reqwest`] /
//!   [`crate::transport::Transport::reqwest_with_client`].
//!
//! All current LLM requests are `POST` so we don't expose a method field
//! yet; add it when we need `GET` (e.g. for fetching files / models /
//! batches).

use std::pin::Pin;
use std::sync::Arc;
#[cfg(feature = "reqwest")]
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt as _};

use crate::Error;

/// Connect timeout for the default reqwest-backed transport. We
/// deliberately do **not** set a total request timeout — streaming
/// responses (especially reasoning / extended thinking) can
/// legitimately run for many minutes.
#[cfg(feature = "reqwest")]
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// A request to be sent by a [`Transport`]. POST-only for now.
#[derive(Debug, Clone)]
pub struct TransportRequest {
    /// Full request URL.
    pub url: String,
    /// Request headers (case preserved as supplied).
    pub headers: Vec<(String, String)>,
    /// Raw request body bytes.
    pub body: Vec<u8>,
}

/// HTTP method for a streaming [`UploadRequest`]. File-upload endpoints use
/// `POST` (multipart create) or `PUT` (resumable-session data).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// HTTP `POST`.
    Post,
    /// HTTP `PUT`.
    Put,
}

/// A streaming-body request used for **file uploads** — the one place the
/// library must not buffer the whole payload in memory.
///
/// Kept separate from [`TransportRequest`] (which stays buffered + `Clone`
/// for the `generate` path) so the streaming body — which is single-use and
/// neither `Clone` nor `Debug` — never infects that path. The library never
/// replays an upload body; on retry it re-opens a fresh stream via
/// [`FileResolver::open`](crate::FileResolver::open).
pub struct UploadRequest {
    /// HTTP method.
    pub method: Method,
    /// Full request URL.
    pub url: String,
    /// Request headers (case preserved as supplied).
    pub headers: Vec<(String, String)>,
    /// Total body length when known. Enables `Content-Length`; `None` falls
    /// back to chunked transfer-encoding.
    pub content_length: Option<u64>,
    /// Streaming request body. Mirrors [`TransportResponse::body`]; dropping
    /// it mid-flight must terminate the upload cleanly.
    pub body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
}

/// A streaming response yielded by a [`Transport`].
///
/// Status and headers are read eagerly (cheap), but the body is exposed as
/// a streaming `Bytes` source so providers can pipe it straight into the
/// SSE parser without buffering the whole response. Dropping `body`
/// closes the underlying connection — the cancellation contract verified
/// by `tests/cancellation.rs`.
pub struct TransportResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response headers, in the order the server sent them.
    pub headers: Vec<(String, String)>,
    /// Streaming response body. Dropping closes the underlying connection.
    pub body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
}

/// Parse a `Retry-After` header value into whole seconds.
///
/// Handles **both** forms RFC 7231 defines for the header:
///
/// - Delta-seconds (`"30"`): direct parse.
/// - HTTP-date (`"Wed, 21 Oct 2026 07:28:00 GMT"`): converted to
///   delta-seconds against the current system clock. If the date is
///   in the past (clock skew) we floor to 0 — "retry now" — rather
///   than returning `None`. If the date is malformed, `None`.
///
/// OpenAI is documented as sending the delta-seconds form, but
/// real-world responses (especially from edge proxies / WAFs in
/// front of the providers) occasionally use the HTTP-date form;
/// silently ignoring those would defeat the whole point of the
/// rate-limit hint.
///
/// Only the hosted providers consume this; gated to those features so
/// a `--no-default-features` (core-only) build doesn't flag it as dead.
#[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
pub(crate) fn parse_retry_after(value: Option<&str>) -> Option<u64> {
    let raw = value?.trim();
    if let Ok(seconds) = raw.parse::<u64>() {
        return Some(seconds);
    }
    // HTTP-date form. RFC 7231 §7.1.1.1 specifies RFC 5322 (IMF-fixdate),
    // RFC 850, and asctime() forms; in practice the IMF-fixdate form is
    // overwhelmingly what shows up. Parse it without pulling in
    // `chrono` — the cost of a tiny stdlib parser is much less than a
    // multi-MB time crate.
    parse_imf_fixdate_offset_seconds(raw)
}

/// Parse an IMF-fixdate / RFC 5322 date (`"Wed, 21 Oct 2026 07:28:00 GMT"`)
/// and return the number of seconds between *now* and that instant.
/// Returns `Some(0)` for past dates (clock skew → retry immediately),
/// `None` for malformed input.
///
/// Only the IMF-fixdate form is supported — RFC 850 and asctime()
/// forms predate the modern HTTP spec and don't appear in any
/// provider response we've seen. If one shows up, callers fall back
/// to their own backoff.
#[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
fn parse_imf_fixdate_offset_seconds(s: &str) -> Option<u64> {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Expected shape: "Day, DD Mon YYYY HH:MM:SS GMT"
    // Strip the optional weekday prefix (`"Wed, "`) — we don't validate it.
    let rest = s.split_once(", ").map(|(_, r)| r).unwrap_or(s);
    let mut parts = rest.split_whitespace();
    let day: u32 = parts.next()?.parse().ok()?;
    let month = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i32 = parts.next()?.parse().ok()?;
    let time = parts.next()?;
    let mut t = time.split(':');
    let hour: u32 = t.next()?.parse().ok()?;
    let minute: u32 = t.next()?.parse().ok()?;
    let second: u32 = t.next()?.parse().ok()?;
    // We don't validate the trailing "GMT" — IMF-fixdate is always
    // GMT per spec; if it's missing we still treat the date as GMT.

    // Civil-date → Unix-epoch seconds via Howard Hinnant's algorithm
    // (https://howardhinnant.github.io/date_algorithms.html#days_from_civil).
    // No leap-second handling — same as every other HTTP date parser.
    let y = year - i32::from(month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let m = month as i32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day as i32 - 1;
    let doy = doy as u32; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era as i64 * 146097 + doe as i64 - 719_468;
    let target_secs = days * 86400 + (hour as i64) * 3600 + (minute as i64) * 60 + second as i64;

    let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    if target_secs <= now_secs {
        // Past or now — retry immediately.
        Some(0)
    } else {
        Some((target_secs - now_secs) as u64)
    }
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
    /// Issue an HTTP request and return a streaming response. Implementations
    /// must propagate cancellation through the returned body stream.
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error>;

    /// Issue a streaming-body request (file upload) and return the response.
    ///
    /// Default implementation errors — only transports that genuinely talk
    /// HTTP (e.g. [`ReqwestTransport`]) need to support uploads; mocks and
    /// replayers can ignore it. Implementations must stream `req.body`
    /// without buffering it whole and propagate cancellation.
    async fn send_upload(&self, req: UploadRequest) -> Result<TransportResponse, Error> {
        // Drop the body stream (closing any underlying source) and report
        // that this transport can't upload.
        drop(req);
        Err(Error::config(
            "this transport does not support file uploads (send_upload)",
        ))
    }
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
    ///
    /// **Footgun:** there is no overall or idle-read timeout (a
    /// streaming response has no fixed duration). A server that
    /// accepts the connection then stalls will hang `generate()`
    /// indefinitely. Wrap calls in [`tokio::time::timeout`], or
    /// supply a custom client via [`Self::reqwest_with_client`] /
    /// [`Self::new`] with your own idle timeout.
    ///
    /// Available when any hosted-provider feature
    /// (`openai` / `google` / `anthropic-vertex`) is enabled.
    #[cfg(feature = "reqwest")]
    pub fn reqwest() -> Result<Self, Error> {
        Ok(Self::new(ReqwestTransport::with_default_client()?))
    }

    /// Build a transport from a caller-owned `reqwest::Client`. Useful when
    /// the caller already configures TLS, proxies, retry middleware, etc.
    #[cfg(feature = "reqwest")]
    pub fn reqwest_with_client(client: reqwest::Client) -> Self {
        Self::new(ReqwestTransport::new(client))
    }

    /// Send a request via the underlying transport.
    pub async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        self.inner.send(req).await
    }

    /// Send a streaming-body upload request via the underlying transport.
    pub async fn send_upload(&self, req: UploadRequest) -> Result<TransportResponse, Error> {
        self.inner.send_upload(req).await
    }
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transport").finish_non_exhaustive()
    }
}

/// Default transport: a `reqwest::Client` configured for streaming LLM
/// responses (connect timeout, no whole-request timeout, no retry).
///
/// Available when the `reqwest` feature is enabled — implicitly the
/// case under any hosted-provider feature (`openai`, `google`,
/// `anthropic-vertex`).
#[cfg(feature = "reqwest")]
pub struct ReqwestTransport {
    client: reqwest::Client,
}

#[cfg(feature = "reqwest")]
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

#[cfg(feature = "reqwest")]
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
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|s| (k.as_str().to_string(), s.to_string()))
            })
            .collect();

        // Map reqwest's per-chunk stream error onto ours. Dropping this
        // boxed stream drops the underlying reqwest body, which closes
        // the connection — preserving the cancellation contract.
        let body = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(Error::from));
        let body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> = Box::pin(body);

        Ok(TransportResponse {
            status,
            headers,
            body,
        })
    }

    async fn send_upload(&self, req: UploadRequest) -> Result<TransportResponse, Error> {
        let mut builder = match req.method {
            Method::Post => self.client.post(&req.url),
            Method::Put => self.client.put(&req.url),
        };
        for (k, v) in &req.headers {
            builder = builder.header(k, v);
        }
        if let Some(len) = req.content_length {
            builder = builder.header("content-length", len);
        }
        // wrap_stream streams the body to the wire without buffering it whole;
        // dropping the response (and thus this request future) cancels it.
        builder = builder.body(reqwest::Body::wrap_stream(req.body));

        let response = builder.send().await?;

        let status = response.status().as_u16();
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|s| (k.as_str().to_string(), s.to_string()))
            })
            .collect();
        let body = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(Error::from));
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

    #[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
    #[test]
    fn parse_retry_after_handles_delta_seconds_and_garbage() {
        assert_eq!(parse_retry_after(Some("30")), Some(30));
        assert_eq!(parse_retry_after(Some("  7 ")), Some(7));
        assert_eq!(parse_retry_after(None), None);
        assert_eq!(parse_retry_after(Some("")), None);
        assert_eq!(parse_retry_after(Some("not a number")), None);
    }

    /// HTTP-date form: must convert to delta-seconds against the
    /// current clock. A past date floors to 0 (retry now); a future
    /// date returns a positive delta.
    #[cfg(any(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
    #[test]
    fn parse_retry_after_handles_http_date_form() {
        // A date deep in the past must floor to 0 ("retry now") rather
        // than returning None — clock skew shouldn't defeat the hint.
        assert_eq!(
            parse_retry_after(Some("Wed, 21 Oct 1990 07:28:00 GMT")),
            Some(0),
        );
        // A date deep in the future must return a positive delta.
        let far_future = parse_retry_after(Some("Sun, 21 Oct 2099 07:28:00 GMT"))
            .expect("future date must parse");
        assert!(
            far_future > 60 * 60 * 24 * 365,
            "year-2099 should be > 1 year out, got {far_future}s",
        );
        // Malformed must return None.
        assert_eq!(
            parse_retry_after(Some("Wed, 99 Foo 9999 99:99:99 GMT")),
            None
        );
        // Mixed gibberish must return None.
        assert_eq!(parse_retry_after(Some("Wed, 21")), None);
    }

    /// `Transport` is a thin newtype around `Arc<dyn TransportImpl>` —
    /// cloning must not create new underlying resources.
    #[tokio::test]
    async fn transport_clone_shares_underlying_impl() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // The counter lives in a shared Arc the test also holds, so
        // we can actually assert both clones routed to the *same*
        // impl instance (a count of 2), not merely that clone()
        // compiles.
        struct Counting(Arc<AtomicUsize>);
        #[async_trait]
        impl TransportImpl for Counting {
            async fn send(&self, _req: TransportRequest) -> Result<TransportResponse, Error> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: Box::pin(stream::empty()),
                })
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let t = Transport::new(Counting(calls.clone()));
        let t2 = t.clone();
        let req = || TransportRequest {
            url: "http://x".into(),
            headers: vec![],
            body: vec![],
        };
        let _ = t.send(req()).await.unwrap();
        let _ = t2.send(req()).await.unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "both clones must route to the same underlying impl",
        );
    }
}
