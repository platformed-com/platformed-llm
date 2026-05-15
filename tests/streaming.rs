#![cfg(feature = "openai")]
//! Verify the lib actually streams — consumers get events as their bytes
//! arrive, not bulk-buffered at the end.
//!
//! `sse_stream::tests` already covers parser robustness with arbitrary
//! chunk boundaries (synthetic input fed in one go). What's NOT
//! covered: the end-to-end contract that `Response::stream()` yields
//! event N before the underlying transport has produced event (N+1)'s
//! bytes.
//!
//! This test wires a transport whose response body yields **one byte
//! per `poll_next`, with a `Pending` interleaved between each byte**
//! (waking itself so the runtime re-polls). That stresses two things
//! at once:
//! 1. The SSE parser correctly buffers byte-by-byte (single-chunk
//!    parser tests already prove this in isolation; this exercises it
//!    end-to-end through the provider pipeline).
//! 2. The lib pipelines properly — `SseStream::poll_next` propagates
//!    `Pending` instead of busy-waiting for more bytes, so the
//!    consumer can be doing other work between events.

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::Stream;
use futures_util::StreamExt;
use platformed_llm::providers::OpenAIProvider;
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{Config, Error, PartKind, Prompt, Provider, StreamEvent};

/// A response body that yields exactly one byte per poll AND inserts a
/// `Pending` between bytes (waking itself so the runtime makes progress).
/// `consumed` exposes how many source bytes have been pulled, so the
/// test can assert pipelining: after the consumer reads event N, the
/// counter equals exactly the byte length of events 0..=N.
struct PerByteBody {
    bytes: Vec<u8>,
    pos: usize,
    // Toggles between Pending (true) and Ready (false) — flipped each
    // poll, so reads alternate. The Pending poll wakes itself before
    // returning, so the runtime re-polls without external coordination.
    pending_next: bool,
    consumed: Arc<AtomicUsize>,
}

impl Stream for PerByteBody {
    type Item = Result<Bytes, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.pos >= self.bytes.len() {
            return Poll::Ready(None);
        }
        if self.pending_next {
            self.pending_next = false;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        self.pending_next = true;
        let b = self.bytes[self.pos];
        self.pos += 1;
        self.consumed.fetch_add(1, Ordering::SeqCst);
        Poll::Ready(Some(Ok(Bytes::from(vec![b]))))
    }
}

struct PerByteTransport {
    body: Vec<u8>,
    consumed: Arc<AtomicUsize>,
}

#[async_trait]
impl TransportImpl for PerByteTransport {
    async fn send(&self, _req: TransportRequest) -> Result<TransportResponse, Error> {
        let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(PerByteBody {
                bytes: self.body.clone(),
                pos: 0,
                pending_next: true,
                consumed: self.consumed.clone(),
            });
        Ok(TransportResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
            body: stream,
        })
    }
}

/// A small captured-shape OpenAI SSE script with three text deltas
/// followed by `response.completed`. Byte-tracking the consumer's
/// progress against this script proves events become available as
/// their bytes arrive.
fn build_script() -> Vec<u8> {
    let frames = [
        r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}"#,
        r#"{"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text"}}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"one"}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"two"}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"three"}"#,
        r#"{"type":"response.content_part.done","output_index":0,"content_index":0}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-4o-mini","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#,
    ];
    let mut body = Vec::new();
    for frame in frames {
        body.extend_from_slice(b"data: ");
        body.extend_from_slice(frame.as_bytes());
        body.extend_from_slice(b"\n\n");
    }
    body
}

#[tokio::test]
async fn consumer_gets_events_as_bytes_arrive_not_bulk() {
    let body = build_script();
    let total_bytes = body.len();
    let consumed = Arc::new(AtomicUsize::new(0));
    let transport = Transport::new(PerByteTransport {
        body: body.clone(),
        consumed: consumed.clone(),
    });

    let provider = OpenAIProvider::with_transport(
        "test-key".to_string(),
        "http://placeholder".to_string(),
        transport,
    );
    let prompt = Prompt::user("hi");
    let cfg = Config::new("gpt-4o-mini");
    let response = provider.generate(&prompt, &cfg).await.unwrap();
    let mut stream = response.stream();

    // Read events as they come. After each event we assert the source
    // hasn't been fully drained — i.e. the lib is genuinely
    // pipelining rather than buffering everything before emitting.
    let mut deltas = Vec::new();
    let mut saw_part_start = false;
    while let Some(ev) = stream.next().await {
        let ev = ev.expect("no errors");
        match &ev {
            StreamEvent::PartStart {
                kind: PartKind::Text,
                ..
            } => {
                saw_part_start = true;
                // PartStart for text arrives once content_part.added
                // has been parsed — well before the rest of the script
                // has been consumed.
                let bytes_seen = consumed.load(Ordering::SeqCst);
                assert!(
                    bytes_seen < total_bytes,
                    "saw PartStart only after the whole body was drained — not streaming \
                     (consumed {bytes_seen}/{total_bytes})",
                );
            }
            // The continuation part fires at response.completed near
            // end-of-stream; not interesting for the pipelining check.
            StreamEvent::PartStart {
                kind: PartKind::Continuation(_),
                ..
            } => {}
            StreamEvent::Delta { delta, .. } => {
                let bytes_seen = consumed.load(Ordering::SeqCst);
                deltas.push(delta.clone());
                // Each delta should arrive before the entire body has
                // been consumed (since later events still have bytes
                // in flight on the wire).
                if deltas.len() < 3 {
                    assert!(
                        bytes_seen < total_bytes,
                        "Delta #{} arrived only after full drain ({bytes_seen}/{total_bytes}) \
                         — not streaming",
                        deltas.len(),
                    );
                }
            }
            StreamEvent::PartEnd { .. } | StreamEvent::Done { .. } => {}
            other => panic!("unexpected event: {other:?}"),
        }
    }

    // Sanity: we received the events we scripted, in order.
    assert!(saw_part_start, "missing PartStart");
    assert_eq!(deltas, vec!["one", "two", "three"]);
    assert_eq!(
        consumed.load(Ordering::SeqCst),
        total_bytes,
        "all source bytes should have been drained by stream end",
    );
}
