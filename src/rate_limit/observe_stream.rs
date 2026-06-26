//! Stream wrapper that defers [`RatePermit::observe`] until the
//! response stream terminates.
//!
//! Each hosted provider's `generate()` acquires a permit before the
//! HTTP send. The HTTP-status-time outcome is unambiguous: 429s are
//! `RateLimited`, transport errors are `OtherFailure`, non-429 4xx /
//! 5xx are `OtherFailure`. But success at the HTTP layer doesn't
//! imply success at the application layer — Anthropic in
//! particular routinely returns a 200 OK and then emits an
//! `overloaded_error` or `rate_limit_error` SSE event mid-response,
//! and OpenAI / Vertex can drop the connection mid-stream. Observing
//! `Success` at HTTP-200 time would tell the AIMD model "things are
//! fine" on a request that was in fact rate-limited.
//!
//! [`observe_response_stream`] wraps the success-path SSE stream and
//! defers the [`RatePermit::observe`] call to the terminal event:
//!
//! - Terminal `StreamEvent::Done` → observe [`RateOutcome::Success`].
//! - Stream item `Err(Error::RateLimit { … })` → observe
//!   [`RateOutcome::RateLimited`] with the typed error's
//!   `retry_after` (or none).
//! - Stream item `Err(_)` for any other error → observe
//!   [`RateOutcome::OtherFailure`].
//! - Stream dropped without ever yielding a terminal event → the
//!   wrapper's `Drop` fires `Cancelled` via the permit's own Drop.
//!
//! Pre-stream errors (transport failure, non-2xx HTTP) are still
//! observed at the call site before the wrapper is even constructed;
//! that path doesn't need the wrapper.
//!
//! # `info` field & per-provider asymmetry
//!
//! The `info: ProviderRateInfo` the wrapper holds is captured at
//! HTTP-200 time from response headers. Only **OpenAI** currently
//! populates it (`x-ratelimit-{remaining,reset}-*`); Anthropic-via-
//! Vertex and Vertex-Gemini pass `ProviderRateInfo::default()`
//! because Vertex doesn't expose comparable headers on its REST
//! responses. The practical consequence inside
//! [`super::InMemoryRateLimiter::observe_success`]: the
//! observed-capacity ceiling — the cap that prevents AIMD from
//! growing past what the provider's `remaining` budget suggests —
//! only ever fires for OpenAI. Other providers grow via plain
//! `additive_step` until a 429 multiplicatively halves them. That's
//! the right behaviour given the headers we have (no signal → no
//! ceiling), but it does mean the AIMD model is *less* informed for
//! non-OpenAI providers.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::stream::Stream;

use super::{ProviderRateInfo, RateOutcome, RatePermit};
use crate::types::StreamEvent;
use crate::Error;

/// Wrap a success-path SSE stream so the rate-limit permit observes
/// the terminal event rather than the HTTP-200 status. See the
/// module docs for the rationale.
///
/// `dead_code` allowed for the no-provider-feature build: every
/// hosted provider (`openai`, `google`, `anthropic-vertex`,
/// `mock`) calls this, but a `--no-default-features` build has
/// none of them enabled and clippy complains.
#[allow(dead_code)]
pub(crate) fn observe_response_stream<S>(
    inner: S,
    permit: RatePermit,
    info: ProviderRateInfo,
) -> ObservingStream<S>
where
    S: Stream<Item = Result<StreamEvent, Error>>,
{
    ObservingStream {
        inner,
        permit: Some(permit),
        info,
    }
}

pin_project_lite::pin_project! {
    /// Stream adapter that holds a [`RatePermit`] across the
    /// response stream and observes it on the terminal event /
    /// Drop. See [`observe_response_stream`].
    pub(crate) struct ObservingStream<S> {
        #[pin]
        inner: S,
        // `Option` so we can `.take()` on terminal events; the
        // `Drop` impl uses the remaining `Some` to fire
        // `Cancelled` if the stream was dropped early.
        permit: Option<RatePermit>,
        info: ProviderRateInfo,
    }

    impl<S> PinnedDrop for ObservingStream<S> {
        fn drop(this: Pin<&mut Self>) {
            // Dropped before terminal event — the permit's own
            // Drop fires Cancelled when we let it fall out of
            // scope here.
            let _ = this.project().permit.take();
        }
    }
}

impl<S> Stream for ObservingStream<S>
where
    S: Stream<Item = Result<StreamEvent, Error>>,
{
    type Item = Result<StreamEvent, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let polled = this.inner.poll_next(cx);
        match &polled {
            Poll::Ready(Some(Ok(event))) => {
                // `Done` is the canonical terminal success event.
                // Some providers emit further keepalive frames
                // after Done (we don't), but the canonical
                // consume-pattern stops here.
                if matches!(event, StreamEvent::Done { .. }) {
                    if let Some(permit) = this.permit.take() {
                        permit.observe(RateOutcome::Success {
                            info: this.info.clone(),
                        });
                    }
                }
            }
            Poll::Ready(Some(Err(e))) => {
                if let Some(permit) = this.permit.take() {
                    let outcome = match e {
                        Error::RateLimit { retry_after, .. } => RateOutcome::RateLimited {
                            retry_after: *retry_after,
                            info: this.info.clone(),
                        },
                        _ => RateOutcome::OtherFailure,
                    };
                    permit.observe(outcome);
                }
            }
            // Stream exhausted with no Done — could be a graceful
            // close after Done already fired (handled above) or a
            // mid-stream close. Leave the permit to Drop with
            // Cancelled if it's still present.
            Poll::Ready(None) => {}
            Poll::Pending => {}
        }
        polled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FinishReason, Usage};
    use futures::StreamExt;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// Helper: an observed-outcome counter the closure records into.
    fn permit_counter() -> (
        RatePermit,
        Arc<AtomicU32>,
        Arc<std::sync::Mutex<Vec<&'static str>>>,
    ) {
        let count = Arc::new(AtomicU32::new(0));
        let kinds: Arc<std::sync::Mutex<Vec<&'static str>>> = Arc::new(Default::default());
        let count_cb = count.clone();
        let kinds_cb = kinds.clone();
        let permit = RatePermit::new(move |outcome| {
            count_cb.fetch_add(1, Ordering::SeqCst);
            kinds_cb.lock().unwrap().push(match outcome {
                RateOutcome::Success { .. } => "success",
                RateOutcome::RateLimited { .. } => "rate-limited",
                RateOutcome::OtherFailure => "other-failure",
                RateOutcome::Cancelled => "cancelled",
            });
        });
        (permit, count, kinds)
    }

    #[tokio::test]
    async fn terminal_done_observes_success() {
        let (permit, count, kinds) = permit_counter();
        let events = vec![Ok(StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        })];
        let stream = futures::stream::iter(events);
        let mut wrapped = observe_response_stream(stream, permit, ProviderRateInfo::default());
        while wrapped.next().await.is_some() {}
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(kinds.lock().unwrap().as_slice(), &["success"]);
    }

    #[tokio::test]
    async fn mid_stream_rate_limit_error_observes_rate_limited() {
        let (permit, count, kinds) = permit_counter();
        let events: Vec<Result<StreamEvent, Error>> =
            vec![Err(Error::rate_limit(Some(5), "synthetic mid-stream 429"))];
        let stream = futures::stream::iter(events);
        let mut wrapped = observe_response_stream(stream, permit, ProviderRateInfo::default());
        while wrapped.next().await.is_some() {}
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(kinds.lock().unwrap().as_slice(), &["rate-limited"]);
    }

    #[tokio::test]
    async fn mid_stream_other_error_observes_other_failure() {
        let (permit, count, kinds) = permit_counter();
        let events: Vec<Result<StreamEvent, Error>> =
            vec![Err(Error::provider("Stream", "connection reset"))];
        let stream = futures::stream::iter(events);
        let mut wrapped = observe_response_stream(stream, permit, ProviderRateInfo::default());
        while wrapped.next().await.is_some() {}
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(kinds.lock().unwrap().as_slice(), &["other-failure"]);
    }

    /// Dropping the wrapped stream before a terminal event fires the
    /// permit's `Cancelled` outcome (via the permit's own `Drop`).
    /// This is the cancellation-safety guarantee — the caller can
    /// abort a future without leaking the limiter slot.
    #[tokio::test]
    async fn early_drop_observes_cancelled() {
        let (permit, count, kinds) = permit_counter();
        let events: Vec<Result<StreamEvent, Error>> = vec![]; // empty
        let stream = futures::stream::iter(events);
        let wrapped = observe_response_stream(stream, permit, ProviderRateInfo::default());
        drop(wrapped);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(kinds.lock().unwrap().as_slice(), &["cancelled"]);
    }
}
