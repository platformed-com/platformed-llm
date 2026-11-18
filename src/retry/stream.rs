//! Streaming a response across retried attempts.

use std::future::Future;
use std::pin::Pin;
use std::task::{ready, Context, Poll};
use std::time::Duration;

use futures_util::future::BoxFuture;
use futures_util::stream::{BoxStream, Stream};

use super::RetryPolicy;
use crate::accumulator::ResponseAccumulator;
use crate::{CompleteResponse, Config, Error, Prompt, Provider, Response, StreamEvent};

/// An item of a [`RetryingResponse`].
#[derive(Debug)]
pub enum RetryEvent {
    /// An event from the current attempt.
    Event(StreamEvent),
    /// The current attempt failed and another will follow after
    /// [`RetryAttempt::delay`]. Everything taken from the failed attempt
    /// is superseded by what the next one streams.
    Reset(RetryAttempt),
    /// The final item of a successful stream: the response the last
    /// attempt's events accumulate to.
    Complete(CompleteResponse),
}

/// A failed attempt that is about to be retried.
#[derive(Debug)]
#[non_exhaustive]
pub struct RetryAttempt {
    /// The 1-indexed attempt that failed.
    pub attempt: u32,
    /// [`RetryPolicy::max_attempts`] of the policy in force.
    pub max_attempts: u32,
    /// How long the stream waits before the next attempt.
    pub delay: Duration,
    /// The error that failed the attempt.
    pub error: Error,
}

/// Generate a response with [`crate::generate`], retrying transient
/// failures under `policy`. See [`RetryingResponse`].
pub fn generate_with_retry<'a>(
    provider: &'a dyn Provider,
    prompt: &'a Prompt,
    config: &'a Config,
    policy: RetryPolicy,
) -> RetryingResponse<'a> {
    RetryingResponse::new(policy, move || crate::generate(provider, prompt, config))
}

type Attempt<'a> = Box<dyn FnMut() -> BoxFuture<'a, Result<Response, Error>> + Send + 'a>;

/// A stream of [`RetryEvent`]s spanning every attempt at one request.
///
/// Re-issues the request whenever an attempt fails
/// with an error its [`RetryPolicy`] retries, whether the failure came
/// before the first event or mid-stream. Each attempt's events are
/// passed through as [`RetryEvent::Event`] and fed to an internal
/// [`ResponseAccumulator`]. A failed attempt that will be retried ends
/// with [`RetryEvent::Reset`], which tells the consumer to discard
/// everything it took from that attempt; the accumulator discards its
/// copy at the same point. A successful attempt ends with its
/// [`StreamEvent::Done`] followed by [`RetryEvent::Complete`]. A failure
/// the policy does not retry ends the stream with that error.
///
/// An attempt ends at its `Done`: nothing after it is read. A stream
/// that runs out before `Done` fails the attempt with
/// [`Error::StreamTruncated`].
///
/// Dropping the stream cancels it, including a backoff in progress. A
/// consumer that stops reading on purpose recovers what the current
/// attempt has produced so far with [`RetryingResponse::into_partial`].
pub struct RetryingResponse<'a> {
    policy: RetryPolicy,
    attempt: u32,
    start_attempt: Attempt<'a>,
    accumulator: ResponseAccumulator,
    state: State<'a>,
}

enum State<'a> {
    Requesting(BoxFuture<'a, Result<Response, Error>>),
    Streaming(BoxStream<'static, Result<StreamEvent, Error>>),
    BackingOff(Pin<Box<tokio::time::Sleep>>),
    Completing,
    Finished,
}

impl<'a> RetryingResponse<'a> {
    /// Retry the requests `start_attempt` makes, calling it once per
    /// attempt. The first call happens here.
    pub fn new<F, Fut>(policy: RetryPolicy, mut start_attempt: F) -> Self
    where
        F: FnMut() -> Fut + Send + 'a,
        Fut: Future<Output = Result<Response, Error>> + Send + 'a,
    {
        let mut start_attempt: Attempt<'a> = Box::new(move || Box::pin(start_attempt()));
        let state = State::Requesting(start_attempt());
        Self {
            policy,
            attempt: 1,
            start_attempt,
            accumulator: ResponseAccumulator::new(),
            state,
        }
    }

    /// Drain the stream and return the final response. Each retry is
    /// logged at `warn`, since nothing else observes it.
    pub async fn buffer(mut self) -> Result<CompleteResponse, Error> {
        use futures_util::StreamExt;
        while let Some(item) = self.next().await {
            match item? {
                RetryEvent::Event(_) => {}
                RetryEvent::Reset(retry) => tracing::warn!(
                    attempt = retry.attempt,
                    max_attempts = retry.max_attempts,
                    delay_ms = retry.delay.as_millis() as u64,
                    error = %retry.error,
                    "retrying after transient failure",
                ),
                RetryEvent::Complete(response) => return Ok(response),
            }
        }
        unreachable!("a retrying response ends with `Complete` or an error")
    }

    /// The response the current attempt has accumulated so far, with the
    /// [`crate::FinishReason::Incomplete`] finish reason unless its `Done`
    /// has arrived. Empty after a [`RetryEvent::Reset`] or
    /// [`RetryEvent::Complete`].
    pub fn into_partial(self) -> CompleteResponse {
        self.accumulator.finalize()
    }

    /// Route a failed attempt through the policy: either a reset and a
    /// backoff, or the end of the stream.
    fn fail(&mut self, error: Error) -> Result<RetryEvent, Error> {
        self.accumulator = ResponseAccumulator::new();
        match self.policy.delay_after(&error, self.attempt) {
            Some(delay) => {
                self.state = State::BackingOff(Box::pin(tokio::time::sleep(delay)));
                Ok(RetryEvent::Reset(RetryAttempt {
                    attempt: self.attempt,
                    max_attempts: self.policy.max_attempts,
                    delay,
                    error,
                }))
            }
            None => {
                self.state = State::Finished;
                Err(error)
            }
        }
    }
}

impl Stream for RetryingResponse<'_> {
    type Item = Result<RetryEvent, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            let failure = match &mut this.state {
                State::Requesting(request) => match ready!(request.as_mut().poll(cx)) {
                    Ok(response) => {
                        this.state = State::Streaming(response.stream());
                        continue;
                    }
                    Err(error) => error,
                },
                State::Streaming(stream) => match ready!(stream.as_mut().poll_next(cx)) {
                    Some(Ok(event)) => {
                        let done = matches!(event, StreamEvent::Done { .. });
                        match this.accumulator.process_event(event.clone()) {
                            Ok(()) => {
                                if done {
                                    this.state = State::Completing;
                                }
                                return Poll::Ready(Some(Ok(RetryEvent::Event(event))));
                            }
                            Err(error) => error,
                        }
                    }
                    Some(Err(error)) => error,
                    None => Error::StreamTruncated {
                        parts_received: this.accumulator.parts_received(),
                    },
                },
                State::BackingOff(sleep) => {
                    ready!(sleep.as_mut().poll(cx));
                    this.attempt = this.attempt.saturating_add(1);
                    this.state = State::Requesting((this.start_attempt)());
                    continue;
                }
                State::Completing => {
                    this.state = State::Finished;
                    let response = std::mem::take(&mut this.accumulator).finalize();
                    return Poll::Ready(Some(Ok(RetryEvent::Complete(response))));
                }
                State::Finished => return Poll::Ready(None),
            };
            return Poll::Ready(Some(this.fail(failure)));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    use futures_util::StreamExt;

    use super::*;
    use crate::providers::mock::{Chunking, MockProvider, MockResponse};
    use crate::{FinishReason, PartKind, Usage};

    fn policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            initial_backoff: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(60),
            jitter: 0.0,
        }
    }

    fn config() -> Config {
        Config::builder("test-model").build()
    }

    /// The items of a finished stream, with each event reduced to a
    /// label so a whole sequence reads as one assertion.
    async fn labels(mut stream: RetryingResponse<'_>) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(match item {
                Ok(RetryEvent::Event(StreamEvent::Delta { delta, .. })) => format!("delta {delta}"),
                Ok(RetryEvent::Event(StreamEvent::Done { .. })) => "done".into(),
                Ok(RetryEvent::Event(_)) => continue,
                Ok(RetryEvent::Reset(retry)) => format!("reset {}", retry.attempt),
                Ok(RetryEvent::Complete(response)) => format!("complete {}", response.text()),
                Err(error) => format!("error {error}"),
            });
        }
        out
    }

    #[tokio::test(start_paused = true)]
    async fn resets_then_retries_after_a_retryable_mid_stream_error() {
        let provider = MockProvider::builder()
            .chunking(Chunking::None)
            .reply(MockResponse::text("partial").with_stream_error(Error::rate_limit(None, "slow")))
            .reply("whole")
            .build();
        let log = provider.call_log();
        let (prompt, config) = (Prompt::user("hi"), config());
        let stream = generate_with_retry(&provider, &prompt, &config, policy(4));
        fn assert_send<T: Send>(_: &T) {}
        assert_send(&stream);

        let items = labels(stream).await;

        assert_eq!(
            items,
            [
                "delta partial",
                "reset 1",
                "delta whole",
                "done",
                "complete whole"
            ],
        );
        assert_eq!(log.len(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn retries_a_failure_before_the_stream_starts() {
        let provider = MockProvider::builder()
            .chunking(Chunking::None)
            .fail(Error::provider_with_status("Mock", 503, "unavailable"))
            .reply("ok")
            .build();
        let (prompt, config) = (Prompt::user("hi"), config());

        let items = labels(generate_with_retry(&provider, &prompt, &config, policy(4))).await;

        assert_eq!(items, ["reset 1", "delta ok", "done", "complete ok"]);
    }

    #[tokio::test(start_paused = true)]
    async fn retries_a_stream_that_ends_without_done() {
        let provider = MockProvider::builder()
            .chunking(Chunking::None)
            .reply(MockResponse::raw_events(vec![
                StreamEvent::PartStart {
                    index: 0,
                    kind: PartKind::Text,
                },
                StreamEvent::Delta {
                    index: 0,
                    delta: "cut".into(),
                },
            ]))
            .reply("ok")
            .build();
        let (prompt, config) = (Prompt::user("hi"), config());
        let mut stream = generate_with_retry(&provider, &prompt, &config, policy(4));

        let mut reset = None;
        while let Some(item) = stream.next().await {
            if let RetryEvent::Reset(retry) = item.unwrap() {
                reset = Some(retry);
            }
        }

        let reset = reset.expect("the truncated attempt is retried");
        assert!(matches!(
            reset.error,
            Error::StreamTruncated { parts_received: 1 }
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn passes_a_non_retryable_error_straight_through() {
        let provider = MockProvider::builder()
            .chunking(Chunking::None)
            .reply(MockResponse::text("partial").with_stream_error(Error::auth("bad key")))
            .reply("never")
            .build();
        let log = provider.call_log();
        let (prompt, config) = (Prompt::user("hi"), config());
        let mut stream = generate_with_retry(&provider, &prompt, &config, policy(4));

        assert!(matches!(
            stream.next().await,
            Some(Ok(RetryEvent::Event(StreamEvent::PartStart { .. })))
        ));
        assert!(matches!(
            stream.next().await,
            Some(Ok(RetryEvent::Event(StreamEvent::Delta { .. })))
        ));
        assert!(matches!(
            stream.next().await,
            Some(Ok(RetryEvent::Event(StreamEvent::PartEnd { .. })))
        ));
        assert!(matches!(stream.next().await, Some(Err(Error::Auth { .. }))));
        assert!(stream.next().await.is_none());
        assert_eq!(log.len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn an_out_of_order_event_is_not_retried() {
        let provider = MockProvider::builder()
            .reply(MockResponse::raw_events(vec![StreamEvent::Delta {
                index: 3,
                delta: "orphan".into(),
            }]))
            .reply("never")
            .build();
        let log = provider.call_log();
        let (prompt, config) = (Prompt::user("hi"), config());

        let result = generate_with_retry(&provider, &prompt, &config, policy(4))
            .buffer()
            .await;

        assert!(matches!(result, Err(Error::Provider { .. })));
        assert_eq!(log.len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn exhausting_the_policy_returns_the_last_error() {
        let attempts = AtomicU32::new(0);
        let stream = RetryingResponse::new(policy(3), || {
            let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
            async move { Err(Error::rate_limit(None, format!("attempt {attempt}"))) }
        });
        let start = tokio::time::Instant::now();

        let items = labels(stream).await;

        assert_eq!(
            items,
            ["reset 1", "reset 2", "error rate limit exceeded: attempt 3"],
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(start.elapsed(), Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn reset_carries_the_policy_delay_and_honours_retry_after() {
        let provider = MockProvider::builder()
            .fail(Error::rate_limit(Some(7), "slow"))
            .reply("ok")
            .build();
        let (prompt, config) = (Prompt::user("hi"), config());
        let mut stream = generate_with_retry(&provider, &prompt, &config, policy(4));
        let start = tokio::time::Instant::now();

        let Some(Ok(RetryEvent::Reset(retry))) = stream.next().await else {
            panic!("expected a reset");
        };
        assert_eq!((retry.attempt, retry.max_attempts), (1, 4));
        assert_eq!(retry.delay, Duration::from_secs(7));
        assert!(matches!(retry.error, Error::RateLimit { .. }));

        stream.next().await;
        assert_eq!(start.elapsed(), Duration::from_secs(7));
    }

    #[tokio::test(start_paused = true)]
    async fn stops_reading_an_attempt_at_its_done() {
        let provider = MockProvider::builder()
            .reply(MockResponse::raw_events(vec![
                StreamEvent::Done {
                    finish_reason: FinishReason::Stop,
                    usage: Usage::default(),
                },
                StreamEvent::Delta {
                    index: 9,
                    delta: "after done".into(),
                },
            ]))
            .build();
        let (prompt, config) = (Prompt::user("hi"), config());

        let items = labels(generate_with_retry(&provider, &prompt, &config, policy(4))).await;

        assert_eq!(items, ["done", "complete "]);
    }

    #[tokio::test(start_paused = true)]
    async fn buffer_returns_the_successful_attempt_alone() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("stale text").with_stream_error(Error::rate_limit(None, "x")))
            .reply(MockResponse::text("fresh").usage(Usage {
                input_tokens: 3,
                output_tokens: 5,
                ..Usage::default()
            }))
            .build();
        let (prompt, config) = (Prompt::user("hi"), config());

        let response = generate_with_retry(&provider, &prompt, &config, policy(4))
            .buffer()
            .await
            .unwrap();

        assert_eq!(response.text(), "fresh");
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(
            (response.usage.input_tokens, response.usage.output_tokens),
            (3, 5)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_stream_during_backoff_cancels_the_retry() {
        let attempts = Arc::new(AtomicU32::new(0));
        let counter = attempts.clone();
        let mut stream = RetryingResponse::new(policy(4), move || {
            counter.fetch_add(1, Ordering::SeqCst);
            async { Err(Error::rate_limit(None, "slow")) }
        });

        assert!(matches!(
            stream.next().await,
            Some(Ok(RetryEvent::Reset(_)))
        ));
        drop(stream);
        tokio::time::sleep(Duration::from_secs(120)).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn into_partial_returns_what_the_current_attempt_streamed() {
        let provider = MockProvider::builder()
            .chunking(Chunking::Words)
            .reply("one two three")
            .build();
        let (prompt, config) = (Prompt::user("hi"), config());
        let mut stream = generate_with_retry(&provider, &prompt, &config, policy(4));

        // PartStart, then the first two word deltas.
        for _ in 0..3 {
            stream.next().await.unwrap().unwrap();
        }
        let partial = stream.into_partial();

        assert_eq!(partial.text(), "one two ");
        assert_eq!(partial.finish_reason, FinishReason::Incomplete);
    }

    #[tokio::test(start_paused = true)]
    async fn into_partial_after_a_reset_drops_the_failed_attempt() {
        let provider = MockProvider::builder()
            .chunking(Chunking::None)
            .reply(MockResponse::text("stale").with_stream_error(Error::rate_limit(None, "x")))
            .build();
        let (prompt, config) = (Prompt::user("hi"), config());
        let mut stream = generate_with_retry(&provider, &prompt, &config, policy(4));

        while let Some(item) = stream.next().await {
            if matches!(item, Ok(RetryEvent::Reset(_))) {
                break;
            }
        }

        assert!(stream.into_partial().content.is_empty());
    }
}
