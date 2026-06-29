//! In-process mock provider for testing downstream code.
//!
//! [`MockProvider`] implements [`Provider`] without any network round-trip
//! or credentials. It speaks in the crate's own semantic types
//! ([`AssistantPart`], [`FunctionCall`], …) rather than provider wire
//! formats: you describe *what the assistant should say* as a
//! [`MockResponse`], and the provider synthesizes the corresponding
//! [`StreamEvent`] sequence and feeds it through the real
//! [`crate::accumulator::ResponseAccumulator`]. Code under test therefore
//! exercises the genuine streaming / buffering path.
//!
//! # Chunking
//!
//! Full messages are automatically split into multiple
//! [`StreamEvent::Delta`]s according to the provider's [`Chunking`]
//! setting (default: [`Chunking::Words`]), so a consumer that renders
//! tokens incrementally sees realistic streaming. Set it once on the
//! provider; it applies to every reply.
//!
//! # Modes
//!
//! - **Scripted queue** ([`MockProvider::builder`]) — the common case: a
//!   FIFO of replies popped one per [`Provider::generate`] call. When the
//!   queue is exhausted, `generate` returns an [`Error`] so an unexpected
//!   extra call is caught loudly.
//! - **Fixed** ([`MockProvider::with_text`] / [`MockProvider::always`]) —
//!   the same reply on every call, forever.
//! - **Dynamic** ([`MockProvider::with_handler`]) — a closure that picks a
//!   reply based on the incoming [`Prompt`] / [`crate::RawConfig`],
//!   enabling full tool-call-loop tests.
//!
//! Every mode records the `(Prompt, RawConfig)` of each call; grab a
//! [`CallLog`] via [`MockProvider::call_log`] *before* moving the provider
//! into the code under test, then assert on what your code actually sent.
//!
//! ```no_run
//! use platformed_llm::providers::mock::{MockProvider, MockResponse, Chunking};
//! use platformed_llm::{generate, Config, Prompt};
//!
//! # async fn demo() -> Result<(), platformed_llm::Error> {
//! let provider = MockProvider::builder()
//!     .chunking(Chunking::Words)
//!     .reply("Hello, world!")
//!     .reply(MockResponse::text("a second turn"))
//!     .build();
//!
//! let log = provider.call_log();
//! let cfg = Config::builder("test-model").build();
//! let text = generate(&provider, &Prompt::user("hi"), &cfg)
//!     .await?
//!     .text()
//!     .await?;
//! assert_eq!(text, "Hello, world!");
//! assert_eq!(log.len(), 1);
//! # Ok(()) }
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::types::{
    AssistantPart, FinishReason, FunctionCall, PartKind, PartUpdate, StreamEvent, Usage,
};
use crate::{Error, Prompt, Provider, RawConfig, Response};

/// How a [`MockResponse`]'s text / tool-argument content is split into
/// streaming [`StreamEvent::Delta`]s.
///
/// Chunking is purely cosmetic with respect to the *buffered* result —
/// the accumulator concatenates deltas, so the final
/// [`crate::CompleteResponse`] is identical regardless of strategy. It
/// matters only to consumers that observe the live event stream.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Chunking {
    /// Emit each part's content as a single delta.
    None,
    /// Split on word boundaries — each chunk is a word together with its
    /// trailing whitespace (so concatenation reproduces the original).
    /// This is the default.
    #[default]
    Words,
    /// Fixed-size chunks of at most `N` Unicode scalar values. A value of
    /// `0` is treated as `1`.
    Chars(usize),
}

impl Chunking {
    /// Split `text` into chunks according to this strategy. Concatenating
    /// the result always reproduces `text` exactly.
    fn split(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        match self {
            Chunking::None => vec![text.to_string()],
            Chunking::Words => split_words(text),
            Chunking::Chars(n) => {
                let n = (*n).max(1);
                let mut chunks = Vec::new();
                let mut cur = String::new();
                let mut count = 0;
                for ch in text.chars() {
                    cur.push(ch);
                    count += 1;
                    if count == n {
                        chunks.push(std::mem::take(&mut cur));
                        count = 0;
                    }
                }
                if !cur.is_empty() {
                    chunks.push(cur);
                }
                chunks
            }
        }
    }
}

/// Split into "word + trailing whitespace" chunks. Concatenation
/// reproduces the input exactly; a leading run of whitespace becomes its
/// own chunk.
fn split_words(s: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            cur.push(ch);
            in_word = false;
        } else {
            if !in_word && !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            cur.push(ch);
            in_word = true;
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// A canned assistant turn the [`MockProvider`] will stream back.
///
/// Build one with [`MockResponse::text`], [`MockResponse::tool_call`],
/// [`MockResponse::from_parts`], or [`MockResponse::raw_events`], then
/// optionally attach [`MockResponse::usage`] or
/// [`MockResponse::with_stream_error`]. A `&str` / `String` converts into
/// a text response, so builder methods accept either directly.
#[derive(Debug, Clone)]
pub struct MockResponse(Repr);

#[derive(Debug, Clone)]
enum Repr {
    /// Synthesize events from these parts (chunking applies), then a
    /// terminal `Done` — unless `stream_error` is set, in which case
    /// the stream ends with `Err(error)` instead. The error lives
    /// behind an `Arc` because `Error` isn't `Clone` and we need
    /// `MockResponse: Clone` for `MockProvider::Always` mode; the
    /// `Arc` is unwrapped (or, on the shared-ref fallback path,
    /// rebuilt preserving retryability) when the response lowers to
    /// events.
    Parts {
        content: Vec<AssistantPart>,
        finish_reason: FinishReason,
        usage: Usage,
        stream_error: Option<std::sync::Arc<Error>>,
    },
    /// Emit these events verbatim. Chunking does *not* apply; the caller
    /// is responsible for a well-formed sequence (monotonic part indices,
    /// terminal `Done`).
    Raw(Vec<StreamEvent>),
}

impl MockResponse {
    /// A plain-text turn that finishes with [`FinishReason::Stop`].
    pub fn text(content: impl Into<String>) -> Self {
        Self(Repr::Parts {
            content: vec![AssistantPart::Text {
                content: content.into(),
                annotations: Vec::new(),
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            stream_error: None,
        })
    }

    /// A turn that emits a single tool call and finishes with
    /// [`FinishReason::ToolCalls`].
    pub fn tool_call(call: FunctionCall) -> Self {
        Self::tool_calls(vec![call])
    }

    /// A turn that emits several tool calls (in order) and finishes with
    /// [`FinishReason::ToolCalls`].
    pub fn tool_calls(calls: Vec<FunctionCall>) -> Self {
        Self(Repr::Parts {
            content: calls.into_iter().map(AssistantPart::ToolCall).collect(),
            finish_reason: FinishReason::ToolCalls,
            usage: Usage::default(),
            stream_error: None,
        })
    }

    /// A turn assembled from arbitrary [`AssistantPart`]s with an explicit
    /// finish reason — full control over interleaved text / reasoning /
    /// tool calls. Usage defaults to zero; override with
    /// [`MockResponse::usage`].
    pub fn from_parts(content: Vec<AssistantPart>, finish_reason: FinishReason) -> Self {
        Self(Repr::Parts {
            content,
            finish_reason,
            usage: Usage::default(),
            stream_error: None,
        })
    }

    /// Emit a pre-built [`StreamEvent`] sequence verbatim — the escape
    /// hatch for testing edge cases (custom delta granularity, malformed
    /// streams). [`Chunking`] does not apply, and [`MockResponse::usage`]
    /// / [`MockResponse::with_stream_error`] are no-ops on a raw response.
    pub fn raw_events(events: Vec<StreamEvent>) -> Self {
        Self(Repr::Raw(events))
    }

    /// Set the token-usage counters reported by the terminal `Done`.
    /// No-op on a [`MockResponse::raw_events`] response.
    pub fn usage(mut self, usage: Usage) -> Self {
        if let Repr::Parts { usage: u, .. } = &mut self.0 {
            *u = usage;
        }
        self
    }

    /// Make the turn fail *mid-stream*: after streaming any content parts,
    /// the stream ends with `Err(err)` (and no terminal `Done`), so
    /// [`Response::buffer`] / [`Response::text`] surface that exact
    /// typed error. Use this to test partial-then-failed streaming
    /// with any [`Error`] variant — e.g. `with_stream_error(
    /// Error::rate_limit(Some(0), "overloaded"))` to simulate an
    /// Anthropic mid-stream rate limit. For a failure *before any*
    /// stream is returned, script [`MockProviderBuilder::fail`]
    /// instead. No-op on a [`MockResponse::raw_events`] response.
    pub fn with_stream_error(mut self, err: Error) -> Self {
        if let Repr::Parts { stream_error, .. } = &mut self.0 {
            *stream_error = Some(std::sync::Arc::new(err));
        }
        self
    }
}

impl From<&str> for MockResponse {
    fn from(s: &str) -> Self {
        MockResponse::text(s)
    }
}

impl From<String> for MockResponse {
    fn from(s: String) -> Self {
        MockResponse::text(s)
    }
}

/// Lower a [`MockResponse`] into the event sequence a real provider would
/// stream.
fn lower_response(resp: MockResponse, chunking: &Chunking) -> Vec<Result<StreamEvent, Error>> {
    match resp.0 {
        Repr::Raw(events) => events.into_iter().map(Ok).collect(),
        Repr::Parts {
            content,
            finish_reason,
            usage,
            stream_error,
        } => {
            let mut out = Vec::new();
            let mut index = 0u32;
            for part in content {
                if push_part_events(&mut out, index, part, chunking) {
                    index += 1;
                }
            }
            match stream_error {
                Some(error) => out.push(Err(unwrap_shared_error(error))),
                None => out.push(Ok(StreamEvent::Done {
                    finish_reason,
                    usage,
                })),
            }
            out
        }
    }
}

/// Extract an owned [`Error`] from an [`std::sync::Arc<Error>`],
/// preserving the inner error's variant identity wherever possible
/// and falling back to a classification-preserving wrap when it
/// can't.
///
/// `MockResponse` holds the mid-stream error behind an `Arc` so the
/// containing response can be `Clone` (required for
/// `MockProvider::Always` mode). At the stream-lowering boundary we
/// try [`std::sync::Arc::try_unwrap`] first — that succeeds in
/// `scripted` / `queue` modes where the response is popped and
/// consumed exactly once. In `Always` mode the original `Arc` stays
/// alive in `Mode::Always(response)` while a clone goes to the
/// caller, so `try_unwrap` *always* fails there and we take the
/// fallback path.
///
/// The fallback rebuilds the error by hand. Variants that don't
/// carry non-`Clone` payloads (`RateLimit`, `Auth`,
/// `ContextWindowExceeded`, `ModelNotAvailable`, `InvalidPrompt`,
/// `Config`, `Compaction`, `UnsupportedInput`) are reconstructed
/// faithfully so callers can match on them. The remaining variants
/// (`Transport` — wraps a non-`Clone` `reqwest::Error`,
/// `Serialization` — same, and `Provider` — easiest to rebuild
/// from-scratch) collapse to a synthetic `Provider("Mock", …)`
/// carrying the inner's `is_retryable()` verdict. Without that
/// preservation, a shared mid-stream rate limit would silently
/// downgrade to a non-retryable provider error and the caller's
/// retry loop would give up.
fn unwrap_shared_error(error: std::sync::Arc<Error>) -> Error {
    std::sync::Arc::try_unwrap(error).unwrap_or_else(|arc| {
        // Reconstruct cloneable variants by hand so callers can
        // still match on the original tag (`compaction` needs to
        // see `ContextWindowExceeded`, not a generic provider error).
        match &*arc {
            Error::RateLimit {
                retry_after,
                message,
            } => Error::RateLimit {
                retry_after: *retry_after,
                message: message.clone(),
            },
            Error::Auth { status, message } => Error::Auth {
                status: *status,
                message: message.clone(),
            },
            Error::ContextWindowExceeded { provider, message } => Error::ContextWindowExceeded {
                provider,
                message: message.clone(),
            },
            Error::ModelNotAvailable(s) => Error::ModelNotAvailable(s.clone()),
            Error::InvalidPrompt(s) => Error::InvalidPrompt(s.clone()),
            Error::Config(s) => Error::Config(s.clone()),
            Error::Compaction { reason } => Error::Compaction {
                reason: reason.clone(),
            },
            Error::UnsupportedInput { provider, modality } => {
                Error::UnsupportedInput { provider, modality }
            }
            // Provider rebuilt from-scratch (cheaper than figuring
            // out which fields to preserve when the most-common case
            // is a test-supplied error anyway). Falls into the
            // catch-all for the non-cloneable variants too
            // (`Transport`, `Serialization`).
            other => Error::Provider {
                provider: "Mock",
                status: None,
                retryable: other.is_retryable(),
                retry_after: other.retry_after(),
                message: format!("mid-stream error (cloned): {arc}"),
            },
        }
    })
}

/// Push the events for one part at `index`. Returns `false` (without
/// emitting anything) for parts that have no streaming representation, so
/// the caller leaves `index` unchanged and the emitted indices stay
/// contiguous from 0 — which the accumulator requires.
fn push_part_events(
    out: &mut Vec<Result<StreamEvent, Error>>,
    index: u32,
    part: AssistantPart,
    chunking: &Chunking,
) -> bool {
    let deltas = |out: &mut Vec<Result<StreamEvent, Error>>, text: &str| {
        for delta in chunking.split(text) {
            out.push(Ok(StreamEvent::Delta { index, delta }));
        }
    };

    match part {
        AssistantPart::Text {
            content,
            annotations,
        } => {
            out.push(Ok(StreamEvent::PartStart {
                index,
                kind: PartKind::Text,
            }));
            deltas(out, &content);
            for annotation in annotations {
                out.push(Ok(StreamEvent::PartUpdate {
                    index,
                    update: PartUpdate::Annotation(annotation),
                }));
            }
            out.push(Ok(StreamEvent::PartEnd { index }));
            true
        }
        AssistantPart::Reasoning { content, signature } => {
            out.push(Ok(StreamEvent::PartStart {
                index,
                kind: PartKind::Reasoning,
            }));
            deltas(out, &content);
            if let Some(sig) = signature {
                out.push(Ok(StreamEvent::PartUpdate {
                    index,
                    update: PartUpdate::Signature(sig),
                }));
            }
            out.push(Ok(StreamEvent::PartEnd { index }));
            true
        }
        AssistantPart::RedactedReasoning { data } => {
            out.push(Ok(StreamEvent::PartStart {
                index,
                kind: PartKind::RedactedReasoning { data },
            }));
            out.push(Ok(StreamEvent::PartEnd { index }));
            true
        }
        AssistantPart::Refusal(content) => {
            out.push(Ok(StreamEvent::PartStart {
                index,
                kind: PartKind::Refusal,
            }));
            deltas(out, &content);
            out.push(Ok(StreamEvent::PartEnd { index }));
            true
        }
        AssistantPart::ToolCall(call) => {
            out.push(Ok(StreamEvent::PartStart {
                index,
                kind: PartKind::ToolCall {
                    call_id: call.call_id,
                    name: call.name,
                },
            }));
            deltas(out, &call.arguments);
            out.push(Ok(StreamEvent::PartEnd { index }));
            true
        }
        AssistantPart::BuiltinToolCall {
            kind,
            arguments,
            result,
        } => {
            out.push(Ok(StreamEvent::PartStart {
                index,
                kind: PartKind::BuiltinToolCall { kind },
            }));
            deltas(out, &arguments);
            if let Some(result) = result {
                out.push(Ok(StreamEvent::PartUpdate {
                    index,
                    update: PartUpdate::BuiltinToolResult(result),
                }));
            }
            out.push(Ok(StreamEvent::PartEnd { index }));
            true
        }
        AssistantPart::Continuation(continuation) => {
            out.push(Ok(StreamEvent::PartStart {
                index,
                kind: PartKind::Continuation(continuation),
            }));
            out.push(Ok(StreamEvent::PartEnd { index }));
            true
        }
        AssistantPart::CacheBreakpoint => {
            // Input-only marker — there is no `PartKind::CacheBreakpoint`,
            // so it has no streaming representation. Drop it (and don't
            // advance the index) rather than emit a malformed event.
            tracing::debug!("MockProvider: dropping input-only CacheBreakpoint part");
            false
        }
    }
}

/// One reply in a scripted queue: either a streamed response or an
/// immediate `generate`-level error.
enum Reply {
    Respond(MockResponse),
    Fail(Error),
}

/// A single recorded [`Provider::generate`] invocation.
#[derive(Debug, Clone)]
pub struct RecordedCall {
    /// The prompt the caller passed.
    pub prompt: Prompt,
    /// The config the caller passed.
    pub config: RawConfig,
}

/// A cheap, cloneable handle to a [`MockProvider`]'s recorded calls.
///
/// Obtain it via [`MockProvider::call_log`] before moving the provider
/// into the code under test; the handle keeps observing as calls land.
#[derive(Clone)]
pub struct CallLog {
    inner: Arc<Mutex<Vec<RecordedCall>>>,
}

impl CallLog {
    /// Snapshot the recorded calls in invocation order.
    pub fn calls(&self) -> Vec<RecordedCall> {
        self.inner.lock().expect("CallLog mutex poisoned").clone()
    }

    /// Number of `generate` calls recorded so far.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("CallLog mutex poisoned").len()
    }

    /// Whether no calls have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

type Handler = Box<dyn Fn(&Prompt, &RawConfig) -> MockResponse + Send + Sync>;

enum Mode {
    Queue(Mutex<VecDeque<Reply>>),
    Always(MockResponse),
    Handler(Handler),
}

/// A [`Provider`] that returns canned responses for tests. See the
/// [module docs](self) for the full picture.
pub struct MockProvider {
    mode: Mode,
    chunking: Chunking,
    log: Arc<Mutex<Vec<RecordedCall>>>,
    /// Cooperative rate limiter consulted before each scripted reply
    /// is yielded — included so multi-tenant / priority behaviour can
    /// be tested without a network round-trip. Defaults to
    /// [`crate::NoOpRateLimiter`].
    rate_limiter: crate::rate_limit::SharedRateLimiter,
}

impl MockProvider {
    fn new(mode: Mode, chunking: Chunking) -> Self {
        Self {
            mode,
            chunking,
            log: Arc::new(Mutex::new(Vec::new())),
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        }
    }

    /// Start building a scripted-queue provider. Add replies with
    /// [`MockProviderBuilder::reply`] / [`MockProviderBuilder::fail`] and
    /// finish with [`MockProviderBuilder::build`].
    pub fn builder() -> MockProviderBuilder {
        MockProviderBuilder {
            chunking: Chunking::default(),
            replies: VecDeque::new(),
        }
    }

    /// A provider that replies with the same text on every call, forever.
    pub fn with_text(text: impl Into<String>) -> Self {
        Self::new(Mode::Always(MockResponse::text(text)), Chunking::default())
    }

    /// A provider that returns the same [`MockResponse`] on every call,
    /// forever.
    pub fn always(response: impl Into<MockResponse>) -> Self {
        Self::new(Mode::Always(response.into()), Chunking::default())
    }

    /// A provider that derives each reply from the incoming prompt and
    /// config — ideal for driving a tool-call loop (return a tool call,
    /// then text once the tool result comes back). For a `generate`-level
    /// error, return a [`MockResponse::with_stream_error`]; immediate
    /// (pre-stream) errors are only scriptable via the queue builder.
    pub fn with_handler<F>(handler: F) -> Self
    where
        F: Fn(&Prompt, &RawConfig) -> MockResponse + Send + Sync + 'static,
    {
        Self::new(Mode::Handler(Box::new(handler)), Chunking::default())
    }

    /// Override the [`Chunking`] strategy (default [`Chunking::Words`]).
    pub fn with_chunking(mut self, chunking: Chunking) -> Self {
        self.chunking = chunking;
        self
    }

    /// Attach a shared [`crate::rate_limit::RateLimiter`]. The mock
    /// honours the limiter exactly the way a hosted provider would
    /// — acquire before "sending", observe `Success` after — so
    /// downstream tests can exercise scheduling, fairness, and
    /// rate-limit observation without standing up a real upstream.
    pub fn with_rate_limiter(mut self, limiter: crate::rate_limit::SharedRateLimiter) -> Self {
        self.rate_limiter = limiter;
        self
    }

    /// A cloneable handle to this provider's recorded calls. Grab it
    /// before moving the provider into the code under test.
    pub fn call_log(&self) -> CallLog {
        CallLog {
            inner: self.log.clone(),
        }
    }

    fn next_reply(&self, prompt: &Prompt, config: &RawConfig) -> Result<Reply, Error> {
        match &self.mode {
            Mode::Always(response) => Ok(Reply::Respond(response.clone())),
            Mode::Handler(handler) => Ok(Reply::Respond(handler(prompt, config))),
            Mode::Queue(queue) => queue
                .lock()
                .expect("MockProvider queue mutex poisoned")
                .pop_front()
                .ok_or_else(|| {
                    Error::config(
                        "MockProvider: scripted reply queue exhausted \
                         (generate was called more times than replies were scripted)",
                    )
                }),
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn generate(&self, prompt: &Prompt, config: &RawConfig) -> Result<Response, Error> {
        self.log
            .lock()
            .expect("MockProvider log mutex poisoned")
            .push(RecordedCall {
                prompt: prompt.clone(),
                config: config.clone(),
            });

        // Honour the rate limiter exactly like a hosted provider —
        // acquire before yielding the scripted reply, observe with
        // the outcome derived from the reply.
        //
        // The `MockProvider/` prefix disambiguates from the hosted
        // providers' bucket-key shape (`OpenAI|…|model`, `Anthropic|…`,
        // …). A test wiring both a `MockProvider` and a real
        // `OpenAIProvider` behind the same limiter with the same
        // model name would otherwise share an AIMD bucket,
        // conflating fake and real backpressure signals.
        let scope = crate::rate_limit::RateScope {
            bucket_key: format!("MockProvider/{}", config.model),
            tenant: config.tenant.unwrap_or(uuid::Uuid::nil()),
            priority: config.priority.unwrap_or_default(),
        };
        let permit = self.rate_limiter.acquire(&scope).await?;

        match self.next_reply(prompt, config)? {
            Reply::Fail(error) => {
                // Surface a scripted rate-limit error to the limiter
                // so downstream tests can drive AIMD / parking
                // behaviour from the scripted queue.
                match &error {
                    Error::RateLimit { retry_after, .. } => {
                        permit.observe(crate::rate_limit::RateOutcome::RateLimited {
                            retry_after: *retry_after,
                            info: crate::rate_limit::ProviderRateInfo::default(),
                        });
                    }
                    _ => permit.observe(crate::rate_limit::RateOutcome::OtherFailure),
                }
                Err(error)
            }
            Reply::Respond(response) => {
                // Defer observation to stream-end so a scripted
                // `with_stream_error` mid-stream produces an
                // `OtherFailure` rather than a misleading `Success`.
                let events = lower_response(response, &self.chunking);
                let stream = futures_util::stream::iter(events);
                let observed = crate::rate_limit::observe_response_stream(
                    stream,
                    permit,
                    crate::rate_limit::ProviderRateInfo::default(),
                );
                Ok(Response::from_stream(observed))
            }
        }
    }
}

/// Builder for a scripted-queue [`MockProvider`]. See
/// [`MockProvider::builder`].
pub struct MockProviderBuilder {
    chunking: Chunking,
    replies: VecDeque<Reply>,
}

impl MockProviderBuilder {
    /// Set the [`Chunking`] strategy applied to every reply (default
    /// [`Chunking::Words`]).
    pub fn chunking(mut self, chunking: Chunking) -> Self {
        self.chunking = chunking;
        self
    }

    /// Queue a successful reply. Accepts a [`MockResponse`] or anything
    /// that converts into one (e.g. a `&str` / `String` for plain text).
    pub fn reply(mut self, response: impl Into<MockResponse>) -> Self {
        self.replies.push_back(Reply::Respond(response.into()));
        self
    }

    /// Queue a `generate`-level failure: the matching call returns
    /// `Err(error)` *before* any stream is produced (a connect/auth-style
    /// failure). For a failure *after* partial streaming, use
    /// [`MockResponse::with_stream_error`] on a [`MockProviderBuilder::reply`]
    /// instead.
    pub fn fail(mut self, error: Error) -> Self {
        self.replies.push_back(Reply::Fail(error));
        self
    }

    /// Finish building the provider.
    pub fn build(self) -> MockProvider {
        MockProvider::new(Mode::Queue(Mutex::new(self.replies)), self.chunking)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RawConfig {
        crate::Config::builder("test-model").build().raw().clone()
    }

    /// The Arc-shared fallback path of `unwrap_shared_error` must
    /// preserve the inner error's retryability so a downstream retry
    /// policy still matches the source's intent. Without this, a
    /// mid-stream `Error::RateLimit` whose Arc happened to be shared
    /// would downgrade to a non-retryable `Provider("Stream", …)`
    /// and the retry loop would give up.
    #[test]
    fn unwrap_shared_error_fallback_preserves_retry_after() {
        let inner = std::sync::Arc::new(Error::rate_limit(Some(7), "overloaded"));
        // Keep a second strong ref so `try_unwrap` fails.
        let _other = inner.clone();
        let unwrapped = unwrap_shared_error(inner);
        match unwrapped {
            Error::RateLimit { retry_after, .. } => {
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(7)));
            }
            other => panic!("expected RateLimit with retry_after preserved, got {other:?}"),
        }
    }

    /// Same as above but for a transient `Provider` error: the
    /// fallback path must carry the `retryable` flag through.
    #[test]
    fn unwrap_shared_error_fallback_preserves_retryable_flag() {
        let inner = std::sync::Arc::new(Error::provider_with_status(
            "OpenAI",
            503,
            "service unavailable",
        ));
        let _other = inner.clone();
        let unwrapped = unwrap_shared_error(inner);
        assert!(
            unwrapped.is_retryable(),
            "transient 5xx must remain retryable across the shared-Arc fallback, \
             got {unwrapped:?}",
        );
    }

    #[tokio::test]
    async fn scripted_queue_pops_in_order() {
        let provider = MockProvider::builder()
            .reply("first")
            .reply("second")
            .build();

        let a = provider.generate(&Prompt::user("x"), &cfg()).await.unwrap();
        assert_eq!(a.text().await.unwrap(), "first");
        let b = provider.generate(&Prompt::user("y"), &cfg()).await.unwrap();
        assert_eq!(b.text().await.unwrap(), "second");
    }

    #[tokio::test]
    async fn exhausted_queue_errors() {
        let provider = MockProvider::builder().reply("only").build();
        let _ = provider.generate(&Prompt::user("x"), &cfg()).await.unwrap();
        let err = provider
            .generate(&Prompt::user("y"), &cfg())
            .await
            .map(|_| ())
            .expect_err("queue should be exhausted");
        assert!(err.to_string().contains("exhausted"), "got: {err}");
    }

    #[tokio::test]
    async fn always_repeats_forever() {
        let provider = MockProvider::with_text("same");
        for _ in 0..3 {
            let r = provider.generate(&Prompt::user("x"), &cfg()).await.unwrap();
            assert_eq!(r.text().await.unwrap(), "same");
        }
    }

    #[tokio::test]
    async fn words_chunking_splits_into_multiple_deltas() {
        let provider = MockProvider::builder()
            .chunking(Chunking::Words)
            .reply("Hello, world!")
            .build();
        let (events, complete) = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let deltas = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Delta { .. }))
            .count();
        assert_eq!(deltas, 2, "expected one delta per word");
        assert_eq!(complete.text(), "Hello, world!");
    }

    #[tokio::test]
    async fn none_chunking_is_single_delta() {
        let provider = MockProvider::builder()
            .chunking(Chunking::None)
            .reply("Hello, world!")
            .build();
        let (events, complete) = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let deltas = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Delta { .. }))
            .count();
        assert_eq!(deltas, 1);
        assert_eq!(complete.text(), "Hello, world!");
    }

    #[tokio::test]
    async fn chars_chunking_respects_unicode_boundaries() {
        // 5 scalar values incl. a multibyte char; Chars(2) → 3 chunks.
        let provider = MockProvider::builder()
            .chunking(Chunking::Chars(2))
            .reply("aé8🦀z")
            .build();
        let (events, complete) = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let deltas = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Delta { .. }))
            .count();
        assert_eq!(deltas, 3);
        assert_eq!(complete.text(), "aé8🦀z");
    }

    #[tokio::test]
    async fn tool_call_round_trips() {
        let provider = MockProvider::builder()
            .reply(MockResponse::tool_call(FunctionCall {
                call_id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"Paris"}"#.into(),
                provider_signature: None,
            }))
            .build();
        let complete = provider
            .generate(&Prompt::user("weather?"), &cfg())
            .await
            .unwrap()
            .buffer()
            .await
            .unwrap();
        assert_eq!(complete.finish_reason, FinishReason::ToolCalls);
        let calls = complete.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[tokio::test]
    async fn usage_is_reported() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("hi").usage(Usage {
                input_tokens: 7,
                output_tokens: 3,
                ..Usage::default()
            }))
            .build();
        let complete = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .unwrap()
            .buffer()
            .await
            .unwrap();
        assert_eq!(complete.usage.input_tokens, 7);
        assert_eq!(complete.usage.output_tokens, 3);
    }

    #[tokio::test]
    async fn fail_returns_generate_level_error() {
        let provider = MockProvider::builder()
            .fail(Error::provider_with_status("MockProvider", 503, "down"))
            .build();
        let err = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .map(|_| ())
            .expect_err("should fail");
        assert!(err.to_string().contains("down"), "got: {err}");
    }

    #[tokio::test]
    async fn stream_error_surfaces_after_partial_content() {
        let provider = MockProvider::builder()
            .reply(
                MockResponse::text("partial")
                    .with_stream_error(Error::provider("Mock", "connection reset")),
            )
            .build();
        let err = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .unwrap()
            .buffer()
            .await
            .expect_err("mid-stream error");
        assert!(matches!(
            err,
            Error::Provider {
                provider: "Mock",
                ..
            }
        ));
        assert!(err.to_string().contains("connection reset"));
    }

    #[tokio::test]
    async fn handler_branches_on_prompt() {
        let provider = MockProvider::with_handler(|prompt, _config| {
            let asked_tool = prompt.items().iter().any(|item| {
                matches!(item, crate::InputItem::User { content }
                    if content.iter().any(|p| matches!(p, crate::UserPart::ToolResult { .. })))
            });
            if asked_tool {
                MockResponse::text("final answer")
            } else {
                MockResponse::tool_call(FunctionCall {
                    call_id: "c1".into(),
                    name: "lookup".into(),
                    arguments: "{}".into(),
                    provider_signature: None,
                })
            }
        });

        let first = provider
            .generate(&Prompt::user("go"), &cfg())
            .await
            .unwrap()
            .buffer()
            .await
            .unwrap();
        assert_eq!(first.finish_reason, FinishReason::ToolCalls);

        let second = provider
            .generate(&Prompt::user("go").with_tool_result("c1", "42"), &cfg())
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(second, "final answer");
    }

    #[tokio::test]
    async fn call_log_records_prompts_and_configs() {
        let provider = MockProvider::with_text("ok");
        let log = provider.call_log();
        assert!(log.is_empty());

        let cfg = crate::Config::builder("model-x").build();
        provider
            .generate(&Prompt::user("hello"), cfg.raw())
            .await
            .unwrap();

        assert_eq!(log.len(), 1);
        let calls = log.calls();
        assert_eq!(calls[0].config.model, "model-x");
        assert_eq!(calls[0].prompt.items().len(), 1);
    }

    #[tokio::test]
    async fn raw_events_emitted_verbatim() {
        let provider = MockProvider::builder()
            .chunking(Chunking::Words) // ignored for raw
            .reply(MockResponse::raw_events(vec![
                StreamEvent::PartStart {
                    index: 0,
                    kind: PartKind::Text,
                },
                StreamEvent::Delta {
                    index: 0,
                    delta: "raw".into(),
                },
                StreamEvent::PartEnd { index: 0 },
                StreamEvent::Done {
                    finish_reason: FinishReason::Stop,
                    usage: Usage::default(),
                },
            ]))
            .build();
        let (events, complete) = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(events.len(), 4);
        assert_eq!(complete.text(), "raw");
    }

    /// End-to-end wiring test: `MockProvider` + `InMemoryRateLimiter`
    /// + `ObservingStream`. A scripted 429 must be observed as
    /// `RateLimited` by the limiter (i.e. the bucket's `rps` halves
    /// on the next inspection), proving the whole permit-lifecycle
    /// flow — acquire → ObservingStream wraps → mid-stream Error
    /// surfaces → RatePermit::Drop fires observe — is wired
    /// correctly.
    #[tokio::test]
    async fn end_to_end_mock_provider_429_observed_by_limiter() {
        use std::sync::Arc;

        let limiter = Arc::new(
            crate::rate_limit::InMemoryRateLimiter::try_with_config(
                crate::rate_limit::InMemoryRateLimiterConfig {
                    initial_rps: 4.0,
                    min_rps: 0.1,
                    max_rps: 100.0,
                    additive_step: 1.0,
                    multiplicative_decrease: 0.5,
                    default_park: std::time::Duration::from_millis(1),
                    max_park: std::time::Duration::from_millis(10),
                },
            )
            .unwrap(),
        );
        let provider = MockProvider::builder()
            .reply("ok") // first success → rps stays at initial 4.0
            .fail(Error::rate_limit(Some(0), "synthetic 429"))
            .build()
            .with_rate_limiter(limiter.clone());

        // First call succeeds.
        let _ = provider
            .generate(&Prompt::user("x"), &cfg())
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        // Inspect the limiter: bucket should exist, rps unchanged.
        let rps_initial = limiter
            .first_bucket_rps()
            .expect("bucket must exist after one acquire");
        assert!(
            (rps_initial - 5.0).abs() < 0.01,
            "after one Success at additive_step=1 starting from initial_rps=4.0, \
             rps should grow to 5.0, got {rps_initial}",
        );

        // Second call: synthetic 429. The mock recognises
        // `Error::RateLimit` specifically and observes the permit as
        // `RateLimited` (the typed rate-limit path) rather than
        // `OtherFailure` — that's the wiring we're proving exists.
        match provider.generate(&Prompt::user("y"), &cfg()).await {
            Err(Error::RateLimit { .. }) => {}
            Err(other) => panic!("expected RateLimit, got {other:?}"),
            Ok(_) => panic!("expected Err"),
        }
        // The `OtherFailure` outcome triggers the AIMD halving (the
        // limiter doesn't know whether OtherFailure was overload-shaped
        // or a deterministic 4xx; defaults to a soft decrease).
        // After the halving from 5.0 with multiplicative_decrease 0.5,
        // rps should be 2.5.
        let rps_after_429 = limiter.first_bucket_rps().unwrap();
        assert!(
            (rps_after_429 - 2.5).abs() < 0.01,
            "after a failure with multiplicative_decrease=0.5, rps should halve to 2.5, \
             got {rps_after_429}",
        );
    }
}
