//! Cooperative rate limiting for multi-tenant deployments.
//!
//! The library exposes a [`RateLimiter`] trait that consumers
//! construct once and share across every [`crate::Provider`] they
//! build. Each request carries an opaque tenant key and a latency
//! priority on [`crate::Config`]; providers consult the limiter
//! before issuing the upstream HTTP call (so it can pace, gate, and
//! park requests) and feed observed rate-limit signals back so the
//! limiter's model of available capacity stays current.
//!
//! # When to use this
//!
//! Roll a limiter when you serve more than one tenant from a shared
//! provider quota (a SaaS calling Anthropic from a shared API key, a
//! workspace product letting each workspace burst into one OpenAI
//! org). Without one, a single noisy tenant can exhaust the quota
//! and tank every other tenant's user-facing latency. With one, the
//! quota is shared cooperatively: each tenant gets a fair slice
//! within a latency band, and a single user-facing request always
//! beats a backlog of batch jobs.
//!
//! Single-tenant deployments don't need this — the [`NoOpRateLimiter`]
//! is the default and adds zero overhead.
//!
//! # Sharing model
//!
//! One limiter instance is shared across providers and across
//! constructions of the same provider type:
//!
//! ```ignore
//! use std::sync::Arc;
//! use platformed_llm::rate_limit::{InMemoryRateLimiter, RateLimiter};
//!
//! let limiter: Arc<dyn RateLimiter> = Arc::new(InMemoryRateLimiter::new());
//!
//! let openai_a = OpenAIProvider::new(key_a)?.with_rate_limiter(limiter.clone());
//! let openai_b = OpenAIProvider::new(key_b)?.with_rate_limiter(limiter.clone());
//! let vertex   = GoogleProvider::new(...)?.with_rate_limiter(limiter.clone());
//! ```
//!
//! The limiter's internal state is keyed by `(provider, model)`, so
//! OpenAI's quota model is tracked independently from Google's, and
//! `gpt-4o`'s rate is tracked independently from `gpt-4o-mini`'s.
//! Cross-tenant fairness operates within each `(provider, model)`
//! bucket.
//!
//! # Scheduling
//!
//! [`InMemoryRateLimiter`] uses **strict priority** across tenants:
//! every [`Priority::Interactive`] request — from any tenant —
//! dispatches before any [`Priority::Standard`] request, which
//! dispatches before any [`Priority::Background`] request. **Tenant
//! fairness operates within a priority band**: when two tenants both
//! have Interactive requests queued, the limiter round-robins
//! between them so a flooding tenant can't starve a polite one at
//! the same priority.
//!
//! This is intentionally a strong-preemption model: a sustained
//! interactive load *can* starve background work. If your background
//! jobs must always make progress, batch them via a separate limiter
//! instance with its own quota — don't share the user-facing
//! limiter.
//!
//! # AIMD capacity tracking
//!
//! The limiter doesn't know how much quota the provider will grant
//! ahead of time. It starts conservatively (1 request per second for
//! a newly-observed `(provider, model)`) and learns from successful
//! responses, additively growing toward observed capacity. On any
//! 429 the available rate is **multiplicatively halved** — this is
//! what makes the limiter resilient when other clients share the
//! provider quota: even if our request count alone wouldn't trigger
//! a 429, observing one from a noisy neighbour tells us the
//! effective ceiling has dropped and we should back off below what
//! the provider's headers suggest.
//!
//! A provider-supplied `Retry-After` parks all queued requests for
//! that `(provider, model)` until the window elapses, then resumes
//! with the halved rate.
//!
//! # Provider-specific signals
//!
//! Each provider parses its own rate-limit headers into the
//! normalised [`ProviderRateInfo`] shape so the limiter doesn't need
//! provider-specific logic:
//!
//! - **OpenAI**: `x-ratelimit-{limit,remaining,reset}-{requests,tokens}`
//!   on every response (success or error).
//! - **Anthropic-via-Vertex**: only `Retry-After` on 429 plus
//!   mid-stream `overloaded_error` / `rate_limit_error` SSE events
//!   (which the lib normalises to [`crate::Error::RateLimit`] so the
//!   limiter sees them).
//! - **Gemini-via-Vertex**: `Retry-After` on 429.

use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;

use crate::Error;

/// Latency priority for a request within a tenant. Strictly ordered
/// — any [`Priority::Interactive`] request from any tenant
/// dispatches before any [`Priority::Standard`] / [`Priority::Background`]
/// request anywhere in the queue. See the module docs for the full
/// scheduling model.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// A user is waiting on this response — minimise queueing
    /// latency at the cost of starving lower-priority work. This is
    /// the default for [`crate::Config`].
    #[default]
    Interactive,
    /// Default for non-user-facing work that still needs reasonable
    /// throughput. Sits between interactive and background.
    Standard,
    /// Batch / non-time-sensitive work. Yields to interactive and
    /// standard requests; expects to wait when the upstream quota is
    /// busy.
    Background,
}

/// The per-request context the [`RateLimiter`] needs to schedule a
/// dispatch and update its model. Constructed by the provider just
/// before calling [`RateLimiter::acquire`].
///
/// The `bucket_key` is **opaque to the limiter** — providers compute
/// it from whatever combination of fields actually shares a rate-limit
/// bucket on the upstream. OpenAI uses `"OpenAI/{model}"`; Vertex
/// providers add the region (`"Vertex-Anthropic/{location}/{model}"`)
/// because each Vertex region has an independent quota. Two requests
/// with the same `bucket_key` share AIMD state; two with different
/// keys are tracked independently.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RateScope {
    /// Opaque bucket identifier. The limiter tracks AIMD and parking
    /// state per unique value. Providers compute this — see the
    /// struct docs.
    pub bucket_key: String,
    /// Opaque tenant identifier for fairness. The limiter
    /// round-robins between tenants within a priority band so one
    /// tenant's burst can't starve another at the same priority.
    /// [`Uuid`] is `Copy` and fixed-size (16 bytes) — pass it by
    /// value, no allocation per request. Map your own tenant
    /// identifier (workspace id, user id, …) to a stable UUID once
    /// per tenant (e.g. UUIDv5 over your id namespace) and reuse it.
    pub tenant: Uuid,
    /// Latency priority. See [`Priority`].
    pub priority: Priority,
}

/// Normalised rate-limit information parsed from a provider's
/// response. Each provider implementation populates the fields it
/// reports; the limiter's AIMD step reads `requests_remaining` /
/// `requests_reset` to understand observed capacity.
///
/// Token-budget headers were exposed in an earlier draft but no
/// in-tree limiter consumed them, so they've been removed for now —
/// adding them back is a non-breaking change thanks to
/// [`#[non_exhaustive]`].
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct ProviderRateInfo {
    /// Remaining request budget for the current window, if reported.
    pub requests_remaining: Option<u32>,
    /// Time until the request budget refills, if reported.
    pub requests_reset: Option<Duration>,
}

/// What happened to the in-flight request the permit was issued for.
/// Reported via [`RatePermit::observe`]; updates the limiter's AIMD
/// state and `Retry-After` parking.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum RateOutcome {
    /// The request completed cleanly (HTTP 2xx, stream terminated
    /// with `Done`). Includes any normalised header info parsed from
    /// the response.
    Success {
        /// Normalised rate-limit headers from the response.
        info: ProviderRateInfo,
    },
    /// The provider rate-limited us, either pre-stream (HTTP 429) or
    /// mid-stream (Anthropic `overloaded_error`). The limiter parks
    /// the bucket for `retry_after` (or its own default if `None`)
    /// and multiplicatively halves the available rate.
    RateLimited {
        /// Provider-supplied wait hint, if any. The limiter parks
        /// for at least this duration before resuming.
        retry_after: Option<Duration>,
        /// Any other rate-limit headers the provider sent alongside
        /// the 429 (often the same `x-ratelimit-*` shape as a
        /// successful response).
        info: ProviderRateInfo,
    },
    /// The request failed for a reason unrelated to rate limiting
    /// (auth, transport, malformed prompt, etc.). The limiter
    /// treats this as a no-op — it neither grows the AIMD rate nor
    /// shrinks it.
    OtherFailure,
    /// The caller dropped the permit without calling
    /// [`RatePermit::observe`]. The limiter treats this the same as
    /// [`Self::OtherFailure`] — no change to the rate.
    Cancelled,
}

/// An RAII handle the provider holds across a request. Created by
/// [`RateLimiter::acquire`]; consumed (with the outcome) via
/// [`RatePermit::observe`] when the request completes.
///
/// If the permit is dropped without `observe`, the limiter is
/// notified with [`RateOutcome::Cancelled`] — important so a
/// cancelled future / panic doesn't leak the slot.
///
/// `#[must_use]`: silently dropping a permit reports the request as
/// cancelled to the limiter, which mutes AIMD growth. That's the
/// right behaviour for actual cancellation but is usually a bug if
/// the caller forgot to call `observe` after a real response.
#[must_use = "drop a permit only on cancellation; otherwise call observe() with the request outcome"]
pub struct RatePermit {
    callback: Option<Box<dyn FnOnce(RateOutcome) + Send + 'static>>,
}

impl RatePermit {
    /// Build a permit from a closure the limiter will run when
    /// `observe` is called (or, failing that, on `Drop` with
    /// [`RateOutcome::Cancelled`]).
    ///
    /// Used by [`RateLimiter`] implementations; consumers receive a
    /// permit from [`RateLimiter::acquire`] and don't construct one
    /// directly.
    pub fn new(callback: impl FnOnce(RateOutcome) + Send + 'static) -> Self {
        Self {
            callback: Some(Box::new(callback)),
        }
    }

    /// A permit that does nothing on observe / drop. Used by
    /// [`NoOpRateLimiter`].
    pub fn noop() -> Self {
        Self { callback: None }
    }

    /// Inform the limiter of the request outcome. Consumes the
    /// permit so the closure can only run once.
    pub fn observe(mut self, outcome: RateOutcome) {
        if let Some(callback) = self.callback.take() {
            callback(outcome);
        }
    }
}

impl Drop for RatePermit {
    fn drop(&mut self) {
        if let Some(callback) = self.callback.take() {
            // Cancellation path — the caller dropped without
            // observing. Tell the limiter so it can free internal
            // bookkeeping; the AIMD model isn't updated since we
            // don't know whether the cancellation reflects real
            // capacity pressure.
            //
            // `catch_unwind` here is a trust-boundary guard, not a
            // panic-recovery trick: the callback is supplied by an
            // arbitrary [`RateLimiter`] impl (the in-tree
            // [`InMemoryRateLimiter`] is panic-free by construction
            // — its own `observe` uses a Drop guard for `wake_head`
            // — but a custom impl might not be). A permit can drop
            // *during* an outer user-code panic, and a panicking
            // callback at that point would compound into a process
            // abort. We swallow at the boundary to keep the outer
            // panic propagating cleanly.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                callback(RateOutcome::Cancelled);
            }));
        }
    }
}

impl std::fmt::Debug for RatePermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RatePermit")
            .field("observed", &self.callback.is_none())
            .finish()
    }
}

/// The pluggable rate-limiter interface.
///
/// Consumers construct one [`RateLimiter`] (typically
/// [`InMemoryRateLimiter`]) and share it across providers via
/// `Arc<dyn RateLimiter>` on each provider's builder. A
/// [`NoOpRateLimiter`] is installed by default on every provider —
/// `.with_rate_limiter(...)` overrides it.
///
/// Implementations should be cheap to clone via `Arc` and
/// thread-safe.
#[async_trait::async_trait]
pub trait RateLimiter: Send + Sync + 'static {
    /// Block until the limiter believes the request can proceed.
    /// Returns a [`RatePermit`] the provider holds across the HTTP
    /// round-trip and consumes via [`RatePermit::observe`] when the
    /// outcome is known.
    ///
    /// May return an error if the limiter rejects the request — not
    /// used by the in-tree implementations but reserved for custom
    /// limiters that enforce hard caps (e.g. per-tenant queue
    /// depth) instead of waiting unboundedly.
    async fn acquire(&self, scope: &RateScope) -> Result<RatePermit, Error>;
}

/// The default limiter — never gates, never tracks state, always
/// returns immediately with a permit that observes back to nothing.
///
/// Installed by default on every provider. Replacing it with
/// [`InMemoryRateLimiter`] (or a custom impl) is opt-in via
/// `.with_rate_limiter(...)`. Behaviour with this limiter is
/// observationally identical to a build without rate limiting at
/// all.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpRateLimiter;

#[async_trait::async_trait]
impl RateLimiter for NoOpRateLimiter {
    async fn acquire(&self, _scope: &RateScope) -> Result<RatePermit, Error> {
        Ok(RatePermit::noop())
    }
}

/// Helper: the [`Arc<dyn RateLimiter>`] each provider stores
/// internally. Constructing this from your own impl is a one-line
/// `Arc::new(my_impl) as Arc<dyn RateLimiter>` cast.
pub type SharedRateLimiter = Arc<dyn RateLimiter>;

/// Default `Arc<dyn RateLimiter>` — a single shared
/// [`NoOpRateLimiter`] every provider falls back to until the caller
/// overrides via `.with_rate_limiter(...)`.
///
/// Shares one `Arc` across every provider construction in the
/// process via [`std::sync::OnceLock`] so the no-op default doesn't
/// allocate per provider.
///
/// `dead_code` allowed for the no-feature build: only the hosted
/// providers (`openai`, `google`, `anthropic-vertex`) consume this,
/// and they're all behind cargo features.
#[allow(dead_code)]
pub(crate) fn default_shared_limiter() -> SharedRateLimiter {
    use std::sync::OnceLock;
    static DEFAULT: OnceLock<SharedRateLimiter> = OnceLock::new();
    DEFAULT.get_or_init(|| Arc::new(NoOpRateLimiter)).clone()
}

mod in_memory;
mod observe_stream;

pub use in_memory::{InMemoryRateLimiter, InMemoryRateLimiterConfig};
// Used by every hosted provider's `generate()` to wrap the success-path
// stream. `dead_code` allowed for the no-feature build where no provider
// imports it.
#[allow(unused_imports)]
pub(crate) use observe_stream::observe_response_stream;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_limiter_grants_immediately() {
        let limiter = NoOpRateLimiter;
        let scope = RateScope {
            bucket_key: "Test/test-model".into(),
            tenant: Uuid::nil(),
            priority: Priority::Interactive,
        };
        let permit = limiter.acquire(&scope).await.unwrap();
        permit.observe(RateOutcome::Success {
            info: ProviderRateInfo::default(),
        });
    }

    #[tokio::test]
    async fn permit_observe_consumes_callback() {
        // After observe, drop must not double-invoke. Tracked via a
        // shared counter the closure increments.
        use std::sync::atomic::{AtomicU32, Ordering};
        let calls = Arc::new(AtomicU32::new(0));
        let calls_for_cb = calls.clone();
        let permit = RatePermit::new(move |_outcome| {
            calls_for_cb.fetch_add(1, Ordering::SeqCst);
        });
        permit.observe(RateOutcome::Success {
            info: ProviderRateInfo::default(),
        });
        // Permit consumed by observe; nothing further to drop.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn permit_drop_invokes_callback_with_cancelled() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let calls = Arc::new(AtomicU32::new(0));
        let outcome_was_cancelled = Arc::new(AtomicU32::new(0));
        let calls_for_cb = calls.clone();
        let cancelled = outcome_was_cancelled.clone();
        {
            let _permit = RatePermit::new(move |outcome| {
                calls_for_cb.fetch_add(1, Ordering::SeqCst);
                if matches!(outcome, RateOutcome::Cancelled) {
                    cancelled.store(1, Ordering::SeqCst);
                }
            });
            // _permit drops here without observe.
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(outcome_was_cancelled.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn priority_ordering_is_strict() {
        // Strict preemption is encoded by the PartialOrd derive: the
        // first-declared variant is the smallest, so the scheduler's
        // `min` pulls Interactive first.
        assert!(Priority::Interactive < Priority::Standard);
        assert!(Priority::Standard < Priority::Background);
    }

    #[test]
    fn priority_defaults_to_interactive() {
        // Default config priority — most callers don't think about
        // priority, so we make the latency-sensitive choice for them.
        assert_eq!(Priority::default(), Priority::Interactive);
    }
}
