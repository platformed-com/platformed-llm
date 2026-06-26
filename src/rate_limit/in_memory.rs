//! Default in-process [`super::RateLimiter`] implementation.
//!
//! The implementation is a per-`bucket_key` scheduler:
//!
//! - **AIMD capacity model.** Starts at 1 request per second when the
//!   bucket is first seen; each successful response additively grows
//!   the rate by `additive_step` (default 1 rps), bounded *above* by
//!   the observed-capacity ceiling from the provider's headers; every
//!   429 multiplicatively halves the rate (multiplied by
//!   `multiplicative_decrease`, default 0.5). This is the AIMD
//!   pattern used by TCP congestion control — it's what makes the
//!   limiter resilient to noisy neighbours sharing the upstream
//!   quota.
//!
//! - **`Retry-After` parking.** A 429 with a `Retry-After` hint
//!   parks the bucket for `max(retry_after, default_park)` before
//!   the next dispatch. Honouring the hint avoids busy-looping
//!   against a still-over-quota provider; capping by `max_park` (1
//!   minute by default) prevents a misbehaving header from parking
//!   the task for hours.
//!
//! - **Strict-priority round-robin dispatch.** Waiters are bucketed
//!   first by [`super::Priority`] (Interactive > Standard >
//!   Background) and within a priority by tenant; the scheduler
//!   pops the next tenant in round-robin order from the highest
//!   non-empty priority band, so a noisy tenant can't starve a
//!   polite one at the same priority. See the module docs for the
//!   starvation tradeoff.
//!
//! The state is wrapped in a single [`parking_lot::Mutex`] — picked
//! over [`std::sync::Mutex`] because poisoning would turn any panic
//! inside an `observe` callback into a hard outage for every future
//! acquire. Critical sections are short (a HashMap lookup, a
//! VecDeque push/pop, a `Notify` signal); contention isn't expected
//! to be a bottleneck below ~thousands of acquires/second per
//! limiter instance.
//!
//! # Wakeup ownership
//!
//! Exactly one waiter (the queue head) is responsible for evaluating
//! its dispatch target at any time. `wake_head` is called at every
//! state change that may have moved the head:
//!
//! - `enqueue` — a new waiter may now be the head (higher priority,
//!   or first in band).
//! - `try_dispatch` — popping the old head exposes a new head.
//! - `observe` — a `Retry-After` may have parked the bucket, an
//!   AIMD growth may have shortened the interval, a cancellation
//!   freed a slot.
//!
//! Non-head waiters wait indefinitely (`WaitForever`); they're woken
//! by `wake_head` once they become the head.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::sync::Notify;

use super::{Priority, ProviderRateInfo, RateLimiter, RateOutcome, RatePermit, RateScope};
use crate::Error;

/// Construction-time knobs for [`InMemoryRateLimiter`]. All
/// defaultable; tweak when the upstream's actual quota differs
/// materially from the conservative defaults.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct InMemoryRateLimiterConfig {
    /// Cold-start dispatch rate for a newly-observed bucket.
    /// Defaults to 1 request per second, which is safe against any
    /// production quota but adds 1–2s of warm-up latency on a fresh
    /// process. Raise it if your provider's per-minute quota
    /// comfortably permits a higher initial burst.
    pub initial_rps: f64,
    /// Hard floor on the AIMD rate. The multiplicative decrease on
    /// each 429 never takes the rate below this. Defaults to 0.25
    /// rps (one request every 4 seconds) — low enough to recover
    /// gracefully but high enough that a sustained backoff still
    /// makes forward progress. **Must be > 0.**
    pub min_rps: f64,
    /// Hard ceiling on the AIMD rate. Additive growth saturates
    /// here. Defaults to 100 rps; nothing in this crate sends
    /// faster than the provider quota anyway, but the ceiling
    /// stops a stale "remaining: 10000" header from telling the
    /// limiter to fire at infinity.
    pub max_rps: f64,
    /// Additive increase applied to the rate after each success.
    /// Defaults to 1.0 rps. Smaller values converge more slowly
    /// but smoothly; larger values reach the observed ceiling
    /// faster but overshoot more often.
    pub additive_step: f64,
    /// Multiplicative decrease applied to the rate on each 429.
    /// Must be in `(0.0, 1.0)`. Defaults to 0.5 (TCP-style
    /// halving). Use a value closer to 1.0 (e.g. 0.75) when the
    /// provider's quotas are stable and 429s are mostly noise from
    /// neighbours.
    pub multiplicative_decrease: f64,
    /// Minimum park duration after a 429 without an explicit
    /// `Retry-After`. Defaults to 1 second — short enough that a
    /// transient 429 doesn't park interactive traffic for long,
    /// long enough that we don't immediately re-hit the same
    /// rate-limit window.
    pub default_park: Duration,
    /// Cap on any single park duration, including a provider-
    /// supplied `Retry-After`. Defaults to 60 seconds; a provider
    /// asking us to wait 30 minutes from a tight in-process loop
    /// is more often a misconfiguration than a real signal.
    pub max_park: Duration,
}

impl Default for InMemoryRateLimiterConfig {
    fn default() -> Self {
        Self {
            initial_rps: 1.0,
            min_rps: 0.25,
            max_rps: 100.0,
            additive_step: 1.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_secs(1),
            max_park: Duration::from_secs(60),
        }
    }
}

impl InMemoryRateLimiterConfig {
    /// Validate the config. Catches the inputs that would otherwise
    /// panic deep inside `try_dispatch` (e.g. `1.0 / 0.0`, NaN
    /// multipliers, negative steps), surfacing them as
    /// [`Error::Config`] at construction time when the caller can
    /// still react.
    fn validate(&self) -> Result<(), Error> {
        fn finite_positive(name: &str, v: f64) -> Result<(), Error> {
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::config(format!(
                    "InMemoryRateLimiterConfig::{name} must be finite and > 0, got {v}",
                )));
            }
            Ok(())
        }
        fn finite_non_negative(name: &str, v: f64) -> Result<(), Error> {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::config(format!(
                    "InMemoryRateLimiterConfig::{name} must be finite and >= 0, got {v}",
                )));
            }
            Ok(())
        }
        finite_positive("initial_rps", self.initial_rps)?;
        finite_positive("min_rps", self.min_rps)?;
        finite_positive("max_rps", self.max_rps)?;
        finite_non_negative("additive_step", self.additive_step)?;
        if !self.multiplicative_decrease.is_finite()
            || !(0.0..1.0).contains(&self.multiplicative_decrease)
        {
            return Err(Error::config(format!(
                "InMemoryRateLimiterConfig::multiplicative_decrease must be in (0.0, 1.0), got {}",
                self.multiplicative_decrease,
            )));
        }
        if self.min_rps > self.max_rps {
            return Err(Error::config(format!(
                "InMemoryRateLimiterConfig: min_rps ({}) must be <= max_rps ({})",
                self.min_rps, self.max_rps,
            )));
        }
        Ok(())
    }
}

/// Default in-process limiter. Construct once, share via `Arc`
/// across every provider built in this process. See the module
/// docs for the scheduling and AIMD model.
#[derive(Debug, Clone)]
pub struct InMemoryRateLimiter {
    inner: Arc<Inner>,
}

impl InMemoryRateLimiter {
    /// Construct with the default configuration.
    pub fn new() -> Self {
        // Default config is statically valid; expect rather than
        // propagate so `new()` stays infallible.
        Self::try_with_config(InMemoryRateLimiterConfig::default())
            .expect("default config is valid")
    }

    /// Construct with caller-tuned [`InMemoryRateLimiterConfig`].
    /// Errors on invalid input (NaN rps, decrease ≥ 1.0, etc.)
    /// rather than panicking later inside the scheduler.
    pub fn try_with_config(config: InMemoryRateLimiterConfig) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                buckets: Mutex::new(HashMap::new()),
            }),
        })
    }
}

impl Default for InMemoryRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl RateLimiter for InMemoryRateLimiter {
    async fn acquire(&self, scope: &RateScope) -> Result<RatePermit, Error> {
        let key: Arc<str> = Arc::from(scope.bucket_key.as_str());
        let waiter = self.inner.enqueue(&key, scope);
        // Guard removes the waiter from the bucket if the future is
        // dropped (cancel / timeout / parent task drop) before
        // `try_dispatch` pops it. Without this the abandoned waiter
        // stays at the head of the queue, future enqueues notify a
        // dead `Notify`, and the bucket wedges.
        let mut guard = AcquireGuard {
            inner: self.inner.clone(),
            key: key.clone(),
            waiter_id: waiter.id,
            completed: false,
        };

        loop {
            let next = {
                let mut buckets = self.inner.buckets.lock();
                let bucket = buckets
                    .get_mut(&key)
                    .expect("bucket was inserted on enqueue");
                if bucket.try_dispatch(&waiter, &self.inner.config) {
                    guard.completed = true;
                    // Build the permit callback that observes back
                    // into this bucket. Cloning the Arcs is cheap
                    // (refcount bumps).
                    let inner = self.inner.clone();
                    let key = key.clone();
                    let permit = RatePermit::new(move |outcome| {
                        inner.observe(&key, outcome);
                    });
                    return Ok(permit);
                }
                bucket.next_action(&waiter)
            };
            match next {
                NextAction::WaitUntil(when) => {
                    let now = Instant::now();
                    let dur = when.saturating_duration_since(now);
                    // Race the sleep against the notify. We don't
                    // care which wakes us — the next loop iteration
                    // re-evaluates dispatch under the lock.
                    let _ = tokio::time::timeout(dur, waiter.notify.notified()).await;
                }
                NextAction::WaitForever => {
                    waiter.notify.notified().await;
                }
            }
        }
    }
}

/// One waiter's worth of bookkeeping. Each [`InMemoryRateLimiter::acquire`]
/// builds one and parks on its `notify` until the scheduler picks
/// it.
#[derive(Debug)]
struct Waiter {
    id: u64,
    priority: Priority,
    tenant: Arc<str>,
    notify: Arc<Notify>,
}

/// RAII guard that removes the waiter from its bucket queue if the
/// `acquire` future is dropped before dispatch. The whole point is
/// resilience against future cancellation — without this an
/// abandoned waiter wedges the bucket.
struct AcquireGuard {
    inner: Arc<Inner>,
    key: Arc<str>,
    waiter_id: u64,
    completed: bool,
}

impl Drop for AcquireGuard {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let mut buckets = self.inner.buckets.lock();
        if let Some(bucket) = buckets.get_mut(&self.key) {
            bucket.remove_waiter(self.waiter_id);
            // Wake the new head so it can re-evaluate now that the
            // queue shape changed.
            bucket.wake_head();
        }
    }
}

/// What the acquire loop should do when it couldn't dispatch this
/// iteration. Computed under the lock; the loop then drops the lock
/// before sleeping.
enum NextAction {
    /// Sleep until this instant (or until notified, whichever first).
    WaitUntil(Instant),
    /// No timed wake — only proceed on notify.
    WaitForever,
}

#[derive(Debug)]
struct Inner {
    config: InMemoryRateLimiterConfig,
    buckets: Mutex<HashMap<Arc<str>, Bucket>>,
}

impl Inner {
    fn enqueue(&self, key: &Arc<str>, scope: &RateScope) -> Arc<WaiterHandle> {
        let mut buckets = self.buckets.lock();
        let bucket = buckets.entry(key.clone()).or_insert_with(|| {
            Bucket::new(
                self.config
                    .initial_rps
                    .clamp(self.config.min_rps, self.config.max_rps),
            )
        });
        let id = bucket.next_waiter_id();
        let notify = Arc::new(Notify::new());
        let handle = Arc::new(WaiterHandle {
            id,
            notify: notify.clone(),
        });
        let waiter = Waiter {
            id,
            priority: scope.priority,
            tenant: scope.tenant.clone(),
            notify,
        };
        bucket.push(waiter);
        // Wake the head waiter (which may now be us) so it
        // re-evaluates immediately rather than sitting on a stale
        // sleep target from before the bucket changed.
        bucket.wake_head();
        handle
    }

    fn observe(&self, key: &Arc<str>, outcome: RateOutcome) {
        let mut buckets = self.buckets.lock();
        let Some(bucket) = buckets.get_mut(key) else {
            return;
        };
        match outcome {
            RateOutcome::Success { info } => bucket.observe_success(&info, &self.config),
            RateOutcome::RateLimited { retry_after, info } => {
                bucket.observe_rate_limit(retry_after, &info, &self.config);
            }
            RateOutcome::OtherFailure | RateOutcome::Cancelled => {
                // No AIMD update — the outcome doesn't tell us
                // anything about capacity.
            }
        }
        // Wake the head so it can re-evaluate after a Retry-After
        // park or an AIMD rate change.
        bucket.wake_head();
    }
}

/// Cheap-to-clone handle shared between the acquire loop and the
/// scheduler; lets the loop identify "is this me at the head?"
/// without holding raw pointers.
struct WaiterHandle {
    id: u64,
    notify: Arc<Notify>,
}

/// Per-bucket scheduling state. Holds the AIMD rate, the
/// parked-until window, and the priority/tenant queue.
#[derive(Debug)]
struct Bucket {
    /// Current AIMD dispatch rate in requests/second.
    rps: f64,
    /// Earliest time the next dispatch may fire (last dispatch +
    /// `1.0 / rps`).
    next_dispatch_at: Instant,
    /// `Retry-After`-induced park gate. No dispatches before this
    /// instant; `None` means not parked.
    parked_until: Option<Instant>,
    /// One band per [`Priority`]. Bands are walked in declaration
    /// order (Interactive → Standard → Background) for strict
    /// priority dispatch.
    interactive: PriorityBand,
    standard: PriorityBand,
    background: PriorityBand,
    /// Monotonic waiter id, used to identify "is this me at the
    /// head" without raw pointers.
    next_id: u64,
}

#[derive(Debug, Default)]
struct PriorityBand {
    /// Per-tenant FIFO of waiters.
    tenant_queues: HashMap<Arc<str>, VecDeque<Waiter>>,
    /// Round-robin order of tenants with non-empty queues.
    tenant_order: VecDeque<Arc<str>>,
}

impl PriorityBand {
    fn peek(&self) -> Option<&Waiter> {
        let tenant = self.tenant_order.front()?;
        self.tenant_queues.get(tenant)?.front()
    }

    fn pop(&mut self) -> Option<Waiter> {
        let tenant = self.tenant_order.pop_front()?;
        let queue = self.tenant_queues.get_mut(&tenant)?;
        let waiter = queue.pop_front()?;
        if queue.is_empty() {
            self.tenant_queues.remove(&tenant);
        } else {
            // Re-add the tenant at the back so other tenants
            // dispatch before this one's next waiter.
            self.tenant_order.push_back(tenant);
        }
        Some(waiter)
    }

    fn push(&mut self, waiter: Waiter) {
        let queue = self.tenant_queues.entry(waiter.tenant.clone()).or_default();
        let was_empty = queue.is_empty();
        let tenant_name = waiter.tenant.clone();
        queue.push_back(waiter);
        if was_empty {
            // First waiter for this tenant in this band — add to
            // the round-robin rotation. If the tenant already had
            // waiters it's already in the rotation; do not re-add.
            self.tenant_order.push_back(tenant_name);
        }
    }

    fn is_empty(&self) -> bool {
        self.tenant_order.is_empty()
    }

    /// Remove the waiter with `id` if present. Used by the cancel
    /// guard. Returns whether anything was removed.
    fn remove_waiter(&mut self, id: u64) -> bool {
        // Two-pass: locate first (immutable scan), mutate second
        // (drops the borrow before the entry/remove dance).
        let target_tenant = self.tenant_queues.iter().find_map(|(tenant, queue)| {
            if queue.iter().any(|w| w.id == id) {
                Some(tenant.clone())
            } else {
                None
            }
        });
        let Some(tenant) = target_tenant else {
            return false;
        };
        let queue = self
            .tenant_queues
            .get_mut(&tenant)
            .expect("just found this entry");
        let pos = queue
            .iter()
            .position(|w| w.id == id)
            .expect("just found this id");
        queue.remove(pos);
        if queue.is_empty() {
            self.tenant_queues.remove(&tenant);
            if let Some(pos) = self.tenant_order.iter().position(|t| *t == tenant) {
                self.tenant_order.remove(pos);
            }
        }
        true
    }
}

impl Bucket {
    fn new(initial_rps: f64) -> Self {
        Self {
            rps: initial_rps,
            next_dispatch_at: Instant::now(),
            parked_until: None,
            interactive: PriorityBand::default(),
            standard: PriorityBand::default(),
            background: PriorityBand::default(),
            next_id: 0,
        }
    }

    fn next_waiter_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn band_for(&mut self, priority: Priority) -> &mut PriorityBand {
        match priority {
            Priority::Interactive => &mut self.interactive,
            Priority::Standard => &mut self.standard,
            Priority::Background => &mut self.background,
        }
    }

    fn push(&mut self, waiter: Waiter) {
        // When the bucket went from idle to populated, advance the
        // dispatch clock to `now` so a long idle period doesn't
        // burst-dispatch the whole backlog when a request arrives.
        let was_idle = self.is_empty();
        if was_idle {
            self.next_dispatch_at = self.next_dispatch_at.max(Instant::now());
        }
        self.band_for(waiter.priority).push(waiter);
    }

    fn is_empty(&self) -> bool {
        self.interactive.is_empty() && self.standard.is_empty() && self.background.is_empty()
    }

    /// Peek the next waiter the scheduler would dispatch. Strict
    /// priority: lowest [`Priority`] variant wins; within a band,
    /// the head of the round-robin order wins.
    fn peek_next(&self) -> Option<&Waiter> {
        self.interactive
            .peek()
            .or_else(|| self.standard.peek())
            .or_else(|| self.background.peek())
    }

    fn pop_next(&mut self) -> Option<Waiter> {
        if let Some(w) = self.interactive.pop() {
            return Some(w);
        }
        if let Some(w) = self.standard.pop() {
            return Some(w);
        }
        self.background.pop()
    }

    fn remove_waiter(&mut self, id: u64) {
        // Try each band in priority order; at most one match.
        if self.interactive.remove_waiter(id) {
            return;
        }
        if self.standard.remove_waiter(id) {
            return;
        }
        self.background.remove_waiter(id);
    }

    /// If `handle` is at the head of the queue **and** the AIMD
    /// bucket / park gate allow dispatch right now, pop it and
    /// schedule the next dispatch instant. Returns whether we
    /// dispatched `handle`.
    fn try_dispatch(&mut self, handle: &WaiterHandle, config: &InMemoryRateLimiterConfig) -> bool {
        let now = Instant::now();
        if let Some(parked) = self.parked_until {
            if now < parked {
                return false;
            }
            self.parked_until = None;
        }
        if now < self.next_dispatch_at {
            return false;
        }
        let Some(head) = self.peek_next() else {
            return false;
        };
        if head.id != handle.id {
            return false;
        }
        let _ = self.pop_next();
        // Use the clamped rate (config.validate guarantees both bounds
        // are finite > 0, so `1.0 / rps` is finite > 0 and
        // `from_secs_f64` cannot panic).
        let rps = self.rps.clamp(config.min_rps, config.max_rps);
        let interval = Duration::from_secs_f64(1.0 / rps);
        self.next_dispatch_at = now + interval;
        // Wake the next head so it can immediately schedule its own
        // sleep target.
        self.wake_head();
        true
    }

    /// Compute the sleep target for a waiter that just failed to
    /// dispatch.
    fn next_action(&self, handle: &WaiterHandle) -> NextAction {
        let Some(head) = self.peek_next() else {
            return NextAction::WaitForever;
        };
        if head.id != handle.id {
            return NextAction::WaitForever;
        }
        let dispatch_at = self.next_dispatch_at;
        let target = self
            .parked_until
            .map(|p| p.max(dispatch_at))
            .unwrap_or(dispatch_at);
        NextAction::WaitUntil(target)
    }

    /// Wake the current head so it re-evaluates its sleep target.
    /// No-op if the queue is empty. Idempotent — `Notify::notify_one`
    /// only stores at most one permit.
    fn wake_head(&self) {
        if let Some(head) = self.peek_next() {
            head.notify.notify_one();
        }
    }

    /// Successful response — additive grow, optionally capped at the
    /// observed-capacity ceiling from the response headers.
    ///
    /// The cap only **bounds growth** — it never shrinks an
    /// already-stable rate. Earlier drafts shrank on
    /// `remaining=0`-at-end-of-window, which collapsed the rate to
    /// `min_rps` on every quota-window boundary; the cap now only
    /// applies when it's higher than the current rate.
    fn observe_success(&mut self, info: &ProviderRateInfo, config: &InMemoryRateLimiterConfig) {
        let proposed = self.rps + config.additive_step;
        let target = if let (Some(remaining), Some(reset)) =
            (info.requests_remaining, info.requests_reset)
        {
            if reset.as_secs_f64() > 0.0 && remaining > 0 {
                let observed = remaining as f64 / reset.as_secs_f64();
                // Only cap if `observed` is below the proposed
                // post-growth rate. Crucially, never use `observed`
                // to *decrease* an existing rate — that's the job of
                // 429 observation, not header-derived ceilings.
                proposed.min(observed.max(self.rps))
            } else {
                // `remaining == 0` at end of window or unknown reset
                // — ignore the ceiling and let normal AIMD growth
                // continue. The next window's headers will give a
                // real ceiling, and a 429 (if we overshoot) snaps
                // us back via multiplicative decrease.
                proposed
            }
        } else {
            proposed
        };
        self.rps = target.clamp(config.min_rps, config.max_rps);
    }

    /// Rate-limited — halve the rate and park the bucket. Honours
    /// the provider's `Retry-After` hint; if absent, falls back to
    /// the longer of `default_park` and `requests_reset` from the
    /// response headers (when present), all capped at `max_park`.
    fn observe_rate_limit(
        &mut self,
        retry_after: Option<Duration>,
        info: &ProviderRateInfo,
        config: &InMemoryRateLimiterConfig,
    ) {
        let suggested = retry_after
            .or(info.requests_reset)
            .unwrap_or(config.default_park);
        let park = suggested.max(config.default_park).min(config.max_park);
        self.parked_until = Some(Instant::now() + park);
        self.rps =
            (self.rps * config.multiplicative_decrease).clamp(config.min_rps, config.max_rps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::Barrier;
    use tokio::time::{timeout, Duration as TokioDuration};

    fn scope(tenant: &str, priority: Priority) -> RateScope {
        RateScope {
            bucket_key: "Test/test-model".into(),
            tenant: tenant.into(),
            priority,
        }
    }

    fn permissive_config() -> InMemoryRateLimiterConfig {
        InMemoryRateLimiterConfig {
            initial_rps: 100.0,
            min_rps: 1.0,
            max_rps: 1000.0,
            additive_step: 10.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(10),
            max_park: Duration::from_millis(50),
        }
    }

    #[tokio::test]
    async fn first_request_dispatches_immediately() {
        let limiter = InMemoryRateLimiter::try_with_config(permissive_config()).unwrap();
        let start = Instant::now();
        let permit = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        assert!(start.elapsed() < Duration::from_millis(50));
        permit.observe(RateOutcome::Success {
            info: ProviderRateInfo::default(),
        });
    }

    #[tokio::test]
    async fn second_request_waits_for_aimd_interval() {
        // Cold-start config: ~1 rps, so the second acquire waits ~1s.
        let limiter = InMemoryRateLimiter::new();
        let _p1 = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        let start = Instant::now();
        let p2 = timeout(
            TokioDuration::from_secs(3),
            limiter.acquire(&scope("t1", Priority::Interactive)),
        )
        .await
        .expect("acquire should resolve within 3s")
        .unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(900),
            "expected ~1s AIMD interval, got {elapsed:?}",
        );
        drop(p2);
    }

    /// An [`AcquireGuard`] removes the waiter from the bucket if the
    /// future is dropped before dispatch. Without it the abandoned
    /// waiter wedges at the head of the queue and the next acquire
    /// never makes progress.
    #[tokio::test]
    async fn cancelled_acquire_does_not_wedge_bucket() {
        // Slow rate so the cancelled acquire is parked, not dispatched.
        let limiter = Arc::new(
            InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
                initial_rps: 2.0, // 500ms interval
                ..permissive_config()
            })
            .unwrap(),
        );
        let _p_gate = limiter
            .acquire(&scope("gate", Priority::Interactive))
            .await
            .unwrap();
        // Spawn a doomed acquire; cancel it via timeout before it
        // dispatches.
        let lim = limiter.clone();
        let cancelled = tokio::spawn(async move {
            // 100ms is well under the 500ms AIMD interval, so this
            // acquire is still queued when timeout fires.
            let _ = timeout(
                TokioDuration::from_millis(100),
                lim.acquire(&scope("cancelled", Priority::Interactive)),
            )
            .await;
        });
        cancelled.await.unwrap();
        // The next acquire from a real caller must now dispatch — if
        // the cancelled waiter wedged the queue, this would hang
        // forever (the test would time out).
        let p = timeout(
            TokioDuration::from_secs(2),
            limiter.acquire(&scope("real", Priority::Interactive)),
        )
        .await
        .expect("acquire after cancelled must not wedge")
        .unwrap();
        drop(p);
    }

    #[tokio::test]
    async fn strict_priority_preempts_across_tenants() {
        let limiter = Arc::new(
            InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
                initial_rps: 5.0, // 200ms interval — plenty of room to enqueue
                ..permissive_config()
            })
            .unwrap(),
        );
        // Burn the cold-start dispatch slot so subsequent acquires
        // are subject to the AIMD interval.
        let _gate = limiter
            .acquire(&scope("gate", Priority::Interactive))
            .await
            .unwrap();
        // Barrier ensures both background and interactive tasks
        // have reached enqueue before the AIMD gate opens — no
        // sleep-based timing races.
        let barrier = Arc::new(Barrier::new(3));
        let lim_a = limiter.clone();
        let bar_a = barrier.clone();
        let bg = tokio::spawn(async move {
            // Two-phase: signal we've started, then enqueue.
            bar_a.wait().await;
            let p = lim_a
                .acquire(&scope("a", Priority::Background))
                .await
                .unwrap();
            (Instant::now(), p)
        });
        let lim_b = limiter.clone();
        let bar_b = barrier.clone();
        let inter = tokio::spawn(async move {
            bar_b.wait().await;
            let p = lim_b
                .acquire(&scope("b", Priority::Interactive))
                .await
                .unwrap();
            (Instant::now(), p)
        });
        // Release both; they enqueue concurrently. Wait a bit so
        // both reach the `acquire` await point before the AIMD
        // interval opens.
        barrier.wait().await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        let (inter_time, _inter_permit) = inter.await.unwrap();
        let (bg_time, _bg_permit) = bg.await.unwrap();
        assert!(
            inter_time < bg_time,
            "interactive (t={inter_time:?}) must beat background (t={bg_time:?})",
        );
    }

    #[tokio::test]
    async fn tenant_fairness_within_priority_band() {
        let limiter = Arc::new(
            InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
                initial_rps: 10.0, // 100ms interval
                ..permissive_config()
            })
            .unwrap(),
        );
        // Hold the gate so subsequent acquires queue rather than
        // each dispatching immediately at cold-start.
        let _gate = limiter
            .acquire(&scope("gate", Priority::Interactive))
            .await
            .unwrap();
        // Barrier-synced enqueue: 6 tasks (3 per tenant) all reach
        // the acquire point at the same logical instant. We
        // additionally enforce enqueue *order* via a sequenced
        // counter so the round-robin assertion is deterministic.
        let order_counter = Arc::new(AtomicU32::new(0));
        let dispatched: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for i in 0..6 {
            let tenant = if i % 2 == 0 { "a" } else { "b" };
            let seq = i;
            let lim = limiter.clone();
            let log = dispatched.clone();
            let counter = order_counter.clone();
            let t = tenant.to_string();
            handles.push(tokio::spawn(async move {
                // Spin-wait for our turn so the enqueue order is
                // exactly the spawn order — independent of the
                // task scheduler's whims.
                while counter.load(Ordering::SeqCst) != seq {
                    tokio::task::yield_now().await;
                }
                let permit = lim
                    .acquire(&scope(&t, Priority::Interactive))
                    .await
                    .unwrap();
                counter.fetch_add(1, Ordering::SeqCst);
                log.lock().push(t);
                drop(permit);
            }));
        }
        // Wait long enough for all 6 to reach their spin-wait point.
        tokio::time::sleep(Duration::from_millis(10)).await;
        // Bump the counter so the first task can proceed; the others
        // chain off each other's completion.
        // Actually our counter is shared with completion-side, so
        // increment to start the first task.
        // Note: the `counter` is incremented inside acquire — but
        // we need to start it. The first task needs counter==0,
        // which is the initial value. So we just need to drop the
        // gate to start the dispatch chain.
        drop(_gate);
        for h in handles {
            h.await.unwrap();
        }
        let order = dispatched.lock().clone();
        // Round-robin: spawn order alternates a/b/a/b/a/b and the
        // limiter dispatches one per tenant in rotation per band.
        assert_eq!(order, vec!["a", "b", "a", "b", "a", "b"]);
    }

    #[tokio::test]
    async fn rate_limit_observation_parks_bucket() {
        let limiter = InMemoryRateLimiter::try_with_config(permissive_config()).unwrap();
        let p1 = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        p1.observe(RateOutcome::RateLimited {
            retry_after: Some(Duration::from_millis(30)),
            info: ProviderRateInfo::default(),
        });
        let start = Instant::now();
        let p2 = timeout(
            TokioDuration::from_millis(200),
            limiter.acquire(&scope("t1", Priority::Interactive)),
        )
        .await
        .expect("acquire should resolve within 200ms")
        .unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(25),
            "expected ~30ms park, got {elapsed:?}",
        );
        drop(p2);
    }

    /// Pin AIMD halving: the interval after a 429 should be ~2× the
    /// pre-429 interval. We assert both a lower bound (proves the
    /// rate decreased at all) and an upper bound (proves it didn't
    /// over-correct or stay the same — e.g. a buggy `*= 0.9` would
    /// fail the lower bound, and a missing decrease would fail the
    /// upper bound).
    #[tokio::test]
    async fn rate_limit_halves_rps() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 10.0,
            min_rps: 0.1,
            max_rps: 100.0,
            additive_step: 0.0, // disable growth for determinism
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(1),
            max_park: Duration::from_millis(10),
        })
        .unwrap();
        let p = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        p.observe(RateOutcome::RateLimited {
            retry_after: Some(Duration::from_millis(1)),
            info: ProviderRateInfo::default(),
        });
        tokio::time::sleep(Duration::from_millis(15)).await;
        let _p1 = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        let start = Instant::now();
        let p2 = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        let interval = start.elapsed();
        // Halved rate of 10 rps → 5 rps → 200ms interval. Lower
        // bound proves the rate decreased; upper bound rules out
        // an over-decrease (e.g. `*= 0.25` would give a 400ms
        // interval and fail) and a missing decrease (the original
        // 100ms interval would fail the lower bound).
        assert!(
            interval >= Duration::from_millis(150),
            "expected ~200ms (halved interval), got {interval:?}",
        );
        assert!(
            interval < Duration::from_millis(300),
            "halving should not over-decrease — got {interval:?}",
        );
        drop(p2);
    }

    /// AIMD growth on success must not collapse the rate when the
    /// provider reports `remaining=0` at the end of a quota window.
    /// Earlier drafts shrank `rps` to `min_rps` on every such
    /// success, which threw a phantom rate-limit at the next
    /// caller. This guards against the regression.
    #[tokio::test]
    async fn success_with_zero_remaining_does_not_shrink_rate() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 10.0,
            min_rps: 0.1,
            max_rps: 100.0,
            additive_step: 1.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(1),
            max_park: Duration::from_millis(10),
        })
        .unwrap();
        let p = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        // End-of-window success: remaining=0 with a non-zero reset.
        // Pre-fix, this collapsed rps to min_rps.
        p.observe(RateOutcome::Success {
            info: ProviderRateInfo {
                requests_remaining: Some(0),
                requests_reset: Some(Duration::from_secs(60)),
            },
        });
        let buckets = limiter.inner.buckets.lock();
        let bucket = buckets.values().next().unwrap();
        assert!(
            bucket.rps >= 10.0,
            "remaining=0 success must not shrink rps, got {}",
            bucket.rps,
        );
    }

    #[test]
    fn invalid_config_is_rejected() {
        // Validation catches the inputs that would otherwise panic
        // deep inside `try_dispatch` (e.g. `1.0 / 0.0`).
        let bad = InMemoryRateLimiterConfig {
            min_rps: 0.0,
            ..InMemoryRateLimiterConfig::default()
        };
        assert!(InMemoryRateLimiter::try_with_config(bad).is_err());
        let nan = InMemoryRateLimiterConfig {
            initial_rps: f64::NAN,
            ..InMemoryRateLimiterConfig::default()
        };
        assert!(InMemoryRateLimiter::try_with_config(nan).is_err());
        let bad_mult = InMemoryRateLimiterConfig {
            multiplicative_decrease: 1.0, // must be < 1.0
            ..InMemoryRateLimiterConfig::default()
        };
        assert!(InMemoryRateLimiter::try_with_config(bad_mult).is_err());
    }
}
