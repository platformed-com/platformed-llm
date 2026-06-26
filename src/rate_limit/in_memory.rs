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
use uuid::Uuid;

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
                #[cfg(test)]
                poison_next_observe: std::sync::atomic::AtomicBool::new(false),
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
    tenant: Uuid,
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
    /// Test-only escape hatch: when set, the next `observe()` call
    /// panics inside the catch_unwind block so we can verify
    /// `wake_head()` still fires and the next acquire isn't wedged.
    #[cfg(test)]
    poison_next_observe: std::sync::atomic::AtomicBool,
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
            tenant: scope.tenant,
            notify,
        };
        bucket.push(waiter);
        // Wake the head waiter (which may now be us) so it
        // re-evaluates immediately rather than sitting on a stale
        // sleep target from before the bucket changed.
        bucket.wake_head();
        handle
    }

    fn observe(self: &Arc<Self>, key: &Arc<str>, outcome: RateOutcome) {
        // RAII guard ensures `wake_head` fires on the way out even
        // if the AIMD math panics — losing a wakeup would let the
        // next head waiter sleep on `WaitForever` and wedge the
        // bucket. The guard re-locks the buckets map (the inner
        // `buckets` guard below releases first since it's declared
        // later → drops earlier); cost is one extra `parking_lot`
        // mutex acquisition per observe, ~25ns.
        //
        // Declared *before* the inner lock so its `Drop` runs
        // *after* the inner lock releases — both on normal exit and
        // during unwind.
        struct WakeOnExit {
            inner: Arc<Inner>,
            key: Arc<str>,
        }
        impl Drop for WakeOnExit {
            fn drop(&mut self) {
                let mut buckets = self.inner.buckets.lock();
                if let Some(bucket) = buckets.get_mut(&self.key) {
                    bucket.wake_head();
                }
            }
        }
        let _wake = WakeOnExit {
            inner: self.clone(),
            key: key.clone(),
        };

        let mut buckets = self.buckets.lock();
        let Some(bucket) = buckets.get_mut(key) else {
            return;
        };
        #[cfg(test)]
        if self
            .poison_next_observe
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            panic!("test-only: poison_next_observe triggered");
        }
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
        // `buckets` (the inner lock) drops here, then `_wake` drops
        // after it (reverse-declaration order) and fires wake_head.
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
    tenant_queues: HashMap<Uuid, VecDeque<Waiter>>,
    /// Round-robin order of tenants with non-empty queues.
    tenant_order: VecDeque<Uuid>,
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
        let tenant = waiter.tenant;
        let queue = self.tenant_queues.entry(tenant).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(waiter);
        if was_empty {
            // First waiter for this tenant in this band — add to
            // the round-robin rotation. If the tenant already had
            // waiters it's already in the rotation; do not re-add.
            self.tenant_order.push_back(tenant);
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
                Some(*tenant)
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
    use tokio::sync::Barrier;
    use tokio::time::{timeout, Duration as TokioDuration};

    /// Stable, **order-sensitive** per-name Uuid for tests. Lets
    /// tests keep using readable names while the limiter sees Uuids.
    ///
    /// Earlier this used a plain byte-sum which collided for any
    /// byte-permutation of a name (e.g. `"ab"` and `"ba"` mapped to
    /// the same Uuid), silently sharing bucket state if two scopes
    /// happened to be such anagrams. This formulation mixes the
    /// byte into a multiplicative running state (constants borrowed
    /// from `splitmix64`), so order matters.
    fn tenant_uuid(name: &str) -> Uuid {
        let mut hash: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C834;
        for b in name.bytes() {
            hash = hash.wrapping_mul(0x100_0000_0000_0000_0000_0000_0000_013B);
            hash ^= b as u128;
        }
        // Clear the top nibble and replace with `0x1` so the Uuid is
        // clearly synthetic (and never `Uuid::nil`) when printed.
        let cleared = hash & 0x0FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF;
        Uuid::from_u128(cleared | 0x1000_0000_0000_0000_0000_0000_0000_0000)
    }

    /// Sanity check that the test helper actually distinguishes
    /// byte-permuted names — the earlier order-insensitive version
    /// silently returned the same Uuid for `"ab"` and `"ba"`,
    /// fragmenting bucket isolation if any test happened to use
    /// such a pair.
    #[test]
    fn tenant_uuid_distinguishes_byte_permutations() {
        assert_ne!(tenant_uuid("ab"), tenant_uuid("ba"));
        assert_ne!(tenant_uuid("loud"), tenant_uuid("doul"));
        assert_eq!(tenant_uuid("vip"), tenant_uuid("vip"));
    }

    fn scope(tenant: &str, priority: Priority) -> RateScope {
        RateScope {
            bucket_key: "Test/test-model".into(),
            tenant: tenant_uuid(tenant),
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

    /// Direct unit test of [`PriorityBand`]'s round-robin pop order.
    ///
    /// An earlier version of this test used the async acquire path
    /// with a spin-wait counter that serialised the acquires — each
    /// task waited for the previous one's `acquire().await` to
    /// complete before its own enqueue. That meant only one waiter
    /// was ever queued at a time and the round-robin pop logic was
    /// never exercised: a FIFO `PriorityBand::pop` would have
    /// passed. This version drives the data structure directly so
    /// the invariant we care about — "alternating tenants A and B
    /// each get one dispatch per rotation, regardless of insertion
    /// order within the band" — is genuinely pinned.
    #[test]
    fn priority_band_round_robins_between_tenants() {
        use std::sync::Arc as StdArc;
        let tenant_a = Uuid::from_u128(0xa);
        let tenant_b = Uuid::from_u128(0xb);
        let mut band = PriorityBand::default();
        let make_waiter = |id: u64, tenant: Uuid| Waiter {
            id,
            priority: Priority::Interactive,
            tenant,
            notify: StdArc::new(Notify::new()),
        };
        // Push 3 each in alternating order: a, b, a, b, a, b.
        band.push(make_waiter(0, tenant_a));
        band.push(make_waiter(1, tenant_b));
        band.push(make_waiter(2, tenant_a));
        band.push(make_waiter(3, tenant_b));
        band.push(make_waiter(4, tenant_a));
        band.push(make_waiter(5, tenant_b));
        // Pop until empty; record the order of tenant ids.
        let mut popped = Vec::new();
        while let Some(w) = band.pop() {
            popped.push(w.tenant);
        }
        // Round-robin: each pop alternates tenant, never two of the
        // same tenant in a row while the other has waiters queued.
        assert_eq!(
            popped,
            vec![tenant_a, tenant_b, tenant_a, tenant_b, tenant_a, tenant_b],
        );
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

    // ====================================================================
    // Permit-safety tests
    //
    // The pipeline-wedge failure mode: an "lost" permit — a slot consumed
    // but the next head never woken — causes every subsequent acquire on
    // that bucket to hang. These tests assert the recovery paths.
    // ====================================================================

    /// A caller panic between `acquire` and `observe` must not wedge
    /// the bucket: the permit's `Drop` impl fires `Cancelled`, which
    /// in turn wakes the next head. Without that, the next acquire
    /// would hang.
    #[tokio::test]
    async fn caller_panic_between_acquire_and_observe_does_not_wedge() {
        let limiter = Arc::new(
            InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
                initial_rps: 5.0, // 200ms interval so we'd notice a wedge
                ..permissive_config()
            })
            .unwrap(),
        );
        let lim_a = limiter.clone();
        let task_a = tokio::spawn(async move {
            let _permit = lim_a
                .acquire(&scope("a", Priority::Interactive))
                .await
                .unwrap();
            // Hold the permit briefly to ensure it's dispatched first,
            // then panic. The `_permit` drops during unwind which fires
            // `Cancelled` on the limiter's wakeup path.
            panic!("simulated user code panic while holding a permit");
        });
        // Wait for the panicking task to finish before the next acquire.
        let result = task_a.await;
        assert!(result.is_err(), "task A should have panicked");

        let permit_b = timeout(
            TokioDuration::from_secs(3),
            limiter.acquire(&scope("b", Priority::Interactive)),
        )
        .await
        .expect("acquire B must not wedge after A's panic")
        .unwrap();
        drop(permit_b);
    }

    /// A permit dropping during an *outer* panic (the canonical
    /// double-panic risk) must not abort the process. `RatePermit::Drop`
    /// wraps the callback in `catch_unwind` for exactly this.
    #[test]
    fn permit_drop_inside_outer_panic_does_not_abort() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        let observed = Arc::new(AtomicU32::new(0));
        let observed_cb = observed.clone();
        // Construct a permit whose callback panics. If our Drop guard
        // is missing, the panic-from-panic aborts the process and this
        // test wouldn't even produce a failure — it'd crash the
        // runner. With the guard, `catch_unwind` swallows it and the
        // outer panic propagates cleanly.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _permit = RatePermit::new(move |_outcome| {
                observed_cb.fetch_add(1, Ordering::SeqCst);
                panic!("simulated panic from observe callback during drop");
            });
            panic!("outer panic that triggers the permit's Drop");
        }));
        // Sentinel: only set if execution actually proceeded past the
        // `catch_unwind` boundary. If a double-panic had aborted the
        // process, we'd never reach this line — the runner would just
        // exit. Asserting it explicitly distinguishes "caught cleanly"
        // from "process kept running but the assertion happened to be
        // skipped".
        let reached_after_unwind = AtomicBool::new(true);
        assert!(
            reached_after_unwind.load(Ordering::SeqCst),
            "execution must continue past catch_unwind — a double-panic abort would have killed the runner",
        );
        assert!(result.is_err(), "outer panic should still be observed");
        // The payload propagated out of catch_unwind must be the
        // *outer* panic, not the swallowed inner observe panic. If the
        // catch_unwind boundary were inverted (catching the outer,
        // letting the inner through) this assertion would fail.
        let payload = result.err().unwrap();
        let msg = payload
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic payload>");
        assert!(
            msg.contains("outer panic"),
            "propagated panic should be the outer one, got: {msg:?}",
        );
        assert_eq!(
            observed.load(Ordering::SeqCst),
            1,
            "callback should have run exactly once before its own panic was swallowed",
        );
    }

    /// Many simultaneous cancellations (acquire-then-drop-without-
    /// observe) must leave the bucket cleanly empty. This guards
    /// against the AcquireGuard / Drop interaction missing a
    /// waiter under load — including the harder case where multiple
    /// waiters share a tenant, so `remove_waiter` exercises the
    /// "queue not empty, leave tenant in `tenant_order`" branch.
    #[tokio::test]
    async fn concurrent_cancellations_leave_bucket_empty() {
        let limiter = Arc::new(
            InMemoryRateLimiter::try_with_config(
                // Slow rate so most acquires queue rather than dispatch.
                InMemoryRateLimiterConfig {
                    initial_rps: 2.0,
                    ..permissive_config()
                },
            )
            .unwrap(),
        );
        let mut tasks = Vec::new();
        // Two shared tenants ("a" / "b") across both priorities —
        // so each tenant ends up with 5 waiters in its
        // `interactive` band and 5 in its `background` band.
        // Earlier versions used `format!("t{i}")` giving every
        // task its own tenant, which never exercised the
        // multi-waiter-per-tenant-queue cancellation path.
        for i in 0..20 {
            let lim = limiter.clone();
            let priority = if i % 2 == 0 {
                Priority::Interactive
            } else {
                Priority::Background
            };
            let tenant_name = if i % 4 < 2 { "a" } else { "b" };
            tasks.push(tokio::spawn(async move {
                let _ = timeout(
                    TokioDuration::from_millis(50),
                    lim.acquire(&scope(tenant_name, priority)),
                )
                .await;
                // permit either dispatched (then drops here) or
                // cancelled by timeout
            }));
        }
        for t in tasks {
            let _ = t.await;
        }
        // After all cancellations / drops resolve, the bucket should
        // have no pending waiters. A new acquire must dispatch within
        // one AIMD interval — if a queued waiter leaked, we'd block
        // behind it indefinitely.
        let permit = timeout(
            TokioDuration::from_secs(3),
            limiter.acquire(&scope("clean", Priority::Interactive)),
        )
        .await
        .expect("a fresh acquire after 20 cancellations must not wedge")
        .unwrap();
        drop(permit);
        // Verify the bucket bookkeeping is actually empty.
        let buckets = limiter.inner.buckets.lock();
        let bucket = buckets.values().next().unwrap();
        assert!(
            bucket.is_empty(),
            "bucket should have no queued waiters after cancellations resolve",
        );
    }

    // ====================================================================
    // AIMD coverage gaps
    // ====================================================================

    /// Successful responses additively grow `rps` toward the
    /// observed-capacity ceiling. A regression that disabled growth
    /// (e.g. `additive_step = 0`, or accidentally inverting the math)
    /// would fail this.
    #[tokio::test]
    async fn aimd_grows_rate_on_success() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 10.0,
            min_rps: 0.1,
            max_rps: 1000.0,
            additive_step: 5.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(1),
            max_park: Duration::from_millis(10),
        })
        .unwrap();
        for _ in 0..3 {
            let p = limiter
                .acquire(&scope("t1", Priority::Interactive))
                .await
                .unwrap();
            p.observe(RateOutcome::Success {
                info: ProviderRateInfo::default(),
            });
        }
        let buckets = limiter.inner.buckets.lock();
        let bucket = buckets.values().next().unwrap();
        // 3 successes * 5 rps each = +15. Initial 10 → ~25.
        // Allow a small slack for clamping at max.
        assert!(
            bucket.rps >= 24.0 && bucket.rps <= 25.0,
            "expected rps ≈ 25 after 3 successes, got {}",
            bucket.rps,
        );
    }

    /// Repeated 429s must not push `rps` below `min_rps`. A bug in
    /// the multiplicative-decrease clamp would let it underflow.
    ///
    /// Drives the halvings via direct `Inner::observe` calls rather
    /// than the async acquire path because once `rps → min_rps`
    /// the dispatch interval grows to seconds, making the test slow
    /// for what's really just AIMD arithmetic.
    #[tokio::test]
    async fn aimd_floor_holds_under_repeated_429() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 4.0,
            min_rps: 1.0,
            max_rps: 100.0,
            additive_step: 0.0, // disable growth so test is deterministic
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(1),
            max_park: Duration::from_millis(5),
        })
        .unwrap();
        // Establish the bucket via a single acquire (so it exists
        // in the map), then drop the permit (Cancelled — no AIMD
        // change).
        let scope = scope("t1", Priority::Interactive);
        let key: Arc<str> = Arc::from(scope.bucket_key.as_str());
        drop(limiter.acquire(&scope).await.unwrap());
        // 5 halvings: 4 → 2 → 1 → 0.5 → 0.25 → 0.125 without the
        // floor. With `min_rps = 1.0` the floor must hold.
        for _ in 0..5 {
            limiter.inner.observe(
                &key,
                RateOutcome::RateLimited {
                    retry_after: Some(Duration::from_millis(1)),
                    info: ProviderRateInfo::default(),
                },
            );
        }
        let buckets = limiter.inner.buckets.lock();
        let bucket = buckets.values().next().unwrap();
        assert!(
            bucket.rps >= 1.0,
            "min_rps floor must hold after sustained 429s, got {}",
            bucket.rps,
        );
    }

    /// Repeated successes must not push `rps` above `max_rps`. A bug
    /// in the additive-growth clamp would let it overshoot.
    #[tokio::test]
    async fn aimd_ceiling_holds_under_repeated_success() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 5.0,
            min_rps: 0.1,
            max_rps: 10.0,
            additive_step: 50.0, // big step, would blow past 10 without clamp
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(1),
            max_park: Duration::from_millis(5),
        })
        .unwrap();
        for _ in 0..3 {
            let p = limiter
                .acquire(&scope("t1", Priority::Interactive))
                .await
                .unwrap();
            p.observe(RateOutcome::Success {
                info: ProviderRateInfo::default(),
            });
        }
        let buckets = limiter.inner.buckets.lock();
        let bucket = buckets.values().next().unwrap();
        assert!(
            bucket.rps <= 10.0,
            "max_rps ceiling must hold after sustained successes, got {}",
            bucket.rps,
        );
    }

    // ====================================================================
    // Park-handling tests
    // ====================================================================

    /// An extreme `Retry-After` hint must be capped at `max_park`.
    /// Without the cap, a misbehaving provider header could park the
    /// bucket for hours.
    #[tokio::test]
    async fn retry_after_capped_at_max_park() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 100.0,
            min_rps: 1.0,
            max_rps: 1000.0,
            additive_step: 10.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(10),
            max_park: Duration::from_millis(50),
        })
        .unwrap();
        let p = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        // Hostile retry-after: 1 hour. Must cap at max_park (50ms).
        p.observe(RateOutcome::RateLimited {
            retry_after: Some(Duration::from_secs(3600)),
            info: ProviderRateInfo::default(),
        });
        let start = Instant::now();
        let p2 = timeout(
            TokioDuration::from_millis(300),
            limiter.acquire(&scope("t1", Priority::Interactive)),
        )
        .await
        .expect("acquire must resolve within 300ms — max_park=50ms")
        .unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "park honoured max_park cap, but elapsed was {elapsed:?}",
        );
        drop(p2);
    }

    /// When the 429 carries no `Retry-After` but the response info
    /// includes `requests_reset`, that value drives the park instead
    /// of `default_park`.
    #[tokio::test]
    async fn requests_reset_used_as_park_fallback() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 100.0,
            min_rps: 1.0,
            max_rps: 1000.0,
            additive_step: 10.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(5),
            max_park: Duration::from_millis(200),
        })
        .unwrap();
        let p = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        // No retry-after; info.requests_reset says wait 80ms.
        p.observe(RateOutcome::RateLimited {
            retry_after: None,
            info: ProviderRateInfo {
                requests_remaining: Some(0),
                requests_reset: Some(Duration::from_millis(80)),
            },
        });
        let start = Instant::now();
        let p2 = timeout(
            TokioDuration::from_millis(300),
            limiter.acquire(&scope("t1", Priority::Interactive)),
        )
        .await
        .expect("acquire must resolve within 300ms")
        .unwrap();
        let elapsed = start.elapsed();
        // requests_reset is the floor (max(default_park, reset)); we
        // honoured 80ms, not 5ms default.
        assert!(
            elapsed >= Duration::from_millis(70),
            "expected ~80ms park from requests_reset, got {elapsed:?}",
        );
        drop(p2);
    }

    // ====================================================================
    // Bucket isolation
    // ====================================================================

    /// Two different `bucket_key`s must track independent AIMD state.
    /// A 429 on bucket A must not affect bucket B's rate.
    #[tokio::test]
    async fn aimd_state_is_per_bucket() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 10.0,
            min_rps: 0.1,
            max_rps: 100.0,
            additive_step: 0.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(1),
            max_park: Duration::from_millis(5),
        })
        .unwrap();
        let scope_a = RateScope {
            bucket_key: "ProviderA/model-x".into(),
            tenant: Uuid::nil(),
            priority: Priority::Interactive,
        };
        let scope_b = RateScope {
            bucket_key: "ProviderB/model-y".into(),
            tenant: Uuid::nil(),
            priority: Priority::Interactive,
        };
        let p_a = limiter.acquire(&scope_a).await.unwrap();
        p_a.observe(RateOutcome::RateLimited {
            retry_after: Some(Duration::from_millis(1)),
            info: ProviderRateInfo::default(),
        });
        // Bucket A's rps now halved to 5.0. Bucket B should be
        // untouched at 10.0.
        {
            let buckets = limiter.inner.buckets.lock();
            let a = buckets
                .get(&Arc::<str>::from("ProviderA/model-x"))
                .expect("A bucket exists");
            assert!(
                a.rps <= 5.5,
                "A's rps halved by 429: expected ~5.0, got {}",
                a.rps,
            );
        }
        // B's bucket isn't created yet (lazy); acquire on B and
        // verify it starts fresh at initial_rps.
        let p_b = limiter.acquire(&scope_b).await.unwrap();
        drop(p_b);
        let buckets = limiter.inner.buckets.lock();
        let b = buckets
            .get(&Arc::<str>::from("ProviderB/model-y"))
            .expect("B bucket now exists");
        assert!(
            b.rps >= 9.0 && b.rps <= 10.0,
            "B's rps should be untouched at initial_rps ≈ 10.0, got {}",
            b.rps,
        );
    }

    /// `Retry-After` parking on one bucket must not park another.
    #[tokio::test]
    async fn park_state_is_per_bucket() {
        let limiter = InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
            initial_rps: 100.0,
            min_rps: 1.0,
            max_rps: 1000.0,
            additive_step: 10.0,
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(10),
            max_park: Duration::from_millis(500),
        })
        .unwrap();
        let scope_a = RateScope {
            bucket_key: "ProviderA".into(),
            tenant: Uuid::nil(),
            priority: Priority::Interactive,
        };
        let scope_b = RateScope {
            bucket_key: "ProviderB".into(),
            tenant: Uuid::nil(),
            priority: Priority::Interactive,
        };
        let p_a = limiter.acquire(&scope_a).await.unwrap();
        // Park A for ≥300ms.
        p_a.observe(RateOutcome::RateLimited {
            retry_after: Some(Duration::from_millis(300)),
            info: ProviderRateInfo::default(),
        });
        // B must dispatch immediately even while A is parked.
        let start = Instant::now();
        let p_b = timeout(TokioDuration::from_millis(100), limiter.acquire(&scope_b))
            .await
            .expect("B must not be affected by A's park")
            .unwrap();
        assert!(start.elapsed() < Duration::from_millis(80));
        drop(p_b);
    }

    // ====================================================================
    // Edge cases
    // ====================================================================

    /// Observing on a permit whose bucket has been forgotten (e.g.
    /// limiter dropped, or a state-management bug removed the entry)
    /// must not panic. The current impl no-ops via `get_mut(key)?`.
    #[test]
    fn observe_for_nonexistent_bucket_is_noop() {
        let limiter = InMemoryRateLimiter::new();
        // Build a permit pointing at a bucket key that was never
        // created via `acquire`.
        let inner = limiter.inner.clone();
        let key: Arc<str> = Arc::from("never/acquired");
        let permit = RatePermit::new(move |outcome| {
            inner.observe(&key, outcome);
        });
        // Must not panic — Inner::observe returns early on miss.
        permit.observe(RateOutcome::Success {
            info: ProviderRateInfo::default(),
        });
    }

    /// A panic *inside* `Inner::observe`'s AIMD math must not prevent
    /// `wake_head()` from running. The `WakeOnExit` Drop guard fires
    /// during unwind (after the inner buckets-lock releases); the
    /// next waiter dispatches on schedule. Without the guard the
    /// next waiter would sleep indefinitely on `WaitForever` and
    /// the bucket would wedge.
    #[tokio::test]
    async fn observe_panic_does_not_wedge_next_waiter() {
        let limiter = Arc::new(
            InMemoryRateLimiter::try_with_config(
                // Slow enough that the second acquire actually waits on
                // the wake.
                InMemoryRateLimiterConfig {
                    initial_rps: 2.0, // 500ms interval
                    ..permissive_config()
                },
            )
            .unwrap(),
        );
        let p1 = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        // Arm the next observe call to panic inside its catch_unwind
        // block. The math is genuinely panic-free under validated
        // config; this hatch exists so we can verify the guard.
        limiter
            .inner
            .poison_next_observe
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // Spawn a follower BEFORE we observe — so it's queued and
        // will be the head once we pop.
        let lim_f = limiter.clone();
        let follower = tokio::spawn(async move {
            let p = lim_f
                .acquire(&scope("t1", Priority::Interactive))
                .await
                .unwrap();
            drop(p);
        });
        // Give the follower time to enqueue.
        tokio::time::sleep(Duration::from_millis(50)).await;
        // Observe (which panics inside catch_unwind, but wake_head
        // still runs on the way out). The panic propagates up here.
        let observe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p1.observe(RateOutcome::Success {
                info: ProviderRateInfo::default(),
            });
        }));
        assert!(
            observe_result.is_err(),
            "the injected panic should propagate from observe()",
        );
        // Despite the panic, the follower must be dispatched on its
        // normal AIMD schedule. If wake_head hadn't run, the follower
        // would be stuck at `WaitForever`.
        timeout(TokioDuration::from_secs(2), follower)
            .await
            .expect("follower must dispatch after panic — bucket would wedge otherwise")
            .unwrap();
    }

    /// Cancelling the actual *head* waiter (one that's waiting on
    /// the AIMD gate, not one further back in the queue) must
    /// promote a real *follower* that was already enqueued behind
    /// it. The existing `cancelled_acquire_does_not_wedge_bucket`
    /// test covers the non-head case (a permit is held so the
    /// cancelled acquire is queued behind it); this variant cancels
    /// the genuine head while a follower is parked at
    /// `WaitForever`, and asserts the follower is woken and
    /// dispatches. Without `AcquireGuard::drop → wake_head()` the
    /// follower would never re-evaluate its dispatch target.
    #[tokio::test]
    async fn head_cancel_during_wait_promotes_next_waiter() {
        let limiter = Arc::new(
            InMemoryRateLimiter::try_with_config(InMemoryRateLimiterConfig {
                initial_rps: 2.0, // 500ms interval — leaves plenty of cancel window
                ..permissive_config()
            })
            .unwrap(),
        );
        // Burn the cold-start slot so the next acquire actually
        // *waits* on the AIMD gate (rather than dispatching
        // immediately).
        let _p0 = limiter
            .acquire(&scope("warmup", Priority::Interactive))
            .await
            .unwrap();
        // Spawn the head waiter with a short timeout — it will be
        // cancelled while still actively waiting on the AIMD gate.
        let lim_head = limiter.clone();
        let head = tokio::spawn(async move {
            let _ = timeout(
                TokioDuration::from_millis(100),
                lim_head.acquire(&scope("head", Priority::Interactive)),
            )
            .await;
        });
        // Let the head reach the await point and claim head-of-queue
        // before the follower enqueues behind it.
        tokio::time::sleep(Duration::from_millis(20)).await;
        // Spawn a real follower that genuinely needs to be promoted —
        // it's parked at `WaitForever` until the head cancels and
        // `wake_head` re-fires.
        let lim_follower = limiter.clone();
        let follower = tokio::spawn(async move {
            let p = lim_follower
                .acquire(&scope("follower", Priority::Interactive))
                .await
                .unwrap();
            (Instant::now(), p)
        });
        // Let the follower reach the await point and enqueue behind
        // the head.
        tokio::time::sleep(Duration::from_millis(20)).await;
        // Wait for the head to be cancelled (its 100ms timeout
        // elapses). The `AcquireGuard` drop must remove the head and
        // fire `wake_head`, promoting the follower.
        head.await.unwrap();
        // The follower must now dispatch on the bucket's normal
        // schedule — if `wake_head` didn't re-fire after the head
        // dropped, it would stay at `WaitForever`.
        let (_when, p) = timeout(TokioDuration::from_secs(2), follower)
            .await
            .expect("follower must dispatch after head cancellation")
            .unwrap();
        drop(p);
    }
}
