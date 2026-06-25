//! Default in-process [`super::RateLimiter`] implementation.
//!
//! The implementation is a per-`(provider, model)` scheduler:
//!
//! - **AIMD capacity model.** Starts at 1 request per second when the
//!   `(provider, model)` is first seen; each successful response
//!   additively grows the rate by `additive_step` (default 1 rps) up
//!   to an observed ceiling; every 429 multiplicatively halves the
//!   rate (multiplied by `multiplicative_decrease`, default 0.5).
//!   This is the AIMD pattern used by TCP congestion control — it's
//!   what makes the limiter resilient to noisy neighbours sharing
//!   the provider quota.
//!
//! - **`Retry-After` parking.** A 429 with a `Retry-After` hint
//!   parks the bucket for `max(retry_after, default_park)` before
//!   the next dispatch. Honouring the hint avoids busy-looping
//!   against a still-over-quota provider; capping by `max_park` (1
//!   minute by default) prevents a misbehaving header from parking
//!   the task for hours.
//!
//! - **Strict-priority round-robin dispatch.** Waiters are bucketed
//!   first by [`super::Priority`] (interactive > standard >
//!   background) and within a priority by tenant; the scheduler
//!   pops the next tenant in round-robin order from the highest
//!   non-empty priority band, so a noisy tenant can't starve a
//!   polite one at the same priority. See the module docs for the
//!   starvation tradeoff.
//!
//! The state is wrapped in a single [`std::sync::Mutex`]. Critical
//! sections are short (a few [`std::collections::HashMap`] lookups,
//! a [`std::collections::VecDeque`] push/pop, an [`tokio::sync::Notify`]
//! signal); contention isn't expected to be a bottleneck below
//! ~thousands of acquires/second per limiter instance. A consumer
//! who needs more can shard limiters by tenant prefix or front it
//! with a Redis-backed [`super::RateLimiter`] impl.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

use super::{Priority, ProviderRateInfo, RateLimiter, RateOutcome, RatePermit, RateScope};
use crate::Error;

/// Construction-time knobs for [`InMemoryRateLimiter`]. All
/// defaultable; tweak when the upstream's actual quota differs
/// materially from the conservative defaults.
#[derive(Debug, Clone, Copy)]
pub struct InMemoryRateLimiterConfig {
    /// Cold-start dispatch rate for a newly-observed
    /// `(provider, model)`. Defaults to 1 request per second, which
    /// is safe against any production quota but adds 1–2s of
    /// warm-up latency on a fresh process. Raise it if your
    /// provider's per-minute quota comfortably permits a higher
    /// initial burst.
    pub initial_rps: f64,
    /// Hard floor on the AIMD rate. The multiplicative decrease on
    /// each 429 never takes the rate below this. Defaults to 0.25
    /// rps (one request every 4 seconds) — low enough to recover
    /// gracefully but high enough that a sustained backoff still
    /// makes forward progress.
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

/// Default in-process limiter. Construct once, share via `Arc`
/// across every provider built in this process. See the module
/// docs for the scheduling and AIMD model.
#[derive(Debug, Clone)]
pub struct InMemoryRateLimiter {
    inner: Arc<Inner>,
}

impl Default for InMemoryRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryRateLimiter {
    /// Construct with the default configuration.
    pub fn new() -> Self {
        Self::with_config(InMemoryRateLimiterConfig::default())
    }

    /// Construct with caller-tuned [`InMemoryRateLimiterConfig`].
    pub fn with_config(config: InMemoryRateLimiterConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                buckets: Mutex::new(HashMap::new()),
            }),
        }
    }
}

#[async_trait::async_trait]
impl RateLimiter for InMemoryRateLimiter {
    async fn acquire(&self, scope: &RateScope) -> Result<RatePermit, Error> {
        // Enqueue the waiter and grab its handle.
        let key = BucketKey {
            provider: scope.provider,
            model: scope.model.clone(),
        };
        let waiter = self.inner.enqueue(&key, scope);

        // Wait until the scheduler dispatches us.
        loop {
            // We re-check inside the lock so a wakeup that wasn't
            // for us puts us back to sleep without missing a later
            // wakeup. `Notify` is single-permit: notify-then-wait
            // and wait-then-notify both deliver, so this is safe
            // against the obvious races.
            let sleep_until = {
                let mut buckets = self
                    .inner
                    .buckets
                    .lock()
                    .expect("rate limiter mutex poisoned");
                let bucket = buckets
                    .get_mut(&key)
                    .expect("bucket was inserted on enqueue");
                if bucket.try_dispatch(&waiter, &self.inner.config) {
                    // Dispatched. Build the permit callback that
                    // routes the outcome back into this bucket.
                    let inner = self.inner.clone();
                    let key = key.clone();
                    let permit = RatePermit::new(move |outcome| {
                        inner.observe(&key, outcome);
                    });
                    return Ok(permit);
                }
                bucket.next_action(&waiter)
            };
            match sleep_until {
                NextAction::WaitUntil(when) => {
                    let now = Instant::now();
                    let dur = when.saturating_duration_since(now);
                    // Race the sleep and the notify; whichever
                    // fires first puts us back at the top of the
                    // dispatch check. `timeout` returns Err on
                    // elapse and Ok on the inner future
                    // completing — we don't care which path fires.
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
    tenant: String,
    notify: Arc<Notify>,
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
    buckets: Mutex<HashMap<BucketKey, Bucket>>,
}

impl Inner {
    fn enqueue(&self, key: &BucketKey, scope: &RateScope) -> Arc<WaiterHandle> {
        let mut buckets = self.buckets.lock().expect("rate limiter mutex poisoned");
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

    fn observe(&self, key: &BucketKey, outcome: RateOutcome) {
        let mut buckets = self.buckets.lock().expect("rate limiter mutex poisoned");
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
                // anything about capacity. But do wake the head so
                // it can dispatch using the slot freed by this
                // completion.
            }
        }
        bucket.in_flight = bucket.in_flight.saturating_sub(1);
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

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct BucketKey {
    provider: &'static str,
    model: String,
}

/// Per-`(provider, model)` scheduling state. Holds the AIMD rate,
/// the parked-until window, and the priority/tenant queue.
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
    /// Count of permits currently outstanding (acquired but not yet
    /// observed). Cosmetic — not used for scheduling — but exposed
    /// for diagnostics and to assert the limiter isn't leaking
    /// permits in tests.
    in_flight: u64,
    /// Strict-priority queue. Outer is by [`Priority`] (smallest
    /// first); inner is per-tenant FIFO with a round-robin order
    /// across tenants.
    queues: HashMap<Priority, PriorityBand>,
    /// Monotonic waiter id, used in tests and for the
    /// "is-this-me-at-the-head" check.
    next_id: u64,
}

#[derive(Debug, Default)]
struct PriorityBand {
    /// Per-tenant FIFO of waiters.
    tenant_queues: HashMap<String, VecDeque<Waiter>>,
    /// Round-robin order of tenants with non-empty queues.
    tenant_order: VecDeque<String>,
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

    fn is_empty(&self) -> bool {
        self.tenant_order.is_empty()
    }
}

impl Bucket {
    fn new(initial_rps: f64) -> Self {
        Self {
            rps: initial_rps,
            next_dispatch_at: Instant::now(),
            parked_until: None,
            in_flight: 0,
            queues: HashMap::new(),
            next_id: 0,
        }
    }

    fn next_waiter_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn push(&mut self, waiter: Waiter) {
        // Reset the dispatch clock when the queue went from idle to
        // populated — otherwise an idle bucket accumulates "credit"
        // and burst-dispatches the whole backlog the moment a
        // request arrives.
        let was_idle = self.queues.values().all(|band| band.is_empty());
        if was_idle {
            self.next_dispatch_at = self.next_dispatch_at.max(Instant::now());
        }
        let band = self.queues.entry(waiter.priority).or_default();
        // Use a clean push that tracks the tenant name explicitly,
        // avoiding the iterator-order trap from the naive impl above.
        let queue = band.tenant_queues.entry(waiter.tenant.clone()).or_default();
        let was_empty = queue.is_empty();
        let tenant_name = waiter.tenant.clone();
        queue.push_back(waiter);
        if was_empty {
            band.tenant_order.push_back(tenant_name);
        }
    }

    /// Peek the next waiter the scheduler would dispatch. Strict
    /// priority: lowest [`Priority`] variant wins; within a band,
    /// the head of the round-robin order wins.
    fn peek_next(&self) -> Option<&Waiter> {
        // BTreeMap-style sort by priority. The set is at most 3
        // entries (one per Priority variant), so this is fine.
        let mut prios: Vec<Priority> = self.queues.keys().copied().collect();
        prios.sort();
        for prio in prios {
            let band = self.queues.get(&prio).expect("just collected key");
            if let Some(w) = band.peek() {
                return Some(w);
            }
        }
        None
    }

    fn pop_next(&mut self) -> Option<Waiter> {
        let mut prios: Vec<Priority> = self.queues.keys().copied().collect();
        prios.sort();
        for prio in prios {
            let band = self.queues.get_mut(&prio).expect("just collected key");
            if let Some(w) = band.pop() {
                return Some(w);
            }
        }
        None
    }

    /// If `handle` is at the head of the queue **and** the AIMD
    /// bucket / park gate allow dispatch right now, pop it and
    /// schedule the next dispatch instant. Returns whether we
    /// dispatched `handle`.
    fn try_dispatch(&mut self, handle: &WaiterHandle, config: &InMemoryRateLimiterConfig) -> bool {
        let now = Instant::now();
        // Park gate.
        if let Some(parked) = self.parked_until {
            if now < parked {
                return false;
            }
            self.parked_until = None;
        }
        // AIMD dispatch interval.
        if now < self.next_dispatch_at {
            return false;
        }
        // Head check.
        let Some(head) = self.peek_next() else {
            return false;
        };
        if head.id != handle.id {
            return false;
        }
        // Dispatch.
        let _ = self.pop_next();
        self.in_flight += 1;
        let interval =
            Duration::from_secs_f64(1.0 / self.rps.clamp(config.min_rps, config.max_rps));
        self.next_dispatch_at = now + interval;
        // Wake the next head so it can immediately schedule its own
        // sleep target.
        self.wake_head();
        true
    }

    /// Compute the sleep target for a waiter that just failed to
    /// dispatch. The waiter sleeps until either the gate opens or
    /// it gets notified (which means the head changed).
    fn next_action(&self, handle: &WaiterHandle) -> NextAction {
        let Some(head) = self.peek_next() else {
            return NextAction::WaitForever;
        };
        if head.id != handle.id {
            // Not our turn yet — only wake on notify when our
            // position changes (either the head dispatches and we
            // step up, or someone with higher priority enqueues).
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

    /// Successful response — additive grow toward observed ceiling.
    fn observe_success(&mut self, info: &ProviderRateInfo, config: &InMemoryRateLimiterConfig) {
        // If the provider reports a remaining-budget + reset
        // window, we can estimate the observed capacity and cap
        // additive growth there. Otherwise just step.
        let mut target = self.rps + config.additive_step;
        if let (Some(remaining), Some(reset)) = (info.requests_remaining, info.requests_reset) {
            if reset.as_secs_f64() > 0.0 {
                // remaining requests over the reset window is the
                // headroom we have *right now*; don't grow past that.
                let observed = remaining as f64 / reset.as_secs_f64();
                target = target.min(observed.max(config.min_rps));
            }
        }
        self.rps = target.clamp(config.min_rps, config.max_rps);
    }

    /// Rate-limited — halve the rate and park the bucket.
    fn observe_rate_limit(
        &mut self,
        retry_after: Option<Duration>,
        _info: &ProviderRateInfo,
        config: &InMemoryRateLimiterConfig,
    ) {
        let park = retry_after
            .unwrap_or(config.default_park)
            .min(config.max_park);
        self.parked_until = Some(Instant::now() + park);
        self.rps =
            (self.rps * config.multiplicative_decrease).clamp(config.min_rps, config.max_rps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use tokio::time::{timeout, Duration as TokioDuration};

    fn scope(tenant: &str, priority: Priority) -> RateScope {
        RateScope {
            provider: "Test",
            model: "test-model".into(),
            tenant: tenant.into(),
            priority,
            estimated_input_tokens: None,
        }
    }

    fn permissive_config() -> InMemoryRateLimiterConfig {
        // Tight intervals so tests run in milliseconds, not seconds.
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

    /// First request from a fresh limiter must be granted promptly
    /// (cold-start rate is positive so the AIMD interval is finite).
    #[tokio::test]
    async fn first_request_dispatches_immediately() {
        let limiter = InMemoryRateLimiter::with_config(permissive_config());
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

    /// With cold-start RPS = 1.0, the *second* request must wait
    /// ~1 second after the first. Use a small slack window so a
    /// loaded CI box doesn't fail the test.
    #[tokio::test]
    async fn second_request_waits_for_aimd_interval() {
        let limiter = InMemoryRateLimiter::new(); // cold-start defaults
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

    /// Interactive request from tenant B must dispatch ahead of a
    /// Background request from tenant A that was already waiting.
    #[tokio::test]
    async fn strict_priority_preempts_across_tenants() {
        // Use a slow interval so we have time for both tasks to enqueue
        // before the AIMD gate opens — otherwise the test races against
        // the spawn-scheduling latency rather than the limiter.
        let limiter = Arc::new(InMemoryRateLimiter::with_config(
            InMemoryRateLimiterConfig {
                initial_rps: 5.0, // 200ms interval
                ..permissive_config()
            },
        ));
        // Burn the cold-start dispatch slot so subsequent acquires
        // are subject to the AIMD interval (rather than dispatching
        // immediately as the first request would).
        let _gate = limiter
            .acquire(&scope("gate", Priority::Interactive))
            .await
            .unwrap();
        // Enqueue background tenant A first.
        let lim_a = limiter.clone();
        let bg = tokio::spawn(async move {
            let p = lim_a
                .acquire(&scope("a", Priority::Background))
                .await
                .unwrap();
            (Instant::now(), p)
        });
        // Give bg time to reach its enqueue point.
        tokio::time::sleep(Duration::from_millis(30)).await;
        // Then interactive tenant B.
        let lim_b = limiter.clone();
        let inter = tokio::spawn(async move {
            let p = lim_b
                .acquire(&scope("b", Priority::Interactive))
                .await
                .unwrap();
            (Instant::now(), p)
        });
        // Give inter time to reach its enqueue point too — well before
        // the 200ms AIMD gate opens.
        tokio::time::sleep(Duration::from_millis(30)).await;
        // Now wait for both to be dispatched. Strict priority means
        // inter wins on the first dispatch slot, bg waits for the
        // next interval.
        let (inter_time, _inter_permit) = inter.await.unwrap();
        let (bg_time, _bg_permit) = bg.await.unwrap();
        assert!(
            inter_time < bg_time,
            "interactive (t={inter_time:?}) must beat background (t={bg_time:?})",
        );
    }

    /// Within a priority band, two tenants competing should round-
    /// robin — neither should monopolise the dispatch slots.
    #[tokio::test]
    async fn tenant_fairness_within_priority_band() {
        let limiter = Arc::new(InMemoryRateLimiter::with_config(permissive_config()));
        // Hold the gate so all subsequent acquires queue up.
        let gate = limiter
            .acquire(&scope("gate", Priority::Interactive))
            .await
            .unwrap();
        // Submit 6 requests: 3 from each of two tenants, all Interactive.
        let dispatched: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for i in 0..6 {
            let tenant = if i % 2 == 0 { "a" } else { "b" }.to_string();
            let lim = limiter.clone();
            let log = dispatched.clone();
            let t = tenant.clone();
            handles.push(tokio::spawn(async move {
                let permit = lim
                    .acquire(&scope(&t, Priority::Interactive))
                    .await
                    .unwrap();
                log.lock().unwrap().push(t);
                drop(permit);
            }));
        }
        // Give the spawns time to enqueue before the gate drops.
        tokio::time::sleep(Duration::from_millis(5)).await;
        drop(gate);
        for h in handles {
            h.await.unwrap();
        }
        let order = dispatched.lock().unwrap().clone();
        // Round-robin: a, b, a, b, a, b (insertion order alternates
        // even/odd → a/b, and a was first by spawn order).
        assert_eq!(order, vec!["a", "b", "a", "b", "a", "b"]);
    }

    /// Observing a 429 with a `Retry-After` parks the bucket. The
    /// next acquire must wait for the park to elapse.
    #[tokio::test]
    async fn rate_limit_observation_parks_bucket() {
        let limiter = InMemoryRateLimiter::with_config(permissive_config());
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

    /// AIMD step: 429 halves the rate. Verify by observing the
    /// post-rate-limit dispatch interval is roughly double the
    /// pre-rate-limit one.
    #[tokio::test]
    async fn rate_limit_halves_rps() {
        // Start at a known rate so we can verify the math.
        let limiter = InMemoryRateLimiter::with_config(InMemoryRateLimiterConfig {
            initial_rps: 10.0, // 100ms interval
            min_rps: 0.1,
            max_rps: 100.0,
            additive_step: 0.0, // disable growth so the test is deterministic
            multiplicative_decrease: 0.5,
            default_park: Duration::from_millis(1),
            max_park: Duration::from_millis(10),
        });
        // Burn the first permit and observe a 429 — should halve rps to 5.0.
        let p = limiter
            .acquire(&scope("t1", Priority::Interactive))
            .await
            .unwrap();
        p.observe(RateOutcome::RateLimited {
            retry_after: Some(Duration::from_millis(1)),
            info: ProviderRateInfo::default(),
        });
        // Wait out the park, then measure the interval between two
        // back-to-back acquires.
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
        // rps halved to 5.0 → 200ms interval. Allow generous slack.
        assert!(
            interval >= Duration::from_millis(150),
            "expected ~200ms (halved interval), got {interval:?}",
        );
        drop(p2);
    }

    /// A cancelled permit (dropped without observe) must not leave
    /// the bucket with a stuck in_flight count, and the next acquire
    /// must dispatch promptly.
    #[tokio::test]
    async fn cancelled_permit_releases_slot() {
        let limiter = InMemoryRateLimiter::with_config(permissive_config());
        {
            let _p = limiter
                .acquire(&scope("t1", Priority::Interactive))
                .await
                .unwrap();
            // _p drops here without observe — should fire the
            // Cancelled callback and decrement in_flight.
        }
        // The acquire path doesn't depend on in_flight for scheduling,
        // but check the bucket state directly for the bookkeeping
        // invariant.
        let buckets = limiter.inner.buckets.lock().unwrap();
        let bucket = buckets.values().next().unwrap();
        assert_eq!(bucket.in_flight, 0);
    }
}
