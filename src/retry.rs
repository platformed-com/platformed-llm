//! Retry helpers for transient provider failures.
//!
//! [`crate::generate`] / [`crate::Response::buffer`] can fail with a
//! handful of transient shapes (a 429 from Vertex, a 5xx blip,
//! a connection drop mid-SSE). Production callers want to retry those
//! with backoff, and rolling that loop by hand is error-prone — it's
//! easy to forget `Retry-After`, to compound a 30-second hint into a
//! 30-minute wait, or to retry an [`Error::Auth`] that will never
//! succeed.
//!
//! This module centralises the two parts of that loop:
//!
//! - [`RetryPolicy`] — how many attempts, how long to wait between
//!   them, and how a provider-supplied [`Error::retry_after`] hint
//!   interacts with the exponential schedule. Compute a delay with
//!   [`RetryPolicy::delay_after`] if you want to drive the loop
//!   yourself.
//! - [`retry()`] — wrap an async operation in the loop. The closure
//!   encompasses the *entire* operation (including any streaming
//!   consumption inside it), so a mid-stream failure simply re-enters
//!   the closure from the top. The closure is what the caller would
//!   write anyway; the only addition is the wrapping call. See the
//!   `debug_streaming` and `mock_provider` examples for the buffered
//!   and streaming shapes side-by-side.
//!
//! # What gets retried
//!
//! The policy retries any error for which [`Error::is_retryable`]
//! returns `true` — rate limits, transient 5xx / 429, transport
//! errors, and mid-stream failures (SSE parse / connection drop).
//! Every retry is a fresh request; the helper does not attempt to
//! "resume" a partially-streamed response.
//!
//! If you're streaming directly to a user and the first attempt
//! emitted some tokens before failing, retrying will produce
//! different output that won't stitch with what you already showed.
//! That's a caller-policy concern — drive the loop with
//! [`RetryPolicy::delay_after`] yourself and decide whether to
//! discard the partial output, surface a "retry?" prompt, or stop.
//!
//! # What doesn't
//!
//! [`Error::Auth`], [`Error::Config`], [`Error::InvalidPrompt`],
//! [`Error::ModelNotAvailable`], [`Error::ContextWindowExceeded`],
//! [`Error::Compaction`], and [`Error::Provider`] with `retryable:
//! false` are terminal — re-issuing the same request hits the same
//! error. The helper propagates them on the first failure.
//!
//! `ContextWindowExceeded` in particular should be paired with
//! [`crate::Compactor`] (see the `auto_compaction` example), not
//! retried blindly.

use std::time::Duration;

use crate::Error;

/// Knobs governing the retry loop. Construct with
/// [`RetryPolicy::standard`] for sensible defaults, or build manually
/// for fine control. All fields are public; mutate them directly.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum number of attempts. `1` disables retries — the first
    /// failure is the final failure (use [`Self::none`] for this).
    pub max_attempts: u32,
    /// Delay before the second attempt (and the seed for subsequent
    /// exponential backoff).
    pub initial_backoff: Duration,
    /// Multiplier compounded each attempt — the delay before
    /// attempt N is `initial_backoff * multiplier^(N-1)`.
    pub backoff_multiplier: f64,
    /// Upper bound on any single delay, including a provider-supplied
    /// [`Error::retry_after`] hint. A provider asking us to wait 30
    /// minutes from a tight in-process loop probably isn't what the
    /// caller wants; the cap stops a single misbehaving header from
    /// silently parking the task.
    pub max_backoff: Duration,
}

impl RetryPolicy {
    /// Sensible defaults: 4 attempts, 1s initial backoff, 2×
    /// multiplier, 60s cap on any single delay.
    pub fn standard() -> Self {
        Self {
            max_attempts: 4,
            initial_backoff: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(60),
        }
    }

    /// A no-retry policy — the first failure is the final failure.
    /// Useful when the caller wants to opt out of retries at one
    /// call site without restructuring the code.
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            ..Self::standard()
        }
    }

    /// Compute the wait before the next attempt given the error that
    /// just surfaced. Returns `None` if the error is terminal or
    /// attempts are exhausted — the caller propagates the error.
    ///
    /// Honours [`Error::retry_after`] when set; otherwise applies
    /// exponential backoff:
    /// `initial_backoff * backoff_multiplier^(attempt - 1)`. The
    /// result is capped at [`Self::max_backoff`] either way.
    ///
    /// `attempt` is the 1-indexed attempt number that just failed.
    /// `1` after the first failure, `2` after the second, …
    ///
    /// The fields on [`RetryPolicy`] are `pub` for ergonomics; a
    /// caller setting nonsense values (`NaN`, negative multiplier,
    /// overflow) saturates here rather than panicking. The clamp
    /// keeps `Duration::from_secs_f64` away from its panic
    /// preconditions.
    pub fn delay_after(&self, err: &Error, attempt: u32) -> Option<Duration> {
        // `attempt` is documented as 1-indexed (1 after the first
        // failure). `0` is undefined input from this method's POV
        // and would compute `multiplier ** -1 = 1 / multiplier`,
        // returning a delay even when `RetryPolicy::none()` was
        // explicitly chosen. Treat it as terminal.
        if attempt == 0 || !err.is_retryable() || attempt >= self.max_attempts {
            return None;
        }
        let max_secs = self.max_backoff.as_secs_f64();
        let nominal = err.retry_after().unwrap_or_else(|| {
            let base = self.initial_backoff.as_secs_f64();
            let mult = self.backoff_multiplier.powi((attempt as i32) - 1);
            let secs = base * mult;
            // Clamp away from `from_secs_f64`'s panic preconditions
            // (negative, non-finite, overflow). The cap is `max_secs`
            // anyway, so saturating up to that loses nothing.
            let safe_secs = if secs.is_finite() && secs >= 0.0 {
                secs.min(max_secs)
            } else {
                max_secs
            };
            Duration::from_secs_f64(safe_secs)
        });
        Some(nominal.min(self.max_backoff))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

/// Retry an async operation according to `policy`. The closure is
/// invoked once per attempt; on a retryable failure [`retry()`] sleeps
/// for [`RetryPolicy::delay_after`] and re-enters the closure from the
/// top.
///
/// The closure receives the 1-indexed attempt number — useful for
/// "[retry N…]" log lines, for resetting per-attempt UI state (a
/// streaming caller might erase what it printed before retrying), or
/// for tracing.
///
/// Returns the closure's `Ok` value, or the last observed `Err` once
/// the policy gives up. See the `debug_streaming` and `mock_provider`
/// examples for end-to-end uses.
///
/// # Runtime requirements
///
/// Sleeps via [`tokio::time::sleep`]; consumers on a `current_thread`
/// runtime must enable time on their builder (`Builder::new_current_thread().enable_time()`).
/// `#[tokio::main]` and `Builder::new_multi_thread().enable_all()`
/// already do this.
pub async fn retry<F, T>(policy: &RetryPolicy, mut op: F) -> Result<T, Error>
where
    F: AsyncFnMut(u32) -> Result<T, Error>,
{
    let mut attempt: u32 = 0;
    loop {
        attempt = attempt.saturating_add(1);
        match op(attempt).await {
            Ok(value) => return Ok(value),
            Err(err) => match policy.delay_after(&err, attempt) {
                Some(delay) => {
                    tracing::warn!(
                        attempt,
                        max_attempts = policy.max_attempts,
                        delay_ms = delay.as_millis() as u64,
                        error = %err,
                        "retrying after transient failure",
                    );
                    tokio::time::sleep(delay).await;
                }
                None => return Err(err),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn delay_after_returns_none_for_terminal_errors() {
        let policy = RetryPolicy::standard();
        assert!(policy.delay_after(&Error::auth("bad key"), 1).is_none());
        assert!(policy.delay_after(&Error::config("bad"), 1).is_none());
        assert!(policy
            .delay_after(&Error::invalid_prompt("bad"), 1)
            .is_none());
        assert!(policy
            .delay_after(&Error::ModelNotAvailable("gpt-x".into()), 1)
            .is_none());
        assert!(policy
            .delay_after(&Error::context_window_exceeded("OpenAI", "big"), 1)
            .is_none());
        assert!(policy.delay_after(&Error::compaction("empty"), 1).is_none());
        // `Provider { retryable: false }` is the provider explicitly
        // telling us "don't bother" — honour it.
        assert!(policy
            .delay_after(&Error::provider("OpenAI", "bad request"), 1)
            .is_none());
    }

    #[test]
    fn delay_after_returns_none_when_attempts_exhausted() {
        let policy = RetryPolicy {
            max_attempts: 3,
            ..RetryPolicy::standard()
        };
        let err = Error::rate_limit(None, "slow down");
        assert!(policy.delay_after(&err, 1).is_some());
        assert!(policy.delay_after(&err, 2).is_some());
        // `attempt == max_attempts` means we've used our budget.
        assert!(policy.delay_after(&err, 3).is_none());
        assert!(policy.delay_after(&err, 10).is_none());
    }

    #[test]
    fn delay_after_honours_retry_after_hint() {
        let policy = RetryPolicy::standard();
        let err = Error::rate_limit(Some(5), "slow down");
        assert_eq!(policy.delay_after(&err, 1), Some(Duration::from_secs(5)));
    }

    #[test]
    fn delay_after_caps_provider_hint_at_max_backoff() {
        let policy = RetryPolicy {
            max_backoff: Duration::from_secs(10),
            ..RetryPolicy::standard()
        };
        let err = Error::rate_limit(Some(60), "wait a minute");
        assert_eq!(policy.delay_after(&err, 1), Some(Duration::from_secs(10)));
    }

    #[test]
    fn delay_after_applies_exponential_backoff_when_no_hint() {
        let policy = RetryPolicy {
            max_attempts: 10,
            initial_backoff: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(60),
        };
        let err = Error::rate_limit(None, "slow down");
        assert_eq!(policy.delay_after(&err, 1), Some(Duration::from_secs(1)));
        assert_eq!(policy.delay_after(&err, 2), Some(Duration::from_secs(2)));
        assert_eq!(policy.delay_after(&err, 3), Some(Duration::from_secs(4)));
        assert_eq!(policy.delay_after(&err, 4), Some(Duration::from_secs(8)));
    }

    /// Caller-set nonsense values on the public fields must
    /// saturate to `max_backoff` rather than panic inside
    /// `Duration::from_secs_f64`. Use attempt >= 2 because the
    /// multiplier exponent is `attempt - 1` and any value (incl.
    /// NaN) to the 0th power is 1.0 in IEEE 754.
    #[test]
    fn delay_after_saturates_on_pathological_inputs() {
        let base = RetryPolicy {
            max_attempts: 5,
            initial_backoff: Duration::from_secs(1),
            backoff_multiplier: f64::NAN,
            max_backoff: Duration::from_secs(60),
        };
        let err = Error::rate_limit(None, "slow");
        // NaN multiplier × non-zero exponent → NaN → must clamp,
        // not panic.
        assert_eq!(base.delay_after(&err, 2), Some(Duration::from_secs(60)));
        // Negative multiplier × odd exponent → negative → must clamp.
        let neg_policy = RetryPolicy {
            backoff_multiplier: -2.0,
            ..base
        };
        assert_eq!(
            neg_policy.delay_after(&err, 2),
            Some(Duration::from_secs(60)),
        );
        // Overflow: huge multiplier × attempts → +inf → must clamp.
        let huge_policy = RetryPolicy {
            backoff_multiplier: 1e300,
            ..base
        };
        assert_eq!(
            huge_policy.delay_after(&err, 3),
            Some(Duration::from_secs(60)),
        );
    }

    /// `delay_after(_, 0)` must return `None` rather than computing
    /// `multiplier^-1 × initial_backoff` (which would suggest a
    /// retry even for `RetryPolicy::none()`). `attempt` is documented
    /// 1-indexed; `0` is undefined input.
    #[test]
    fn delay_after_with_zero_attempt_returns_none() {
        let policy = RetryPolicy::standard();
        let err = Error::rate_limit(None, "slow");
        assert_eq!(policy.delay_after(&err, 0), None);
        // Even `RetryPolicy::none()` would return `Some` without the
        // defence (because `attempt >= max_attempts` is `0 >= 1 = false`).
        let none = RetryPolicy::none();
        assert_eq!(none.delay_after(&err, 0), None);
    }

    /// The cap on the *exponential* branch (when no provider hint)
    /// must actually hold — a regression dropping `.min(max_backoff)`
    /// from the no-hint path should fail this.
    #[test]
    fn delay_after_caps_exponential_branch_at_max_backoff() {
        let policy = RetryPolicy {
            max_attempts: 10,
            initial_backoff: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(30),
        };
        // attempt 4 → 10 * 2^3 = 80s, capped at 30s.
        let err = Error::rate_limit(None, "slow");
        assert_eq!(policy.delay_after(&err, 4), Some(Duration::from_secs(30)));
    }

    /// Tight policy for tests — millisecond-scale delays so the
    /// suite doesn't pay for our exponential schedule.
    fn fast_policy() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(1),
            backoff_multiplier: 1.0,
            max_backoff: Duration::from_millis(1),
        }
    }

    #[tokio::test]
    async fn retry_returns_after_transient_failures() {
        let policy = fast_policy();
        let count = Cell::new(0u32);
        let result: Result<&'static str, Error> = retry(&policy, async |_| {
            count.set(count.get() + 1);
            if count.get() < 3 {
                Err(Error::rate_limit(None, "slow"))
            } else {
                Ok("done")
            }
        })
        .await;
        assert_eq!(result.unwrap(), "done");
        assert_eq!(count.get(), 3);
    }

    #[tokio::test]
    async fn retry_propagates_terminal_error_without_retrying() {
        let policy = RetryPolicy::standard();
        let count = Cell::new(0u32);
        let result: Result<(), Error> = retry(&policy, async |_| {
            count.set(count.get() + 1);
            Err(Error::auth_with_status(401, "bad key"))
        })
        .await;
        assert!(matches!(result, Err(Error::Auth { .. })));
        assert_eq!(count.get(), 1);
    }

    #[tokio::test]
    async fn retry_surfaces_last_error_when_exhausted() {
        let policy = fast_policy();
        let count = Cell::new(0u32);
        let result: Result<(), Error> = retry(&policy, async |attempt| {
            count.set(count.get() + 1);
            Err(Error::provider_with_status(
                "MockProvider",
                503,
                format!("attempt {attempt} failed"),
            ))
        })
        .await;
        assert!(matches!(result, Err(Error::Provider { .. })));
        assert_eq!(count.get(), 3);
    }

    #[tokio::test]
    async fn retry_passes_attempt_number_to_closure() {
        let policy = fast_policy();
        let observed: Cell<u32> = Cell::new(0);
        let result: Result<u32, Error> = retry(&policy, async |attempt| {
            observed.set(attempt);
            if attempt < 2 {
                Err(Error::rate_limit(None, "slow"))
            } else {
                Ok(attempt)
            }
        })
        .await;
        assert_eq!(result.unwrap(), 2);
        assert_eq!(observed.get(), 2);
    }
}
