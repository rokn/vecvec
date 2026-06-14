//! Automatic-commit rules.
//!
//! A [`VersioningPolicy`] declares when the working head should be auto-committed.
//! Triggers are OR-ed: every N writes, and/or every T milliseconds. Writes
//! accumulate into the working head (git staging); the trigger commits them. We
//! never commit per write (that would explode the version count). Time is taken
//! from an injectable [`Clock`] so the interval trigger is deterministically
//! testable.

use serde::{Deserialize, Serialize};

/// Rules for automatic commits. All `None` = manual commits only.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct VersioningPolicy {
    /// Commit once this many writes have accumulated since the last commit.
    pub every_n_writes: Option<u64>,
    /// Commit once this many milliseconds have elapsed since the last commit (only
    /// if there are uncommitted writes).
    pub interval_ms: Option<u64>,
}

impl VersioningPolicy {
    /// A policy that only commits on explicit request.
    pub fn manual() -> Self {
        Self::default()
    }

    /// A policy that commits every `n` writes.
    pub fn every_n_writes(n: u64) -> Self {
        Self {
            every_n_writes: Some(n),
            interval_ms: None,
        }
    }

    /// Whether any automatic trigger is configured.
    pub fn is_automatic(&self) -> bool {
        self.every_n_writes.is_some() || self.interval_ms.is_some()
    }
}

/// A source of wall-clock time (injectable for tests).
pub trait Clock: Send + Sync {
    /// The current time in unix milliseconds.
    fn now_ms(&self) -> u64;
}

/// The default clock backed by the system time.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// Tracks progress toward the next automatic commit.
#[derive(Debug, Clone)]
pub struct TriggerEvaluator {
    policy: VersioningPolicy,
    writes_since_commit: u64,
    last_commit_ms: u64,
}

impl TriggerEvaluator {
    /// Creates an evaluator anchored at the current time.
    pub fn new(policy: VersioningPolicy, clock: &dyn Clock) -> Self {
        Self {
            policy,
            writes_since_commit: 0,
            last_commit_ms: clock.now_ms(),
        }
    }

    /// The current policy.
    pub fn policy(&self) -> VersioningPolicy {
        self.policy
    }

    /// Records `n` writes toward the next trigger.
    pub fn record_writes(&mut self, n: u64) {
        self.writes_since_commit = self.writes_since_commit.saturating_add(n);
    }

    /// Uncommitted writes accumulated so far.
    pub fn pending_writes(&self) -> u64 {
        self.writes_since_commit
    }

    /// Whether a commit should fire now.
    pub fn should_commit(&self, clock: &dyn Clock) -> bool {
        if let Some(n) = self.policy.every_n_writes
            && self.writes_since_commit >= n
        {
            return true;
        }
        if let Some(ms) = self.policy.interval_ms
            && self.writes_since_commit > 0
            && clock.now_ms().saturating_sub(self.last_commit_ms) >= ms
        {
            return true;
        }
        false
    }

    /// Records that a commit happened, advancing the counters. For the N-writes
    /// trigger this *subtracts* N (rather than zeroing) so a burst that overshoots
    /// still fires at the next exact multiple.
    pub fn note_commit(&mut self, clock: &dyn Clock) {
        if let Some(n) = self.policy.every_n_writes
            && self.writes_since_commit >= n
        {
            self.writes_since_commit -= n;
        } else {
            self.writes_since_commit = 0;
        }
        self.last_commit_ms = clock.now_ms();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Default)]
    struct ManualClock(AtomicU64);
    impl ManualClock {
        fn advance(&self, ms: u64) {
            self.0.fetch_add(ms, Ordering::Relaxed);
        }
    }
    impl Clock for ManualClock {
        fn now_ms(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn fires_at_exact_multiples_of_n() {
        let clock = ManualClock::default();
        let mut ev = TriggerEvaluator::new(VersioningPolicy::every_n_writes(100), &clock);
        let mut commits = Vec::new();
        let mut total = 0u64;
        // Bursty writes of 30 at a time up to 300.
        for _ in 0..10 {
            ev.record_writes(30);
            total += 30;
            if ev.should_commit(&clock) {
                commits.push(total);
                ev.note_commit(&clock);
            }
        }
        // Commits should be observed right after crossing 100, 200, 300.
        assert_eq!(commits, vec![120, 210, 300]);
        // 0 pending after the last exact multiple (300 - 3*100 = 0).
        assert_eq!(ev.pending_writes(), 0);
    }

    #[test]
    fn interval_fires_only_with_pending_writes() {
        let clock = ManualClock::default();
        let policy = VersioningPolicy {
            every_n_writes: None,
            interval_ms: Some(1000),
        };
        let mut ev = TriggerEvaluator::new(policy, &clock);
        clock.advance(2000);
        assert!(!ev.should_commit(&clock)); // time passed but no writes
        ev.record_writes(1);
        assert!(ev.should_commit(&clock));
        ev.note_commit(&clock);
        assert!(!ev.should_commit(&clock)); // just committed
        clock.advance(1500);
        ev.record_writes(1);
        assert!(ev.should_commit(&clock));
    }
}
