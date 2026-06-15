//! The blocking bridge: runs CPU-bound work off the async reactor.
//!
//! HNSW build/search and the flat scan are CPU-bound and would starve tokio's
//! worker threads if `.await`ed directly. [`BlockingBridge`] dispatches such work
//! to a dedicated rayon pool, gated by a [`Semaphore`] for backpressure, and
//! delivers the result back over a oneshot channel. Because the work runs in a
//! rayon closure, any synchronous locks it takes (e.g. the collection's appendable
//! `RwLock`) are released before control returns to async code — never held across
//! an `.await`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use tokio::sync::{Semaphore, oneshot};

/// Error from running work on the [`BlockingBridge`].
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// The task was dropped before producing a result (shutdown).
    #[error("blocking task was cancelled")]
    Cancelled,
    /// The task panicked; the panic was contained and reported as an error.
    #[error("blocking task panicked")]
    Panicked,
}

/// A rayon pool + semaphore for running CPU-bound closures from async code.
pub struct BlockingBridge {
    permits: Arc<Semaphore>,
    pool: Arc<rayon::ThreadPool>,
}

impl BlockingBridge {
    /// Creates a bridge with a `threads`-wide rayon pool and `max_inflight`
    /// concurrent jobs.
    pub fn new(threads: usize, max_inflight: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("vecvec-cpu-{i}"))
            .build()
            .expect("failed to build rayon thread pool");
        Self {
            permits: Arc::new(Semaphore::new(max_inflight.max(1))),
            pool: Arc::new(pool),
        }
    }

    /// Runs `f` on the CPU pool and awaits its result, applying backpressure once
    /// `max_inflight` jobs are in flight. A panic in `f` is contained and surfaced
    /// as [`BridgeError::Panicked`].
    pub async fn run<F, T>(&self, f: F) -> Result<T, BridgeError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| BridgeError::Cancelled)?;
        let (tx, rx) = oneshot::channel();
        self.pool.spawn(move || {
            let _permit = permit; // released when the closure finishes
            let result = catch_unwind(AssertUnwindSafe(f));
            let _ = tx.send(result);
        });
        match rx.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(_panic)) => Err(BridgeError::Panicked),
            Err(_) => Err(BridgeError::Cancelled),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn runs_work_and_returns_result() {
        let bridge = BlockingBridge::new(2, 4);
        let out = bridge.run(|| 21 * 2).await.unwrap();
        assert_eq!(out, 42);
    }

    #[tokio::test]
    async fn respects_max_inflight_backpressure() {
        // The semaphore caps concurrently-running closures at max_inflight even when
        // the rayon pool has more threads. Track live concurrency from inside the
        // closures; the observed peak must never exceed the cap.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let bridge = Arc::new(BlockingBridge::new(8, 2)); // 8 threads, cap 2 in flight
        let cur = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..24 {
            let bridge = bridge.clone();
            let cur = cur.clone();
            let peak = peak.clone();
            handles.push(tokio::spawn(async move {
                bridge
                    .run(move || {
                        let now = cur.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        cur.fetch_sub(1, Ordering::SeqCst);
                    })
                    .await
                    .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let observed = peak.load(Ordering::SeqCst);
        assert!(observed >= 1);
        assert!(
            observed <= 2,
            "peak in-flight {observed} exceeded max_inflight=2"
        );
    }

    #[tokio::test]
    async fn contains_panics() {
        let bridge = BlockingBridge::new(2, 4);
        let err = bridge
            .run(|| panic!("boom"))
            .await
            .map(|_: ()| ())
            .unwrap_err();
        assert!(matches!(err, BridgeError::Panicked));
        // The bridge is still usable afterwards.
        assert_eq!(bridge.run(|| 7).await.unwrap(), 7);
    }
}
