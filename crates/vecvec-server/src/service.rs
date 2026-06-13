//! The core service facade.
//!
//! [`Service`] is the single transport-agnostic entry point for all logical
//! operations; the gRPC adapters (and, later, REST) call straight into it. It owns
//! the [`Registry`] and the [`BlockingBridge`], so CPU-bound work (search, batch
//! insert) runs off the reactor.

use vecvec_core::{CollectionConfig, Metric};

use crate::blocking::{BlockingBridge, BridgeError};
use crate::registry::{AlreadyExists, Registry};

/// Errors surfaced by [`Service`] operations.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// No collection with the given name exists.
    #[error("collection {0:?} not found")]
    NotFound(String),
    /// A collection with the given name already exists.
    #[error("collection {0:?} already exists")]
    AlreadyExists(String),
    /// An error from the core engine (e.g. dimension mismatch).
    #[error(transparent)]
    Core(#[from] vecvec_core::CoreError),
    /// CPU-bound work failed to run to completion.
    #[error("compute error: {0}")]
    Bridge(#[from] BridgeError),
}

impl From<AlreadyExists> for ServiceError {
    fn from(e: AlreadyExists) -> Self {
        ServiceError::AlreadyExists(e.0)
    }
}

/// The transport-agnostic service facade.
pub struct Service {
    registry: Registry,
    bridge: BlockingBridge,
}

impl Service {
    /// Builds a service with a `cpu_threads`-wide compute pool allowing
    /// `max_inflight` concurrent CPU jobs.
    pub fn new(cpu_threads: usize, max_inflight: usize) -> Self {
        Self {
            registry: Registry::new(),
            bridge: BlockingBridge::new(cpu_threads, max_inflight),
        }
    }

    /// The collection registry (read-only access for callers that need stats).
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Creates a collection. Cheap, so it runs inline (no compute pool).
    pub fn create_collection(
        &self,
        name: String,
        dim: usize,
        metric: Metric,
    ) -> Result<(), ServiceError> {
        if dim == 0 {
            return Err(ServiceError::Core(
                vecvec_core::CoreError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                },
            ));
        }
        self.registry
            .create(CollectionConfig::new(name, dim, metric))?;
        Ok(())
    }

    /// Appends a batch of vectors to a collection, returning their assigned ids.
    /// Runs the insert on the compute pool.
    pub async fn upsert(
        &self,
        collection: String,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<u64>, ServiceError> {
        let col = self
            .registry
            .get(&collection)
            .ok_or(ServiceError::NotFound(collection))?;
        let ids = self
            .bridge
            .run(move || col.insert_batch(&vectors))
            .await??;
        Ok(ids.into_iter().map(|g| g.get()).collect())
    }

    /// Returns the best `k` matches for `query` in a collection. Runs the search on
    /// the compute pool.
    pub async fn query(
        &self,
        collection: String,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<(u64, f32)>, ServiceError> {
        let col = self
            .registry
            .get(&collection)
            .ok_or(ServiceError::NotFound(collection))?;
        let results = self
            .bridge
            .run(move || col.search(&query, k, None))
            .await??;
        Ok(results.into_iter().map(|s| (s.id.get(), s.score)).collect())
    }
}
