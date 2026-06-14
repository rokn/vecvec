//! The core service facade.
//!
//! [`Service`] is the single transport-agnostic entry point for all logical
//! operations; the gRPC adapters (and, later, REST) call straight into it. It owns
//! the [`Registry`] of durable collections and the [`BlockingBridge`], so CPU-bound
//! work (search, batch insert + fsync) runs off the reactor. On construction it
//! recovers every collection found under the data directory before serving.

use std::path::PathBuf;
use std::sync::Arc;

use vecvec_core::durable::read_config;
use vecvec_core::{CollectionConfig, DurableCollection, FsyncMode, Metric};

use crate::blocking::{BlockingBridge, BridgeError};
use crate::registry::Registry;

/// Errors surfaced by [`Service`] operations.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// No collection with the given name exists.
    #[error("collection {0:?} not found")]
    NotFound(String),
    /// A collection with the given name already exists.
    #[error("collection {0:?} already exists")]
    AlreadyExists(String),
    /// An error from the core engine (e.g. dimension mismatch, I/O).
    #[error(transparent)]
    Core(#[from] vecvec_core::CoreError),
    /// CPU-bound work failed to run to completion.
    #[error("compute error: {0}")]
    Bridge(#[from] BridgeError),
}

/// The transport-agnostic service facade.
pub struct Service {
    registry: Registry,
    bridge: BlockingBridge,
    data_dir: PathBuf,
    fsync: FsyncMode,
}

impl Service {
    /// Opens a service rooted at `data_dir`, recovering all existing collections
    /// before returning. Uses a `cpu_threads`-wide compute pool allowing
    /// `max_inflight` concurrent CPU jobs.
    pub fn open(
        data_dir: impl Into<PathBuf>,
        cpu_threads: usize,
        max_inflight: usize,
        fsync: FsyncMode,
    ) -> Result<Self, ServiceError> {
        let service = Self {
            registry: Registry::new(),
            bridge: BlockingBridge::new(cpu_threads, max_inflight),
            data_dir: data_dir.into(),
            fsync,
        };
        service.recover_all()?;
        Ok(service)
    }

    /// The collection registry (read-only access for callers that need stats).
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    fn collections_dir(&self) -> PathBuf {
        self.data_dir.join("collections")
    }

    /// Recovers every collection directory under the data dir.
    fn recover_all(&self) -> Result<(), ServiceError> {
        let dir = self.collections_dir();
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(&dir).map_err(vecvec_core::CoreError::from)? {
            let entry = entry.map_err(vecvec_core::CoreError::from)?;
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let config = read_config(&path)?;
                let collection = DurableCollection::open(&path, config, self.fsync)?;
                self.registry.insert_new(name, Arc::new(collection));
            }
        }
        Ok(())
    }

    /// Creates a durable collection.
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
        if self.registry.get(&name).is_some() {
            return Err(ServiceError::AlreadyExists(name));
        }
        let dir = self.collections_dir().join(&name);
        let config = CollectionConfig::new(name.clone(), dim, metric);
        let collection = Arc::new(DurableCollection::open(&dir, config, self.fsync)?);
        if !self.registry.insert_new(name.clone(), collection) {
            return Err(ServiceError::AlreadyExists(name));
        }
        Ok(())
    }

    /// Durably appends a batch of vectors to a collection, returning their ids.
    pub async fn upsert(
        &self,
        collection: String,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<u64>, ServiceError> {
        let durable = self
            .registry
            .get(&collection)
            .ok_or(ServiceError::NotFound(collection))?;
        let ids = self.bridge.run(move || durable.upsert(vectors)).await??;
        Ok(ids)
    }

    /// Returns the best `k` matches for `query` in a collection.
    pub async fn query(
        &self,
        collection: String,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<(u64, f32)>, ServiceError> {
        let durable = self
            .registry
            .get(&collection)
            .ok_or(ServiceError::NotFound(collection))?;
        let col = durable.collection().clone();
        let results = self
            .bridge
            .run(move || col.search(&query, k, None))
            .await??;
        Ok(results.into_iter().map(|s| (s.id.get(), s.score)).collect())
    }
}
