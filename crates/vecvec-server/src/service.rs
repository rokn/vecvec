//! The core service facade.
//!
//! [`Service`] is the single transport-agnostic entry point for all logical
//! operations; the gRPC adapters (and, later, REST) call straight into it. It owns
//! the [`Registry`] of durable collections and the [`BlockingBridge`], so CPU-bound
//! work (search, batch insert + fsync) runs off the reactor. On construction it
//! recovers every collection found under the data directory before serving.

use std::path::PathBuf;
use std::sync::Arc;

use std::sync::Arc as StdArc;

use vecvec_core::durable::read_config;
use vecvec_core::version::{Manifest, VersionSelector};
use vecvec_core::{
    CollectionConfig, CompactionPolicy, DurableCollection, Filter, FsyncMode, Metric, Payload,
    ScoredGlobal,
};

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

/// Summary stats for a collection (powers the explorer's collection list + header).
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// The collection name.
    pub name: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// The number of live points.
    pub count: usize,
    /// The current HEAD version, if any commits exist.
    pub head_version: Option<u64>,
}

/// The transport-agnostic service facade.
pub struct Service {
    registry: Registry,
    bridge: BlockingBridge,
    data_dir: PathBuf,
    fsync: FsyncMode,
    /// Auto-compaction policy applied to every collection this service opens.
    compaction: CompactionPolicy,
}

impl Service {
    /// Opens a service rooted at `data_dir`, recovering all existing collections
    /// before returning. Uses a `cpu_threads`-wide compute pool allowing
    /// `max_inflight` concurrent CPU jobs. Auto-compaction is disabled (manual
    /// only); use [`open_with_compaction`](Self::open_with_compaction) to enable it.
    pub fn open(
        data_dir: impl Into<PathBuf>,
        cpu_threads: usize,
        max_inflight: usize,
        fsync: FsyncMode,
    ) -> Result<Self, ServiceError> {
        Self::open_with_compaction(
            data_dir,
            cpu_threads,
            max_inflight,
            fsync,
            CompactionPolicy::default(),
        )
    }

    /// Like [`open`](Self::open), but applies `compaction` to every collection it
    /// opens or creates. The server's background maintenance loop polls each
    /// collection's trigger and compacts off the write path.
    pub fn open_with_compaction(
        data_dir: impl Into<PathBuf>,
        cpu_threads: usize,
        max_inflight: usize,
        fsync: FsyncMode,
        compaction: CompactionPolicy,
    ) -> Result<Self, ServiceError> {
        let service = Self {
            registry: Registry::new(),
            bridge: BlockingBridge::new(cpu_threads, max_inflight),
            data_dir: data_dir.into(),
            fsync,
            compaction,
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
                let mut config = read_config(&path)?;
                config.compaction = self.compaction;
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
        let mut config = CollectionConfig::new(name.clone(), dim, metric);
        config.compaction = self.compaction;
        let collection = Arc::new(DurableCollection::open(&dir, config, self.fsync)?);
        if !self.registry.insert_new(name.clone(), collection) {
            return Err(ServiceError::AlreadyExists(name));
        }
        Ok(())
    }

    /// Durably appends a batch of `(vector, payload)` points, returning their ids.
    pub async fn upsert(
        &self,
        collection: String,
        points: Vec<(Vec<f32>, Option<Payload>)>,
    ) -> Result<Vec<u64>, ServiceError> {
        let durable = self
            .registry
            .get(&collection)
            .ok_or(ServiceError::NotFound(collection))?;
        let ids = self.bridge.run(move || durable.upsert(points)).await??;
        Ok(ids)
    }

    /// Atomically applies a mixed delete + upsert batch, optionally committing a
    /// new version afterwards (a transaction-like unit of work). Returns the
    /// upserted ids, how many points were newly deleted, and the version if one
    /// was created. `commit` is `Some((message, tag))` to commit after the batch.
    pub async fn write_batch(
        &self,
        collection: String,
        upserts: Vec<(Vec<f32>, Option<Payload>)>,
        deletes: Vec<u64>,
        commit: Option<(Option<String>, Option<String>)>,
    ) -> Result<vecvec_core::WriteBatchResult, ServiceError> {
        let durable = self.get(&collection)?;
        Ok(self
            .bridge
            .run(move || durable.write_batch(upserts, deletes, commit))
            .await??)
    }

    /// Returns the best `k` matches for `query`, optionally as of a past version
    /// (`at`) and/or constrained by a payload `filter`.
    pub async fn query(
        &self,
        collection: String,
        query: Vec<f32>,
        k: usize,
        at: Option<VersionSelector>,
        filter: Option<Filter>,
        include_payloads: bool,
    ) -> Result<Vec<(u64, f32, Option<Payload>)>, ServiceError> {
        let durable = self
            .registry
            .get(&collection)
            .ok_or(ServiceError::NotFound(collection))?;
        let results = self
            .bridge
            .run(move || {
                let results = match at {
                    Some(selector) => durable.search_at(&selector, &query, k, filter.as_ref()),
                    None => durable.search(&query, k, filter.as_ref()),
                }?;
                Ok::<_, vecvec_core::CoreError>(scored_with_payloads(
                    &durable,
                    results,
                    include_payloads,
                ))
            })
            .await??;
        Ok(results)
    }

    /// Recommend-by-example over positive/negative point ids.
    pub async fn recommend(
        &self,
        collection: String,
        positive: Vec<u64>,
        negative: Vec<u64>,
        k: usize,
        filter: Option<Filter>,
        include_payloads: bool,
    ) -> Result<Vec<(u64, f32, Option<Payload>)>, ServiceError> {
        let durable = self.get(&collection)?;
        let results = self
            .bridge
            .run(move || {
                let pos: Vec<_> = positive
                    .into_iter()
                    .map(vecvec_core::GlobalId::new)
                    .collect();
                let neg: Vec<_> = negative
                    .into_iter()
                    .map(vecvec_core::GlobalId::new)
                    .collect();
                let results = durable.recommend(&pos, &neg, k, filter.as_ref())?;
                Ok::<_, vecvec_core::CoreError>(scored_with_payloads(
                    &durable,
                    results,
                    include_payloads,
                ))
            })
            .await??;
        Ok(results)
    }

    fn get(&self, collection: &str) -> Result<StdArc<DurableCollection>, ServiceError> {
        self.registry
            .get(collection)
            .ok_or_else(|| ServiceError::NotFound(collection.to_owned()))
    }

    /// Commits the working state of a collection as a new version.
    pub async fn commit(
        &self,
        collection: String,
        message: Option<String>,
        tag: Option<String>,
    ) -> Result<u64, ServiceError> {
        let durable = self.get(&collection)?;
        Ok(self
            .bridge
            .run(move || durable.commit(message, tag))
            .await??)
    }

    /// Restores a collection's working state to a version (a forward commit).
    pub async fn restore(&self, collection: String, version: u64) -> Result<u64, ServiceError> {
        let durable = self.get(&collection)?;
        Ok(self.bridge.run(move || durable.restore(version)).await??)
    }

    /// Tags a version.
    pub async fn create_tag(
        &self,
        collection: String,
        name: String,
        version: u64,
    ) -> Result<(), ServiceError> {
        let durable = self.get(&collection)?;
        self.bridge
            .run(move || durable.create_tag(name, version))
            .await??;
        Ok(())
    }

    /// Branches from a version.
    pub async fn create_branch(
        &self,
        collection: String,
        name: String,
        version: u64,
    ) -> Result<(), ServiceError> {
        let durable = self.get(&collection)?;
        self.bridge
            .run(move || durable.create_branch(name, version))
            .await??;
        Ok(())
    }

    /// Lists a collection's versions (and its current HEAD).
    pub fn list_versions(
        &self,
        collection: &str,
    ) -> Result<(Vec<StdArc<Manifest>>, Option<u64>), ServiceError> {
        let durable = self.get(collection)?;
        Ok((durable.list_versions(), durable.collection().head_version()))
    }

    /// Diffs two versions, returning (added, removed) global ids.
    pub fn diff(
        &self,
        collection: &str,
        from: u64,
        to: u64,
    ) -> Result<(Vec<u64>, Vec<u64>), ServiceError> {
        let durable = self.get(collection)?;
        let diff = durable.diff(from, to)?;
        Ok((diff.added, diff.removed))
    }

    /// Lists every collection with its summary stats (name, dim, metric, count, head),
    /// ordered by name.
    pub fn list_collections(&self) -> Vec<CollectionStats> {
        let mut out: Vec<CollectionStats> = self
            .registry
            .list_all()
            .into_iter()
            .map(|(name, c)| {
                let cfg = c.config();
                CollectionStats {
                    name,
                    dim: cfg.dim,
                    metric: cfg.metric,
                    count: c.len(),
                    head_version: c.head_version(),
                }
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Summary stats for a single collection.
    pub fn collection_stats(&self, collection: &str) -> Result<CollectionStats, ServiceError> {
        let durable = self.get(collection)?;
        let cfg = durable.config();
        Ok(CollectionStats {
            name: cfg.name.clone(),
            dim: cfg.dim,
            metric: cfg.metric,
            count: durable.len(),
            head_version: durable.head_version(),
        })
    }

    /// Drops a collection: removes it from the registry and deletes its data on disk.
    pub fn drop_collection(&self, collection: &str) -> Result<(), ServiceError> {
        if self.registry.remove(collection).is_none() {
            return Err(ServiceError::NotFound(collection.to_owned()));
        }
        let dir = self.collections_dir().join(collection);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(vecvec_core::CoreError::from)?;
        }
        Ok(())
    }

    /// Materializes a page of live points (vectors + payloads), optionally as of a
    /// past version. Returns `(page, total_live_count)`. Backs the table + graph view.
    pub async fn scroll(
        &self,
        collection: String,
        at: Option<VersionSelector>,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<vecvec_core::PointRecord>, usize), ServiceError> {
        let durable = self.get(&collection)?;
        let page = self
            .bridge
            .run(move || durable.scroll(at.as_ref(), offset, limit))
            .await??;
        Ok(page)
    }

    /// Fetches a single live point (vector + payload) by id.
    pub fn get_point(
        &self,
        collection: &str,
        id: u64,
    ) -> Result<Option<vecvec_core::PointRecord>, ServiceError> {
        let durable = self.get(collection)?;
        Ok(durable.get_point(id))
    }

    /// Durably tombstones a point, returning whether it was newly deleted.
    pub async fn delete(&self, collection: String, id: u64) -> Result<bool, ServiceError> {
        let durable = self.get(&collection)?;
        Ok(self.bridge.run(move || durable.delete(id)).await??)
    }
}

/// Flattens scored hits to `(id, score, payload)` tuples, looking up each point's
/// payload only when `include_payloads` is set (otherwise the payload is `None`).
/// Runs inside the blocking bridge so the payload reads stay off the async reactor.
fn scored_with_payloads(
    durable: &DurableCollection,
    results: Vec<ScoredGlobal>,
    include_payloads: bool,
) -> Vec<(u64, f32, Option<Payload>)> {
    results
        .into_iter()
        .map(|s| {
            let payload = if include_payloads {
                durable.payload(s.id)
            } else {
                None
            };
            (s.id.get(), s.score, payload)
        })
        .collect()
}
