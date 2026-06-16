//! A collection: vectors, their segments, and the version DAG.
//!
//! A collection holds one mutable [`AppendableSegment`] plus a *working* set of
//! sealed segments (the live view) and a master registry of every sealed segment
//! ever created (so any past version can be reassembled). **Deletions live here**, as
//! a collection-level [`DeletionVector`], not inside segments — so committing simply
//! freezes a clone of it, giving snapshot isolation.
//!
//! A *commit* snapshots {working segment refs, current deletions} into a [`Manifest`]
//! in the [`VersionStore`]. Time-travel ([`Collection::search_at`]) reassembles a
//! past version's segments + frozen deletions; [`Collection::restore`] is a forward
//! commit re-pointing the working state at an old version. Lock discipline: the
//! appendable `RwLock` and the segment locks are only ever held inside synchronous
//! work (run on the server's rayon pool), never across an `.await`.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};

use crate::distance::Metric;
use crate::error::{CoreError, Result};
use crate::id::{GlobalId, SegmentId};
use crate::index::{BoundedTopK, HnswConfig};
use crate::payload::{Filter, FilterQuery, Payload, PayloadMap};
use crate::segment::{AppendableSegment, SealedSegment, SegmentSet};
use crate::version::policy::Clock;
use crate::version::{
    DeletionVector, Diff, Manifest, RetentionRules, SegmentRef, SystemClock, VersionSelector,
    VersionStore, VersioningPolicy,
};

/// Rules for automatic compaction (merging the working segments into one, with a
/// freshly-built HNSW graph). Triggers are OR-ed: when the working set reaches
/// `max_segments` sealed segments, and/or every `interval_ms`. All `None` = manual
/// compaction only. Evaluated by a background maintenance loop, never on the write
/// path, since compaction rebuilds a graph and is comparatively expensive.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CompactionPolicy {
    /// Compact once the working set holds at least this many sealed segments.
    pub max_segments: Option<usize>,
    /// Compact once this many milliseconds have elapsed since the last compaction
    /// (only when there is more than one working segment to merge).
    pub interval_ms: Option<u64>,
}

impl CompactionPolicy {
    /// A policy that compacts only on explicit request.
    pub fn manual() -> Self {
        Self::default()
    }

    /// Whether any automatic trigger is configured.
    pub fn is_automatic(&self) -> bool {
        self.max_segments.is_some() || self.interval_ms.is_some()
    }
}

/// Static configuration of a collection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CollectionConfig {
    /// The collection name.
    pub name: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// HNSW parameters used when sealing segments.
    pub hnsw: HnswConfig,
    /// Automatic-commit policy.
    #[serde(default)]
    pub versioning: VersioningPolicy,
    /// Automatic-compaction policy.
    #[serde(default)]
    pub compaction: CompactionPolicy,
}

impl CollectionConfig {
    /// Convenience constructor with default HNSW + manual versioning + manual
    /// compaction.
    pub fn new(name: impl Into<String>, dim: usize, metric: Metric) -> Self {
        Self {
            name: name.into(),
            dim,
            metric,
            hnsw: HnswConfig::default(),
            versioning: VersioningPolicy::default(),
            compaction: CompactionPolicy::default(),
        }
    }
}

/// A scored result keyed by collection-global id (the output of collection search).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredGlobal {
    /// The collection-global point id.
    pub id: GlobalId,
    /// The raw metric score.
    pub score: f32,
}

/// A full materialized point: its id, f32 vector, and optional payload. Returned by
/// the explorer scroll / point-fetch APIs (the UI's table + 2D graph view).
#[derive(Debug, Clone)]
pub struct PointRecord {
    /// The collection-global point id.
    pub id: GlobalId,
    /// The stored (normalized for cosine) f32 vector.
    pub vector: Vec<f32>,
    /// The point's JSON payload, if any.
    pub payload: Option<Payload>,
}

/// A collection of vectors served from RAM, with a git-like version DAG.
pub struct Collection {
    config: CollectionConfig,
    appendable: RwLock<AppendableSegment>,
    /// The live (working) segment set; lock-free point-in-time reads.
    working: ArcSwap<SegmentSet>,
    /// Every sealed segment ever created, so any past version can be reassembled.
    all_segments: RwLock<BTreeMap<SegmentId, Arc<SealedSegment>>>,
    /// The live tombstones.
    working_deletions: ArcSwap<DeletionVector>,
    /// Per-point payloads, keyed by global id (collection-level, like deletions).
    payloads: RwLock<PayloadMap>,
    versions: Mutex<VersionStore>,
    next_global_id: AtomicU64,
    next_segment_id: AtomicU64,
}

impl Collection {
    /// Creates an empty collection from `config`.
    pub fn create(config: CollectionConfig) -> Self {
        let appendable = AppendableSegment::new(config.dim, config.metric);
        Self {
            config,
            appendable: RwLock::new(appendable),
            working: ArcSwap::from_pointee(SegmentSet::empty()),
            all_segments: RwLock::new(BTreeMap::new()),
            working_deletions: ArcSwap::from_pointee(DeletionVector::new()),
            payloads: RwLock::new(PayloadMap::new()),
            versions: Mutex::new(VersionStore::new()),
            next_global_id: AtomicU64::new(0),
            next_segment_id: AtomicU64::new(0),
        }
    }

    /// The collection's configuration.
    #[inline]
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// The number of live points across all segments.
    pub fn len(&self) -> usize {
        let rows = self.working.load().total_rows() + self.appendable.read().len();
        rows.saturating_sub(self.working_deletions.load().len() as usize)
    }

    /// Whether the collection has no live points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of sealed segments in the working set.
    pub fn sealed_count(&self) -> usize {
        self.working.load().len()
    }

    /// Allocates the next global id (used by the durability layer before logging).
    pub(crate) fn alloc_global_id(&self) -> GlobalId {
        GlobalId::new(self.next_global_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Appends a vector under a specific global id, advancing the allocator past it.
    /// The single in-memory upsert apply path (live writes and WAL recovery).
    pub fn insert_with_id(&self, id: GlobalId, vector: &[f32]) -> Result<()> {
        self.insert_with_id_and_payload(id, vector, None)
    }

    /// Like [`Collection::insert_with_id`] but also stores a payload.
    pub fn insert_with_id_and_payload(
        &self,
        id: GlobalId,
        vector: &[f32],
        payload: Option<Payload>,
    ) -> Result<()> {
        self.check_dim(vector)?;
        self.appendable.write().append(id, vector);
        self.next_global_id
            .fetch_max(id.get() + 1, Ordering::Relaxed);
        if let Some(p) = payload {
            self.payloads.write().insert(id.get(), p);
        }
        Ok(())
    }

    /// A snapshot copy of the payload map (for checkpoint persistence).
    pub fn payloads_snapshot(&self) -> PayloadMap {
        self.payloads.read().clone()
    }

    /// Replaces the payload map (recovery).
    pub(crate) fn set_payloads(&self, payloads: PayloadMap) {
        *self.payloads.write() = payloads;
    }

    fn check_dim(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dim {
            return Err(CoreError::DimensionMismatch {
                expected: self.config.dim,
                got: vector.len(),
            });
        }
        Ok(())
    }

    /// Inserts a single vector, returning its assigned global id.
    pub fn insert(&self, vector: &[f32]) -> Result<GlobalId> {
        let id = self.alloc_global_id();
        self.insert_with_id(id, vector)?;
        Ok(id)
    }

    /// Inserts a batch of vectors under one write lock, returning their ids.
    pub fn insert_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<GlobalId>> {
        for v in vectors {
            self.check_dim(v)?;
        }
        let mut ids = Vec::with_capacity(vectors.len());
        let mut app = self.appendable.write();
        for v in vectors {
            let id = self.alloc_global_id();
            app.append(id, v);
            ids.push(id);
        }
        self.next_global_id
            .fetch_max(ids.last().map_or(0, |g| g.get() + 1), Ordering::Relaxed);
        Ok(ids)
    }

    fn contains_any(&self, global: GlobalId) -> bool {
        if self.appendable.read().contains(global) {
            return true;
        }
        self.working.load().iter().any(|s| s.contains(global))
    }

    /// Tombstones the point `global` in the live deletion vector. Returns whether it
    /// was newly deleted.
    pub fn delete(&self, global: GlobalId) -> bool {
        if !self.contains_any(global) {
            return false;
        }
        let mut newly = false;
        self.working_deletions.rcu(|current| {
            let mut next = DeletionVector::clone(current);
            newly = next.insert(global);
            next
        });
        newly
    }

    fn merge_finalize(merger: BoundedTopK<GlobalId>) -> Vec<ScoredGlobal> {
        let mut seen = HashSet::new();
        merger
            .into_sorted()
            .into_iter()
            .filter(|(id, _)| seen.insert(*id))
            .map(|(id, score)| ScoredGlobal { id, score })
            .collect()
    }

    /// Returns the best `k` live points for `query`, merging the working sealed set
    /// and the appendable segment, excluding the live deletion vector.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredGlobal>> {
        self.check_dim(query)?;
        let higher = self.config.metric.higher_is_better();
        let deletions = self.working_deletions.load_full();
        let working = self.working.load_full();
        let payloads = self.payloads.read();
        let fq = filter.map(|f| FilterQuery {
            filter: f,
            payloads: &payloads,
        });
        let mut merger = BoundedTopK::<GlobalId>::new(k, higher);
        for seg in working.iter() {
            for (id, score) in seg.search(query, k, &deletions, fq.as_ref()) {
                merger.offer(id, score);
            }
        }
        {
            let app = self.appendable.read();
            for (id, score) in app.search(query, k, &deletions, fq.as_ref()) {
                merger.offer(id, score);
            }
        }
        Ok(Self::merge_finalize(merger))
    }

    /// Time-travel search: returns the best `k` points as of `selector`, using that
    /// version's segment set and frozen deletion vector (snapshot isolation).
    pub fn search_at(
        &self,
        selector: &VersionSelector,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredGlobal>> {
        self.check_dim(query)?;
        let (segment_ids, deletions) = {
            let versions = self.versions.lock();
            let version = versions
                .resolve(selector)
                .ok_or_else(|| CoreError::Version {
                    detail: format!("unresolved selector {selector:?}"),
                })?;
            let manifest = versions.get(version).expect("resolved version exists");
            (
                manifest.segments.iter().map(|s| s.id).collect::<Vec<u64>>(),
                manifest.deletions.clone(),
            )
        };
        let higher = self.config.metric.higher_is_better();
        let all = self.all_segments.read();
        let payloads = self.payloads.read();
        let fq = filter.map(|f| FilterQuery {
            filter: f,
            payloads: &payloads,
        });
        let mut merger = BoundedTopK::<GlobalId>::new(k, higher);
        for sid in segment_ids {
            if let Some(seg) = all.get(&SegmentId::new(sid)) {
                for (id, score) in seg.search(query, k, &deletions, fq.as_ref()) {
                    merger.offer(id, score);
                }
            }
        }
        Ok(Self::merge_finalize(merger))
    }

    /// Seals the current appendable segment into the working set + master registry.
    /// Returns the new segment id, or `None` if the appendable was empty.
    pub fn seal(&self) -> Option<SegmentId> {
        let mut app = self.appendable.write();
        if app.is_empty() {
            return None;
        }
        let seg_id = SegmentId::new(self.next_segment_id.fetch_add(1, Ordering::Relaxed));
        let frozen = std::mem::replace(
            &mut *app,
            AppendableSegment::new(self.config.dim, self.config.metric),
        );
        drop(app);
        let sealed = Arc::new(frozen.seal(seg_id, self.config.hnsw));
        self.all_segments.write().insert(seg_id, sealed.clone());
        self.working
            .rcu(|current| Arc::new(current.with_appended(sealed.clone())));
        Some(seg_id)
    }

    /// Commits the working state as a new version. Seals the appendable first so the
    /// version references only immutable segments.
    pub fn commit(
        &self,
        trigger: impl Into<String>,
        message: Option<String>,
        tag: Option<String>,
    ) -> Result<u64> {
        self.seal();
        let working = self.working.load_full();
        let segments: Vec<SegmentRef> = working
            .iter()
            .filter_map(|s| {
                s.global_id_range().map(|(lo, hi)| SegmentRef {
                    id: s.id().get(),
                    id_lo: lo,
                    id_hi: hi,
                    count: s.len() as u64,
                })
            })
            .collect();
        let deletions = DeletionVector::clone(&self.working_deletions.load_full());
        let now = SystemClock.now_ms();
        let mut versions = self.versions.lock();
        let manifest = versions.commit(trigger, message, tag, segments, deletions, now);
        Ok(manifest.version)
    }

    /// Restores the working state to `version`: a forward commit re-pointing the
    /// working segment set + deletions at the old version (history is preserved).
    /// Discards uncommitted appendable writes.
    pub fn restore(&self, version: u64) -> Result<u64> {
        let (segment_ids, deletions) = {
            let versions = self.versions.lock();
            let manifest = versions.get(version).ok_or_else(|| CoreError::Version {
                detail: format!("no such version {version}"),
            })?;
            (
                manifest.segments.iter().map(|s| s.id).collect::<Vec<u64>>(),
                manifest.deletions.clone(),
            )
        };
        {
            let all = self.all_segments.read();
            let segs: Vec<Arc<SealedSegment>> = segment_ids
                .iter()
                .filter_map(|id| all.get(&SegmentId::new(*id)).cloned())
                .collect();
            self.working.store(Arc::new(SegmentSet::from_sealed(segs)));
        }
        self.working_deletions.store(Arc::new(deletions));
        *self.appendable.write() = AppendableSegment::new(self.config.dim, self.config.metric);
        self.commit("restore", Some(format!("restore of v{version}")), None)
    }

    /// The live f32 vector for `global`, if present (appendable or working segments).
    pub fn get_vector(&self, global: GlobalId) -> Option<Vec<f32>> {
        if let Some(v) = self.appendable.read().vector_of(global) {
            return Some(v.to_vec());
        }
        self.working
            .load()
            .iter()
            .find_map(|s| s.vector_of(global).map(<[f32]>::to_vec))
    }

    /// A full live point (vector + payload) by id, or `None` if missing or tombstoned.
    pub fn get_point(&self, global: GlobalId) -> Option<PointRecord> {
        if self.working_deletions.load().contains(global) {
            return None;
        }
        let vector = self.get_vector(global)?;
        let payload = self.payloads.read().get(&global.get()).cloned();
        Some(PointRecord {
            id: global,
            vector,
            payload,
        })
    }

    /// The JSON payload for a single point by id, or `None` if it has none.
    pub fn payload(&self, global: GlobalId) -> Option<Payload> {
        self.payloads.read().get(&global.get()).cloned()
    }

    /// Materializes a page of live points — every row across the working sealed set
    /// and the appendable segment, minus tombstones, joined with payloads and ordered
    /// by ascending global id. With `at`, reads as of a past version (its frozen
    /// segment set + deletion vector). Returns `(page, total_live_count)`.
    ///
    /// This is the read path behind the explorer table and the 2D graph view, which
    /// need the raw vectors that search (`{id, score}` only) does not surface.
    pub fn scroll(
        &self,
        at: Option<&VersionSelector>,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<PointRecord>, usize)> {
        let payloads = self.payloads.read();
        let mut rows: Vec<(GlobalId, Vec<f32>)> = Vec::new();

        match at {
            None => {
                let deletions = self.working_deletions.load_full();
                for seg in self.working.load_full().iter() {
                    for (id, v) in seg.iter_points() {
                        if !deletions.contains(id) {
                            rows.push((id, v.to_vec()));
                        }
                    }
                }
                let app = self.appendable.read();
                for (id, v) in app.iter_points() {
                    if !deletions.contains(id) {
                        rows.push((id, v.to_vec()));
                    }
                }
            }
            Some(selector) => {
                let (segment_ids, deletions) = {
                    let versions = self.versions.lock();
                    let version = versions
                        .resolve(selector)
                        .ok_or_else(|| CoreError::Version {
                            detail: format!("unresolved selector {selector:?}"),
                        })?;
                    let manifest = versions.get(version).expect("resolved version exists");
                    (
                        manifest.segments.iter().map(|s| s.id).collect::<Vec<u64>>(),
                        manifest.deletions.clone(),
                    )
                };
                let all = self.all_segments.read();
                for sid in segment_ids {
                    if let Some(seg) = all.get(&SegmentId::new(sid)) {
                        for (id, v) in seg.iter_points() {
                            if !deletions.contains(id) {
                                rows.push((id, v.to_vec()));
                            }
                        }
                    }
                }
            }
        }

        rows.sort_by_key(|(g, _)| g.get());
        let total = rows.len();
        let page = rows
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(id, vector)| PointRecord {
                payload: payloads.get(&id.get()).cloned(),
                id,
                vector,
            })
            .collect();
        Ok((page, total))
    }

    /// Recommend-by-example: builds a query vector `mean(positives) - mean(negatives)`
    /// from the example points' vectors and searches, excluding the positive
    /// examples from the results. Requires at least one resolvable example.
    pub fn recommend(
        &self,
        positive: &[GlobalId],
        negative: &[GlobalId],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredGlobal>> {
        let dim = self.config.dim;
        let mut query = vec![0.0f32; dim];
        let mut np = 0usize;
        for &id in positive {
            if let Some(v) = self.get_vector(id) {
                for (q, x) in query.iter_mut().zip(&v) {
                    *q += x;
                }
                np += 1;
            }
        }
        let mut neg = vec![0.0f32; dim];
        let mut nn = 0usize;
        for &id in negative {
            if let Some(v) = self.get_vector(id) {
                for (q, x) in neg.iter_mut().zip(&v) {
                    *q += x;
                }
                nn += 1;
            }
        }
        if np + nn == 0 {
            return Err(CoreError::Version {
                detail: "recommend: no resolvable example points".into(),
            });
        }
        if np > 0 {
            for q in &mut query {
                *q /= np as f32;
            }
        }
        if nn > 0 {
            for (q, n) in query.iter_mut().zip(&neg) {
                *q -= n / nn as f32;
            }
        }
        if self.config.metric.requires_normalization() {
            crate::distance::l2_normalize(&mut query);
        }

        // Over-fetch so we can drop the positive examples, then trim to k.
        let exclude: HashSet<u64> = positive.iter().map(|g| g.get()).collect();
        let mut results = self.search(&query, k + exclude.len(), filter)?;
        results.retain(|s| !exclude.contains(&s.id.get()));
        results.truncate(k);
        Ok(results)
    }

    /// Merges all working sealed segments into one, cutting search fan-out. The old
    /// segments stay in the master registry (so versions referencing them remain
    /// queryable) until GC. Returns the merged segment id, or `None` if there was
    /// nothing to merge.
    ///
    /// Assumes the working segments form one contiguous global-id range (true unless
    /// a prior restore re-pointed at a non-contiguous subset).
    pub fn compact(&self) -> Option<SegmentId> {
        let working = self.working.load_full();
        if working.len() <= 1 {
            return None;
        }
        let mut points: Vec<(GlobalId, Vec<f32>)> = Vec::new();
        for seg in working.iter() {
            for (g, v) in seg.iter_points() {
                points.push((g, v.to_vec()));
            }
        }
        points.sort_by_key(|(g, _)| g.get());

        let seg_id = SegmentId::new(self.next_segment_id.fetch_add(1, Ordering::Relaxed));
        let mut merged = AppendableSegment::new(self.config.dim, self.config.metric);
        for (g, v) in &points {
            merged.append(*g, v);
        }
        let sealed = Arc::new(merged.seal(seg_id, self.config.hnsw));
        self.all_segments.write().insert(seg_id, sealed.clone());
        self.working
            .store(Arc::new(SegmentSet::from_sealed(vec![sealed])));
        Some(seg_id)
    }

    /// Runs a GC pass: drops versions not matched by `retention` and removes any
    /// segments no longer referenced by a retained version (but never a live working
    /// segment). Returns the dropped segment ids so the caller can delete their files.
    pub fn gc(&self, retention: &RetentionRules) -> Vec<SegmentId> {
        let report = self.versions.lock().gc(retention);
        let working_ids: HashSet<u64> = self.working.load().iter().map(|s| s.id().get()).collect();
        let mut all = self.all_segments.write();
        let mut dropped = Vec::new();
        for sid in report.orphan_segments {
            if !working_ids.contains(&sid) && all.remove(&SegmentId::new(sid)).is_some() {
                dropped.push(SegmentId::new(sid));
            }
        }
        dropped
    }

    /// Diffs two versions' live id sets.
    pub fn diff(&self, from: u64, to: u64) -> Result<Diff> {
        self.versions
            .lock()
            .diff(from, to)
            .map_err(|e| CoreError::Version {
                detail: e.to_string(),
            })
    }

    /// Creates or moves a tag to a version.
    pub fn create_tag(&self, name: impl Into<String>, version: u64) -> Result<()> {
        self.versions
            .lock()
            .set_tag(name, version)
            .map_err(|e| CoreError::Version {
                detail: e.to_string(),
            })
    }

    /// Creates or moves a branch to a version.
    pub fn create_branch(&self, name: impl Into<String>, version: u64) -> Result<()> {
        self.versions
            .lock()
            .set_branch(name, version)
            .map_err(|e| CoreError::Version {
                detail: e.to_string(),
            })
    }

    /// All committed versions, oldest first.
    pub fn list_versions(&self) -> Vec<Arc<Manifest>> {
        self.versions.lock().list()
    }

    /// The current `HEAD` version, if any.
    pub fn head_version(&self) -> Option<u64> {
        self.versions.lock().head()
    }

    // ---- recovery / durability hooks ----

    /// A point-in-time snapshot of the working segment set.
    pub fn sealed_snapshot(&self) -> Arc<SegmentSet> {
        self.working.load_full()
    }

    /// The live deletion vector (for checkpoint persistence).
    pub fn deletions_snapshot(&self) -> Arc<DeletionVector> {
        self.working_deletions.load_full()
    }

    /// Installs recovered segments: `all` go into the master registry (so any past
    /// version is queryable), and those in `working_ids` form the live working set.
    pub(crate) fn install_recovered(&self, all: Vec<Arc<SealedSegment>>, working_ids: &[u64]) {
        let working_set: HashSet<u64> = working_ids.iter().copied().collect();
        let mut map = self.all_segments.write();
        for seg in &all {
            map.insert(seg.id(), seg.clone());
        }
        drop(map);
        let working: Vec<Arc<SealedSegment>> = all
            .into_iter()
            .filter(|s| working_set.contains(&s.id().get()))
            .collect();
        self.working
            .store(Arc::new(SegmentSet::from_sealed(working)));
    }

    /// A serializable snapshot of the version DAG (for durable persistence).
    pub fn version_snapshot(&self) -> crate::version::VersionStoreSnapshot {
        self.versions.lock().snapshot()
    }

    /// Loads a persisted version DAG (recovery).
    pub(crate) fn load_version_snapshot(&self, snapshot: crate::version::VersionStoreSnapshot) {
        *self.versions.lock() = VersionStore::from_snapshot(snapshot);
    }

    /// A sealed segment by id (from the master registry).
    pub(crate) fn get_segment(&self, id: SegmentId) -> Option<Arc<SealedSegment>> {
        self.all_segments.read().get(&id).cloned()
    }

    /// All segment ids that must be persisted to keep every version queryable: the
    /// working set plus everything referenced by any committed version.
    pub fn segment_ids_to_persist(&self) -> Vec<SegmentId> {
        let mut ids: std::collections::BTreeSet<u64> =
            self.working.load().iter().map(|s| s.id().get()).collect();
        ids.extend(self.versions.lock().all_referenced_segments());
        ids.into_iter().map(SegmentId::new).collect()
    }

    /// The ids of the live working set segments.
    pub fn working_segment_ids(&self) -> Vec<u64> {
        self.working.load().iter().map(|s| s.id().get()).collect()
    }

    /// Sets the live deletion vector (recovery).
    pub(crate) fn set_deletions(&self, deletions: DeletionVector) {
        self.working_deletions.store(Arc::new(deletions));
    }

    /// Sets the id allocators (recovery).
    pub(crate) fn set_allocators(&self, next_global: u64, next_segment: u64) {
        self.next_global_id.store(next_global, Ordering::Relaxed);
        self.next_segment_id.store(next_segment, Ordering::Relaxed);
    }

    /// The next global id that would be allocated.
    pub fn next_global_id_value(&self) -> u64 {
        self.next_global_id.load(Ordering::Relaxed)
    }

    /// The next segment id that would be allocated.
    pub fn next_segment_id_value(&self) -> u64 {
        self.next_segment_id.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::VersioningPolicy;

    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    fn config(dim: usize) -> CollectionConfig {
        CollectionConfig {
            name: "c".into(),
            dim,
            metric: Metric::Dot,
            hnsw: HnswConfig::default(),
            versioning: VersioningPolicy::manual(),
            compaction: CompactionPolicy::manual(),
        }
    }

    #[test]
    fn insert_search_delete_len() {
        let dim = 8;
        let col = Collection::create(config(dim));
        let ids = col
            .insert_batch(&(0..20).map(|i| vec_of(dim, i + 1)).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(col.len(), 20);
        assert!(col.delete(ids[0]));
        assert!(!col.delete(ids[0]));
        assert_eq!(col.len(), 19);
        let got = col.search(&vec_of(dim, 99), 25, None).unwrap();
        assert!(got.iter().all(|s| s.id != ids[0]));
        assert_eq!(got.len(), 19);
    }

    /// The headline differentiator test: a delete after a commit must NOT change what
    /// the committed version sees (snapshot isolation).
    #[test]
    fn time_travel_snapshot_isolation() {
        let dim = 8;
        let col = Collection::create(config(dim));
        let ids = col
            .insert_batch(&(0..30).map(|i| vec_of(dim, i + 1)).collect::<Vec<_>>())
            .unwrap();
        let v1 = col.commit("manual", None, None).unwrap();
        assert_eq!(col.len(), 30);

        // Delete half the points *after* the commit.
        for &id in &ids[..15] {
            col.delete(id);
        }
        assert_eq!(col.len(), 15);

        // Live search excludes deleted points...
        let live = col.search(&vec_of(dim, 1), 30, None).unwrap();
        assert_eq!(live.len(), 15);
        assert!(live.iter().all(|s| !ids[..15].contains(&s.id)));

        // ...but time-travel to v1 still sees all 30 (frozen deletion vector).
        let at_v1 = col
            .search_at(&VersionSelector::Version(v1), &vec_of(dim, 1), 40, None)
            .unwrap();
        assert_eq!(at_v1.len(), 30);
    }

    #[test]
    fn filtered_search_matches_reference() {
        use crate::payload::{Condition, Filter};
        let dim = 16;
        let n = 400;
        let col = Collection::create(config(dim));
        // Insert with a "bucket" payload field (10 buckets) and seal half.
        for i in 0..n {
            let id = col.insert(&vec_of(dim, i + 1)).unwrap();
            col.payloads
                .write()
                .insert(id.get(), serde_json::json!({ "bucket": i % 10 }));
        }
        col.seal(); // half-ish into a sealed HNSW segment, rest appendable
        for i in n..(n + 100) {
            let id = col.insert(&vec_of(dim, i + 1)).unwrap();
            col.payloads
                .write()
                .insert(id.get(), serde_json::json!({ "bucket": i % 10 }));
        }

        // Filter to a single, highly-selective bucket (~10% of points): exercises the
        // exact-scan fallback in sealed segments.
        let filter = Filter {
            must: vec![Condition {
                key: "bucket".into(),
                r#match: Some(serde_json::json!(3)),
                range: None,
            }],
            ..Default::default()
        };
        let query = vec_of(dim, 99_999);
        let got = col.search(&query, 10, Some(&filter)).unwrap();
        // Every result is in bucket 3.
        let payloads = col.payloads.read();
        assert!(
            got.iter()
                .all(|s| payloads.get(&s.id.get()) == Some(&serde_json::json!({ "bucket": 3 })))
        );
        // And we found a full page (bucket 3 has ~50 points, plenty for k=10).
        assert_eq!(got.len(), 10);
    }

    #[test]
    fn recommend_by_example() {
        let dim = 16;
        let col = Collection::create(CollectionConfig::new("c", dim, Metric::Cosine));
        // Two clusters: ids 0..20 around seed family A, 20..40 around B.
        for i in 0..20 {
            col.insert(&vec_of(dim, i + 1)).unwrap();
        }
        for i in 0..20 {
            col.insert(&vec_of(dim, 10_000 + i)).unwrap();
        }
        // Recommend by a single example == search by that example's own vector,
        // excluding the example itself.
        let pos = [GlobalId::new(3)];
        let got = col.recommend(&pos, &[], 5, None).unwrap();
        assert_eq!(got.len(), 5);
        assert!(got.iter().all(|s| s.id != GlobalId::new(3)));

        let v3 = col.get_vector(GlobalId::new(3)).unwrap();
        let direct: HashSet<u64> = col
            .search(&v3, 6, None)
            .unwrap()
            .into_iter()
            .filter(|s| s.id != GlobalId::new(3))
            .take(5)
            .map(|s| s.id.get())
            .collect();
        let rec: HashSet<u64> = got.iter().map(|s| s.id.get()).collect();
        // Equal as sets (re-normalization can flip near-ties in ordering).
        assert_eq!(rec, direct);

        // No resolvable examples -> error.
        assert!(
            col.recommend(&[GlobalId::new(99_999)], &[], 5, None)
                .is_err()
        );
    }

    #[test]
    fn compaction_merges_segments() {
        let dim = 16;
        let col = Collection::create(config(dim));
        for batch in 0..3 {
            col.insert_batch(
                &(0..30)
                    .map(|i| vec_of(dim, batch * 100 + i + 1))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            col.seal();
        }
        assert_eq!(col.sealed_count(), 3);
        assert_eq!(col.len(), 90);

        let merged = col.compact();
        assert!(merged.is_some());
        assert_eq!(col.sealed_count(), 1);
        assert_eq!(col.len(), 90);
        // Search still returns a full page of valid results.
        let got = col.search(&vec_of(dim, 9_999), 10, None).unwrap();
        assert_eq!(got.len(), 10);
    }

    #[test]
    fn gc_drops_orphaned_segments() {
        let dim = 8;
        let col = Collection::create(config(dim));
        col.insert_batch(&(0..30).map(|i| vec_of(dim, i + 1)).collect::<Vec<_>>())
            .unwrap();
        let v0 = col.commit("manual", None, None).unwrap();
        col.insert_batch(&(0..30).map(|i| vec_of(dim, 100 + i)).collect::<Vec<_>>())
            .unwrap();
        col.commit("manual", None, None).unwrap();

        // Compact the two segments into one, then commit a version referencing it.
        col.compact();
        col.commit("manual", None, None).unwrap();

        // Keep only the latest version: the two pre-merge segments become orphans.
        let dropped = col.gc(&RetentionRules {
            keep_last: Some(1),
            ..Default::default()
        });
        assert_eq!(dropped.len(), 2);
        assert_eq!(col.len(), 60);
        // The dropped version is no longer queryable; live search still works.
        assert!(
            col.search_at(&VersionSelector::Version(v0), &vec_of(dim, 1), 5, None)
                .is_err()
        );
        assert_eq!(col.search(&vec_of(dim, 1), 10, None).unwrap().len(), 10);
    }

    #[test]
    fn diff_branch_restore() {
        let dim = 8;
        let col = Collection::create(config(dim));
        let ids = col
            .insert_batch(&(0..20).map(|i| vec_of(dim, i + 1)).collect::<Vec<_>>())
            .unwrap();
        let v0 = col.commit("manual", None, None).unwrap();

        // Delete some, add some, commit again.
        col.delete(ids[0]);
        col.delete(ids[1]);
        col.insert_batch(&(0..5).map(|i| vec_of(dim, 100 + i)).collect::<Vec<_>>())
            .unwrap();
        let v1 = col.commit("manual", None, None).unwrap();

        // Diff v0 -> v1: 5 added, 2 removed.
        let d = col.diff(v0, v1).unwrap();
        assert_eq!(d.added.len(), 5);
        assert_eq!(d.removed.len(), 2);

        // Branch + tag the versions.
        col.create_branch("main", v1).unwrap();
        col.create_tag("baseline", v0).unwrap();
        assert_eq!(
            col.search_at(
                &VersionSelector::Tag("baseline".into()),
                &vec_of(dim, 1),
                30,
                None
            )
            .unwrap()
            .len(),
            20
        );

        // Restore to v0: a forward commit; live state reflects v0 (20 points).
        let v2 = col.restore(v0).unwrap();
        assert!(v2 > v1);
        assert_eq!(col.len(), 20);
        // History preserved: v1 still queryable with its 23 live points.
        assert_eq!(
            col.search_at(&VersionSelector::Version(v1), &vec_of(dim, 1), 40, None)
                .unwrap()
                .len(),
            23
        );
    }

    #[test]
    fn diff_after_compact_over_deletes_is_exact() {
        // Compaction merges the working segments; deletions stay collection-level. The
        // range-based `live_ids` reconstruction must still report the SAME live set
        // before and after compaction even with interior ids deleted — i.e. it must
        // not resurrect the deleted interior ids as phantom adds.
        let dim = 8;
        let col = Collection::create(config(dim));
        let ids1 = col
            .insert_batch(&(0..15).map(|i| vec_of(dim, i + 1)).collect::<Vec<_>>())
            .unwrap();
        col.seal();
        let ids2 = col
            .insert_batch(&(0..15).map(|i| vec_of(dim, 100 + i)).collect::<Vec<_>>())
            .unwrap();
        col.seal();
        assert_eq!(col.sealed_count(), 2);

        // Delete every other id (interior deletions inside both segment ranges).
        let all_ids: Vec<GlobalId> = ids1.iter().chain(&ids2).copied().collect();
        for (i, &id) in all_ids.iter().enumerate() {
            if i % 2 == 0 {
                assert!(col.delete(id));
            }
        }
        assert_eq!(col.len(), 15);
        let v_prev = col.commit("manual", None, None).unwrap();

        // Compact the two segments into one (its global-id range spans [min,max],
        // including the deleted interior ids), then commit.
        assert!(col.compact().is_some());
        assert_eq!(col.sealed_count(), 1);
        let v_c = col.commit("manual", None, None).unwrap();

        // Same live set across the compaction boundary: no phantom adds/removes.
        let d = col.diff(v_prev, v_c).unwrap();
        assert!(
            d.added.is_empty(),
            "phantom adds after compact-over-deletes: {:?}",
            d.added
        );
        assert!(
            d.removed.is_empty(),
            "phantom removes after compact-over-deletes: {:?}",
            d.removed
        );
        assert_eq!(col.len(), 15);
    }

    #[test]
    fn gc_never_drops_live_working_segment() {
        // After a restore re-points the working set at an OLD segment, GC must never
        // drop that live working segment (live search depends on it), even while
        // dropping the versions that introduced the now-unused segments.
        let dim = 8;
        let col = Collection::create(config(dim));
        // v0 references segA.
        col.insert_batch(&(0..20).map(|i| vec_of(dim, i + 1)).collect::<Vec<_>>())
            .unwrap();
        let v0 = col.commit("manual", None, None).unwrap();
        let seg_a = col.working_segment_ids();
        assert_eq!(seg_a.len(), 1);

        // v1 adds new data: working set becomes {segA, segB}.
        col.insert_batch(&(0..20).map(|i| vec_of(dim, 100 + i)).collect::<Vec<_>>())
            .unwrap();
        col.commit("manual", None, None).unwrap();
        assert_eq!(col.sealed_count(), 2);

        // Restore to v0: working re-points back at segA alone.
        col.restore(v0).unwrap();
        assert_eq!(col.working_segment_ids(), seg_a);

        // GC keeping only the latest version. segA is the live working set: it must
        // survive even though the older versions referencing it are dropped.
        let dropped = col.gc(&RetentionRules {
            keep_last: Some(1),
            ..Default::default()
        });
        assert!(
            !dropped.contains(&SegmentId::new(seg_a[0])),
            "GC dropped the live working segment {}",
            seg_a[0]
        );
        // Live search still returns segA's points after GC.
        assert_eq!(col.len(), 20);
        assert_eq!(col.search(&vec_of(dim, 1), 10, None).unwrap().len(), 10);
    }

    #[test]
    fn recommend_with_negatives_steers_toward_positive_cluster() {
        // Exercises the mean(positives) - mean(negatives) construction with a real
        // negative example, plus the negatives-only (np==0, nn>0) path. A sign error
        // or wrong divisor in the negative term would steer the query the wrong way.
        let dim = 8;
        let col = Collection::create(CollectionConfig::new("c", dim, Metric::Cosine));
        // Cluster A near the +x axis, cluster B near the +y axis (orthogonal).
        let mut a_ids = Vec::new();
        let mut b_ids = Vec::new();
        for i in 0..15 {
            let mut a = vec![0.0f32; dim];
            a[0] = 1.0;
            a[2] = 0.01 * i as f32; // tiny jitter, stays near the A axis
            a_ids.push(col.insert(&a).unwrap());
        }
        for i in 0..15 {
            let mut b = vec![0.0f32; dim];
            b[1] = 1.0;
            b[3] = 0.01 * i as f32;
            b_ids.push(col.insert(&b).unwrap());
        }
        let a_set: HashSet<u64> = a_ids.iter().map(|g| g.get()).collect();

        // positive A, negative B: query mean(A) - mean(B) points toward cluster A.
        let got = col.recommend(&[a_ids[0]], &[b_ids[0]], 5, None).unwrap();
        assert_eq!(got.len(), 5);
        assert!(
            got.iter().all(|s| a_set.contains(&s.id.get())),
            "negative steering should keep results in cluster A, got {:?}",
            got.iter().map(|s| s.id.get()).collect::<Vec<_>>()
        );
        assert!(got.iter().all(|s| s.id != b_ids[0]));

        // Negatives-only path (np==0, nn>0): query = -mean(B), steering away from B
        // toward cluster A. Must succeed and exclude nothing (no positives).
        let neg_only = col.recommend(&[], &[b_ids[0]], 5, None).unwrap();
        assert_eq!(neg_only.len(), 5);
        assert!(
            neg_only.iter().all(|s| a_set.contains(&s.id.get())),
            "negatives-only should steer to cluster A (far side of B)"
        );
    }
}
