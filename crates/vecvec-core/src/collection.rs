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
use crate::index::{BoundedTopK, FilterContext, HnswConfig};
use crate::segment::{AppendableSegment, SealedSegment, SegmentSet};
use crate::version::policy::Clock;
use crate::version::{
    DeletionVector, Diff, Manifest, SegmentRef, SystemClock, VersionSelector, VersionStore,
    VersioningPolicy,
};

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
}

impl CollectionConfig {
    /// Convenience constructor with default HNSW + manual versioning.
    pub fn new(name: impl Into<String>, dim: usize, metric: Metric) -> Self {
        Self {
            name: name.into(),
            dim,
            metric,
            hnsw: HnswConfig::default(),
            versioning: VersioningPolicy::default(),
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
        self.check_dim(vector)?;
        self.appendable.write().append(id, vector);
        self.next_global_id
            .fetch_max(id.get() + 1, Ordering::Relaxed);
        Ok(())
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
        filter: Option<&dyn FilterContext>,
    ) -> Result<Vec<ScoredGlobal>> {
        self.check_dim(query)?;
        let higher = self.config.metric.higher_is_better();
        let deletions = self.working_deletions.load_full();
        let working = self.working.load_full();
        let mut merger = BoundedTopK::<GlobalId>::new(k, higher);
        for seg in working.iter() {
            for (id, score) in seg.search(query, k, &deletions, filter) {
                merger.offer(id, score);
            }
        }
        {
            let app = self.appendable.read();
            for (id, score) in app.search(query, k, &deletions, filter) {
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
        filter: Option<&dyn FilterContext>,
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
        let mut merger = BoundedTopK::<GlobalId>::new(k, higher);
        for sid in segment_ids {
            if let Some(seg) = all.get(&SegmentId::new(sid)) {
                for (id, score) in seg.search(query, k, &deletions, filter) {
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

    /// Installs a recovered set of sealed segments into both the master registry and
    /// the working set.
    pub(crate) fn install_sealed(&self, segments: Vec<Arc<SealedSegment>>) {
        let mut all = self.all_segments.write();
        for seg in &segments {
            all.insert(seg.id(), seg.clone());
        }
        drop(all);
        self.working
            .store(Arc::new(SegmentSet::from_sealed(segments)));
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
}
