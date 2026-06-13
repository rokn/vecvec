//! A collection: the top-level unit of vectors + their segments.
//!
//! At M3 a collection is one mutable [`AppendableSegment`] behind a lock plus a
//! [`SegmentSet`] of sealed segments behind an `ArcSwap`, with a monotonic
//! global-id allocator. Inserts append to the mutable segment; search fans out
//! across the lock-free sealed snapshot and the appendable, merging a global
//! top-k. Versioning, WAL durability, payload, and the full query model layer onto
//! this in later milestones.
//!
//! The lock discipline matters for the server (M3b): the appendable `RwLock` is
//! only ever held *inside* synchronous work (run on the rayon pool), never across
//! an `.await`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::distance::Metric;
use crate::error::{CoreError, Result};
use crate::id::{GlobalId, SegmentId};
use crate::index::{BoundedTopK, FilterContext};
use crate::segment::{AppendableSegment, SegmentSet};

/// Static configuration of a collection.
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    /// The collection name.
    pub name: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
}

impl CollectionConfig {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, dim: usize, metric: Metric) -> Self {
        Self {
            name: name.into(),
            dim,
            metric,
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

/// A collection of vectors served from RAM.
pub struct Collection {
    config: CollectionConfig,
    appendable: RwLock<AppendableSegment>,
    sealed: ArcSwap<SegmentSet>,
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
            sealed: ArcSwap::from_pointee(SegmentSet::empty()),
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
        self.sealed.load().total_live() + self.appendable.read().live_len()
    }

    /// Whether the collection has no live points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of sealed segments.
    pub fn sealed_count(&self) -> usize {
        self.sealed.load().len()
    }

    fn alloc_global_id(&self) -> GlobalId {
        GlobalId::new(self.next_global_id.fetch_add(1, Ordering::Relaxed))
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
        self.check_dim(vector)?;
        let id = self.alloc_global_id();
        self.appendable.write().append(id, vector);
        Ok(id)
    }

    /// Inserts a batch of vectors under one write lock, returning their ids in order.
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
        Ok(ids)
    }

    /// Tombstones the point with id `global`, wherever it lives. Returns whether it
    /// was newly deleted.
    pub fn delete(&self, global: GlobalId) -> bool {
        {
            let app = self.appendable.read();
            if app.contains(global) {
                return app.delete_global(global);
            }
        }
        for seg in self.sealed.load().iter() {
            if seg.contains(global) {
                return seg.delete_global(global);
            }
        }
        false
    }

    /// Returns the best `k` live points for `query`, ordered best-first, merging
    /// across the lock-free sealed snapshot and the appendable segment.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&dyn FilterContext>,
    ) -> Result<Vec<ScoredGlobal>> {
        self.check_dim(query)?;
        let higher = self.config.metric.higher_is_better();
        let mut merger = BoundedTopK::<GlobalId>::new(k, higher);

        // Sealed segments: a single atomic load gives a consistent point-in-time view.
        let sealed = self.sealed.load_full();
        for seg in sealed.iter() {
            for (id, score) in seg.search(query, k, filter) {
                merger.offer(id, score);
            }
        }
        // Appendable segment.
        {
            let app = self.appendable.read();
            for (id, score) in app.search(query, k, filter) {
                merger.offer(id, score);
            }
        }

        Ok(merger
            .into_sorted()
            .into_iter()
            .map(|(id, score)| ScoredGlobal { id, score })
            .collect())
    }

    /// Seals the current appendable segment into an immutable sealed segment and
    /// starts a fresh appendable. Returns the new segment's id, or `None` if the
    /// appendable was empty. (M5 makes this build an HNSW + quantize + persist; here
    /// it just freezes the flat segment.)
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

        let sealed_segment = Arc::new(frozen.seal(seg_id));
        self.sealed
            .rcu(|current| Arc::new(current.with_appended(sealed_segment.clone())));
        Some(seg_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute_force_topk;
    use crate::vector::VectorStorage;

    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    /// A flat oracle over the exact same vectors the collection holds, keyed by the
    /// global ids it assigned (which equal insertion order here).
    fn oracle(dim: usize, metric: Metric, n: usize, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut storage = VectorStorage::new(dim, metric);
        for i in 0..n {
            storage.push(&vec_of(dim, i as u32 + 1));
        }
        let kernel = crate::distance::DistanceKernel::new(metric, dim);
        brute_force_topk(&storage, &kernel, query, k, None, None)
            .into_iter()
            .map(|sp| (sp.id.get() as u64, sp.score))
            .collect()
    }

    #[test]
    fn insert_and_search_matches_oracle() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            let dim = 32;
            let n = 300;
            let col = Collection::create(CollectionConfig::new("c", dim, metric));
            let batch: Vec<Vec<f32>> = (0..n).map(|i| vec_of(dim, i as u32 + 1)).collect();
            let ids = col.insert_batch(&batch).unwrap();
            assert_eq!(ids.len(), n);
            assert_eq!(col.len(), n);

            let query = vec_of(dim, 7777);
            let got = col.search(&query, 10, None).unwrap();
            let want = oracle(dim, metric, n, &query, 10);
            let got_pairs: Vec<(u64, f32)> = got.iter().map(|s| (s.id.get(), s.score)).collect();
            assert_eq!(got_pairs, want, "metric={metric}");
        }
    }

    /// Sealing part of the data exercises the cross-segment merge: results over
    /// (sealed ∪ appendable) must equal the oracle over all vectors.
    #[test]
    fn cross_segment_search_matches_oracle() {
        let dim = 16;
        let metric = Metric::Dot;
        let n_first = 120;
        let n_second = 80;
        let col = Collection::create(CollectionConfig::new("c", dim, metric));

        let first: Vec<Vec<f32>> = (0..n_first).map(|i| vec_of(dim, i as u32 + 1)).collect();
        col.insert_batch(&first).unwrap();
        let seg = col.seal();
        assert!(seg.is_some());
        assert_eq!(col.sealed_count(), 1);

        let second: Vec<Vec<f32>> = (0..n_second)
            .map(|i| vec_of(dim, (n_first + i) as u32 + 1))
            .collect();
        col.insert_batch(&second).unwrap();
        assert_eq!(col.len(), n_first + n_second);

        let query = vec_of(dim, 5555);
        let got = col.search(&query, 15, None).unwrap();
        let want = oracle(dim, metric, n_first + n_second, &query, 15);
        let got_pairs: Vec<(u64, f32)> = got.iter().map(|s| (s.id.get(), s.score)).collect();
        assert_eq!(got_pairs, want);
    }

    #[test]
    fn delete_across_segments() {
        let dim = 8;
        let col = Collection::create(CollectionConfig::new("c", dim, Metric::Dot));
        let ids = col
            .insert_batch(&(0..10).map(|i| vec_of(dim, i + 1)).collect::<Vec<_>>())
            .unwrap();
        col.seal();
        col.insert(&vec_of(dim, 100)).unwrap();
        assert_eq!(col.len(), 11);

        // Delete one sealed point and the appendable one.
        assert!(col.delete(ids[0]));
        assert!(!col.delete(ids[0])); // already deleted
        assert_eq!(col.len(), 10);
        assert!(col.delete(GlobalId::new(10))); // the appendable insert
        assert_eq!(col.len(), 9);
    }

    #[test]
    fn dimension_mismatch_is_an_error() {
        let col = Collection::create(CollectionConfig::new("c", 4, Metric::Dot));
        assert!(matches!(
            col.insert(&[1.0, 2.0]),
            Err(CoreError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
        assert!(col.search(&[1.0, 2.0, 3.0], 5, None).is_err());
    }
}
