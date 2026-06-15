//! Brute-force (flat) index.
//!
//! [`FlatIndex`] scores the query against every live, filter-admitted vector and
//! keeps the top-k via a bounded heap ([`BoundedTopK`]). It is exact, so it serves
//! three roles: the appendable segment's search engine before an HNSW is built, the
//! engine for exact pre-filter scans the planner chooses for highly selective
//! filters (M9), and the ground truth the approximate indexes are tested against.
//!
//! It shares one [`SoftDeleteSet`] for tombstones and an [`Arc<VectorStorage>`] with
//! its owning segment, so deletes and vectors are not duplicated.

use std::sync::Arc;

use super::{Index, ScoredPoint, SearchParams, SoftDeleteSet, scan_topk};
use crate::distance::DistanceKernel;
use crate::id::PointId;
use crate::index::filter::FilterContext;
use crate::vector::VectorStorage;

/// An exact, full-scan index over a shared [`VectorStorage`].
pub struct FlatIndex {
    storage: Arc<VectorStorage>,
    kernel: DistanceKernel,
    deleted: SoftDeleteSet,
}

impl FlatIndex {
    /// Builds a flat index over `storage`, deriving the distance kernel from the
    /// storage's metric and dimensionality.
    pub fn new(storage: Arc<VectorStorage>) -> Self {
        let kernel = DistanceKernel::new(storage.metric(), storage.dim());
        Self {
            storage,
            kernel,
            deleted: SoftDeleteSet::new(),
        }
    }

    /// The shared vector storage.
    #[inline]
    pub fn storage(&self) -> &Arc<VectorStorage> {
        &self.storage
    }

    /// The distance kernel.
    #[inline]
    pub fn kernel(&self) -> &DistanceKernel {
        &self.kernel
    }

    /// The tombstone set (shared semantics with search).
    #[inline]
    pub fn deleted(&self) -> &SoftDeleteSet {
        &self.deleted
    }
}

impl Index for FlatIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        _params: SearchParams,
        filter: Option<&dyn FilterContext>,
    ) -> Vec<ScoredPoint> {
        let tombstones = self.deleted.snapshot();
        scan_topk(
            &self.storage,
            &self.kernel,
            query,
            k,
            Some(&tombstones),
            filter,
        )
    }

    fn delete(&self, id: PointId) -> bool {
        self.deleted.delete(id)
    }

    fn is_deleted(&self, id: PointId) -> bool {
        self.deleted.is_deleted(id)
    }

    fn live_len(&self) -> usize {
        self.storage
            .len()
            .saturating_sub(self.deleted.deleted_count())
    }

    fn capacity(&self) -> usize {
        self.storage.len()
    }

    fn iter_live(&self) -> Box<dyn Iterator<Item = PointId> + '_> {
        let tombstones = self.deleted.snapshot();
        let n = self.storage.len() as u32;
        Box::new(
            (0..n)
                .map(PointId::new)
                .filter(move |id| !tombstones.contains(id.get())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::index::brute_force_topk;
    use crate::index::filter::BitmapFilter;

    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((x % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    fn storage_of(dim: usize, metric: Metric, n: usize) -> Arc<VectorStorage> {
        let mut s = VectorStorage::with_capacity(dim, metric, n);
        for i in 0..n {
            s.push(&vec_of(dim, i as u32 + 1));
        }
        Arc::new(s)
    }

    /// The headline M2 exit test: across metrics, dims, and k — with and without a
    /// filter, and with deletions — FlatIndex returns exactly the oracle's top-k.
    #[test]
    fn matches_brute_force_oracle() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            for &dim in &[3usize, 8, 128] {
                let storage = storage_of(dim, metric, 200);
                let flat = FlatIndex::new(storage.clone());

                // Tombstone a handful of ids (including some likely-near ones).
                for id in [3u32, 7, 50, 120, 199] {
                    assert!(flat.delete(PointId::new(id)));
                }

                let query = vec_of(dim, 9999);
                for &k in &[1usize, 5, 10, 250] {
                    let got = flat.search(&query, k, SearchParams::default(), None);
                    let want = brute_force_topk(
                        &storage,
                        flat.kernel(),
                        &query,
                        k,
                        Some(flat.deleted()),
                        None,
                    );
                    assert_eq!(got, want, "metric={metric} dim={dim} k={k}");
                    // No tombstoned id ever appears.
                    assert!(got.iter().all(|sp| !flat.is_deleted(sp.id)));
                }
            }
        }
    }

    #[test]
    fn filtered_search_matches_oracle() {
        let dim = 16;
        let storage = storage_of(dim, Metric::Dot, 100);
        let flat = FlatIndex::new(storage.clone());
        flat.delete(PointId::new(2));

        // Admit only even ids.
        let filter = BitmapFilter::from_ids((0..100u32).filter(|i| i % 2 == 0).map(PointId::new));
        let query = vec_of(dim, 42);

        let got = flat.search(&query, 10, SearchParams::default(), Some(&filter));
        let want = brute_force_topk(
            &storage,
            flat.kernel(),
            &query,
            10,
            Some(flat.deleted()),
            Some(&filter),
        );
        assert_eq!(got, want);
        // Every result is even and live.
        assert!(
            got.iter()
                .all(|sp| sp.id.get() % 2 == 0 && sp.id != PointId::new(2))
        );
    }

    #[test]
    fn allow_all_filter_equals_no_filter() {
        let dim = 8;
        let storage = storage_of(dim, Metric::Euclidean, 50);
        let flat = FlatIndex::new(storage.clone());
        let query = vec_of(dim, 5);
        let none = flat.search(&query, 7, SearchParams::default(), None);
        let all = flat.search(
            &query,
            7,
            SearchParams::default(),
            Some(&crate::index::filter::AllowAll { total: 50 }),
        );
        assert_eq!(none, all);
    }

    #[test]
    fn counts_and_ratio_track_deletes() {
        let storage = storage_of(4, Metric::Dot, 10);
        let flat = FlatIndex::new(storage);
        assert_eq!(flat.capacity(), 10);
        assert_eq!(flat.live_len(), 10);
        for id in 0..3u32 {
            flat.delete(PointId::new(id));
        }
        assert_eq!(flat.live_len(), 7);
        assert!((flat.deleted_ratio() - 0.3).abs() < 1e-6);
        let live: Vec<_> = flat.iter_live().collect();
        assert_eq!(live.len(), 7);
        assert!(!live.contains(&PointId::new(0)));
    }

    #[test]
    fn k_zero_returns_empty() {
        let storage = storage_of(4, Metric::Dot, 10);
        let flat = FlatIndex::new(storage);
        assert!(
            flat.search(&vec_of(4, 1), 0, SearchParams::default(), None)
                .is_empty()
        );
    }

    #[test]
    fn ties_broken_by_ascending_id() {
        // Several identical vectors => identical scores. The `(badness, id)` rank key
        // must break ties by ascending id, deterministically and identically in both
        // BoundedTopK (flat search) and the brute-force oracle — the property that
        // lets the oracle validate approximate indexes.
        let dim = 8;
        let metric = Metric::Cosine; // normalized => the tie group scores exactly 1.0
        let mut s = VectorStorage::with_capacity(dim, metric, 20);
        let tied = vec_of(dim, 1);
        for _ in 0..10 {
            s.push(&tied); // ids 0..10 share one vector (a 10-way score tie)
        }
        for i in 10..20 {
            s.push(&vec_of(dim, 1000 + i)); // distinct, lower-scoring
        }
        let storage = Arc::new(s);
        let flat = FlatIndex::new(storage.clone());

        // Query == the tied vector: the 10 identical points all score highest (cosine
        // 1.0) and tie. With k < the tie-group size, the smallest ids must win, in
        // ascending order.
        let got = flat.search(&tied, 4, SearchParams::default(), None);
        let ids: Vec<u32> = got.iter().map(|sp| sp.id.get()).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2, 3],
            "ties must resolve to the smallest ids in order"
        );

        // And it matches the oracle exactly (ids and scores).
        let want = brute_force_topk(&storage, flat.kernel(), &tied, 4, None, None);
        assert_eq!(got, want);
    }
}
