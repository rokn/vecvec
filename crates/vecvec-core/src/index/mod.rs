//! The pluggable vector-index seam.
//!
//! Every index type (the [`FlatIndex`] brute-force scanner here, the HNSW graph in
//! M4, future IVF/DiskANN) implements the object-safe [`Index`] trait, so a segment
//! can hold an `Arc<dyn Index>` and the rest of the engine never depends on which
//! index it is. Ids at this layer are **segment-local** [`PointId`]s; the
//! local→global mapping lives one level up in the segment.
//!
//! [`brute_force_topk`] is the exact reference implementation of top-k search. It
//! is reused as the test oracle for every approximate index and as the actual
//! engine for exact pre-filter scans chosen by the planner (M9).

pub mod deleted;
pub mod filter;
pub mod flat;

use std::collections::BinaryHeap;

pub use deleted::{REBUILD_DELETED_RATIO, SoftDeleteSet};
pub use filter::{AllowAll, BitmapFilter, CardinalityEstimate, FilterContext};
pub use flat::FlatIndex;

use crate::distance::DistanceKernel;
use crate::id::PointId;
use crate::ordered::OrderedF32;
use crate::vector::VectorStorage;

/// A scored search result: a segment-local id and its raw metric score (polarity
/// per the index's metric — see [`crate::distance::Metric::higher_is_better`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredPoint {
    /// The segment-local point id.
    pub id: PointId,
    /// The raw metric score.
    pub score: f32,
}

/// Per-query search tuning shared across index types.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchParams {
    /// The HNSW search beam width. `0` (the default) means "let the index choose"
    /// (and is ignored by exact indexes like [`FlatIndex`]). Always effectively at
    /// least `k`.
    pub ef: usize,
    /// Force an exact scan even if an approximate path exists.
    pub exact: bool,
}

/// A pluggable vector index over a single segment's vectors.
///
/// Object-safe by construction: building/inserting are concrete-type concerns (a
/// sealed segment's index is immutable once built), while the shared read +
/// soft-delete surface lives here so segments can treat any index uniformly.
pub trait Index: Send + Sync {
    /// Returns the best `k` live, filter-admitted points for `query`, ordered
    /// best-first. `filter == None` admits all live points.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: SearchParams,
        filter: Option<&dyn FilterContext>,
    ) -> Vec<ScoredPoint>;

    /// Tombstones `id`. Returns `true` if newly deleted.
    fn delete(&self, id: PointId) -> bool;

    /// Whether `id` is tombstoned.
    fn is_deleted(&self, id: PointId) -> bool;

    /// The number of live (non-tombstoned) points.
    fn live_len(&self) -> usize;

    /// The total number of points, including tombstones.
    fn capacity(&self) -> usize;

    /// Whether there are no live points.
    fn is_empty(&self) -> bool {
        self.live_len() == 0
    }

    /// The fraction of points that are tombstoned.
    fn deleted_ratio(&self) -> f32 {
        deleted::deleted_ratio(self.capacity() - self.live_len(), self.capacity())
    }

    /// Iterates the live (non-tombstoned) ids.
    fn iter_live(&self) -> Box<dyn Iterator<Item = PointId> + '_>;
}

/// Ranking key: a `(badness, id)` tuple where a *smaller* tuple is a *better*
/// result. "Badness" negates the score for higher-is-better metrics so a single
/// ascending sort works for all metrics, and the `id` tie-break makes top-k
/// deterministic (so approximate indexes can be compared against the oracle).
#[inline]
fn rank_key<I>(score: f32, id: I, higher_is_better: bool) -> (OrderedF32, I) {
    let badness = if higher_is_better { -score } else { score };
    (OrderedF32::new(badness), id)
}

#[inline]
fn score_from_badness(badness: f32, higher_is_better: bool) -> f32 {
    if higher_is_better { -badness } else { badness }
}

/// A bounded top-k collector over ids of type `I`: keeps the `k` most-preferred
/// `(score, id)` pairs seen so far in `O(log k)` per offer, with the same `(badness,
/// id)` ordering as [`brute_force_topk`]. Generic over the id type so it serves both
/// segment-local ([`PointId`]) results and the collection-global merge ([`GlobalId`]).
///
/// Internally a max-heap whose root is the *worst* kept result, so a better incoming
/// result evicts it.
///
/// [`GlobalId`]: crate::id::GlobalId
pub(crate) struct BoundedTopK<I: Copy + Ord> {
    k: usize,
    higher_is_better: bool,
    heap: BinaryHeap<(OrderedF32, I)>,
}

impl<I: Copy + Ord> BoundedTopK<I> {
    pub(crate) fn new(k: usize, higher_is_better: bool) -> Self {
        Self {
            k,
            higher_is_better,
            heap: BinaryHeap::with_capacity(k.min(1024)),
        }
    }

    #[inline]
    pub(crate) fn offer(&mut self, id: I, score: f32) {
        if self.k == 0 {
            return;
        }
        let key = rank_key(score, id, self.higher_is_better);
        if self.heap.len() < self.k {
            self.heap.push(key);
        } else if let Some(worst) = self.heap.peek()
            && key < *worst
        {
            // The heap root is the worst kept result; replaced when `key` is better.
            self.heap.pop();
            self.heap.push(key);
        }
    }

    /// Drains into a best-first `(id, score)` vector.
    pub(crate) fn into_sorted(self) -> Vec<(I, f32)> {
        let higher = self.higher_is_better;
        let mut keys: Vec<(OrderedF32, I)> = self.heap.into_vec();
        keys.sort_unstable(); // ascending (badness, id) == best-first
        keys.into_iter()
            .map(|(b, id)| (id, score_from_badness(b.into_inner(), higher)))
            .collect()
    }
}

/// Heap-based exact top-k scan over a [`VectorStorage`] — the shared, efficient
/// (`O(n log k)`) search path used by [`FlatIndex`] and the appendable segment.
/// `tombstones`/`filter` of `None` admit everything.
pub(crate) fn scan_topk(
    storage: &VectorStorage,
    kernel: &DistanceKernel,
    query: &[f32],
    k: usize,
    tombstones: Option<&roaring::RoaringBitmap>,
    filter: Option<&dyn FilterContext>,
) -> Vec<ScoredPoint> {
    let higher = kernel.metric().higher_is_better();
    let mut topk = BoundedTopK::<PointId>::new(k, higher);
    for (id, v) in storage.iter() {
        if let Some(t) = tombstones
            && t.contains(id.get())
        {
            continue;
        }
        if let Some(f) = filter
            && !f.matches(id)
        {
            continue;
        }
        topk.offer(id, kernel.score_f32(query, v));
    }
    topk.into_sorted()
        .into_iter()
        .map(|(id, score)| ScoredPoint { id, score })
        .collect()
}

/// Exact top-k search by full scan — the reference implementation and test oracle.
///
/// Scores every live, filter-admitted point and returns the best `k`, ordered
/// best-first with a deterministic `id` tie-break. `deleted`/`filter` of `None`
/// admit everything.
pub fn brute_force_topk(
    storage: &VectorStorage,
    kernel: &DistanceKernel,
    query: &[f32],
    k: usize,
    deleted: Option<&SoftDeleteSet>,
    filter: Option<&dyn FilterContext>,
) -> Vec<ScoredPoint> {
    let higher = kernel.metric().higher_is_better();
    let tombstones = deleted.map(SoftDeleteSet::snapshot);

    let mut scored: Vec<(OrderedF32, PointId)> = Vec::new();
    for (id, v) in storage.iter() {
        if let Some(t) = &tombstones
            && t.contains(id.get())
        {
            continue;
        }
        if let Some(f) = filter
            && !f.matches(id)
        {
            continue;
        }
        let score = kernel.score_f32(query, v);
        scored.push(rank_key(score, id, higher));
    }
    scored.sort_unstable();
    scored.truncate(k);
    scored
        .into_iter()
        .map(|(b, id)| ScoredPoint {
            id,
            score: score_from_badness(b.into_inner(), higher),
        })
        .collect()
}
