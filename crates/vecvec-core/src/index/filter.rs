//! The filtered-search seam.
//!
//! Metadata filtering is expressed to the index layer through one narrow trait,
//! [`FilterContext`]: a per-point predicate plus a cardinality estimate. Keeping
//! the index ignorant of *how* a filter is evaluated lets the payload/query planner
//! (M9) decide between an exact pre-filter scan and a filter-aware HNSW traversal,
//! and lets it compile a complex boolean filter down to a single primary
//! [`RoaringBitmap`] that this layer just probes.
//!
//! M2 ships two concrete contexts used by tests and later by the planner:
//! [`AllowAll`] and [`BitmapFilter`].

use std::sync::Arc;

use roaring::RoaringBitmap;

use crate::id::PointId;

/// An estimate of how many points a filter admits within a segment. Bounds must be
/// conservative: `min <= expected <= max`, and `max` must never *under*-count (the
/// planner relies on this to stay correct).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CardinalityEstimate {
    /// A lower bound on the number of admitted points.
    pub min: usize,
    /// The expected (best-guess) number of admitted points.
    pub expected: usize,
    /// An upper bound on the number of admitted points.
    pub max: usize,
}

impl CardinalityEstimate {
    /// An estimate that is known exactly.
    pub const fn exact(n: usize) -> Self {
        Self {
            min: n,
            expected: n,
            max: n,
        }
    }
}

/// A predicate over segment-local point ids, with a cardinality estimate.
///
/// Implementations must be cheap to probe (`matches` is called per candidate during
/// traversal) and thread-safe (search runs on a rayon pool).
pub trait FilterContext: Send + Sync {
    /// Whether `id` passes the filter.
    fn matches(&self, id: PointId) -> bool;

    /// An estimate of how many of the segment's points pass the filter.
    fn estimate_cardinality(&self) -> CardinalityEstimate;
}

/// A filter that admits every point. Equivalent to passing no filter; handy as a
/// neutral element and in tests.
#[derive(Debug, Clone, Copy)]
pub struct AllowAll {
    /// The number of points in the segment (so the estimate is exact).
    pub total: usize,
}

impl FilterContext for AllowAll {
    #[inline]
    fn matches(&self, _id: PointId) -> bool {
        true
    }
    #[inline]
    fn estimate_cardinality(&self) -> CardinalityEstimate {
        CardinalityEstimate::exact(self.total)
    }
}

/// A filter backed by a precomputed bitmap of admitted ids — the shape the M9
/// planner compiles a concrete filter into (the "primary" clause bitmap).
#[derive(Debug, Clone)]
pub struct BitmapFilter {
    allowed: Arc<RoaringBitmap>,
}

impl BitmapFilter {
    /// Builds a filter admitting exactly the ids in `allowed`.
    pub fn new(allowed: Arc<RoaringBitmap>) -> Self {
        Self { allowed }
    }

    /// Convenience constructor from an iterator of ids.
    pub fn from_ids(ids: impl IntoIterator<Item = PointId>) -> Self {
        let bm: RoaringBitmap = ids.into_iter().map(PointId::get).collect();
        Self::new(Arc::new(bm))
    }
}

impl FilterContext for BitmapFilter {
    #[inline]
    fn matches(&self, id: PointId) -> bool {
        self.allowed.contains(id.get())
    }
    #[inline]
    fn estimate_cardinality(&self) -> CardinalityEstimate {
        CardinalityEstimate::exact(self.allowed.len() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_all_matches_everything() {
        let f = AllowAll { total: 10 };
        assert!(f.matches(PointId::new(0)));
        assert!(f.matches(PointId::new(999)));
        assert_eq!(f.estimate_cardinality(), CardinalityEstimate::exact(10));
    }

    #[test]
    fn bitmap_filter_admits_exact_subset() {
        let f = BitmapFilter::from_ids([1, 4, 9].into_iter().map(PointId::new));
        assert!(f.matches(PointId::new(4)));
        assert!(!f.matches(PointId::new(5)));
        assert_eq!(f.estimate_cardinality(), CardinalityEstimate::exact(3));
    }
}
