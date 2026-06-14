//! Shared per-segment search helpers and the version-aware live filter.
//!
//! Deletions (a [`DeletionVector`]) and payload filters (a [`FilterQuery`]) both
//! live at the collection level and are passed into segment search, which adapts
//! them to the index's [`FilterContext`] via [`SegmentLiveFilter`] by mapping each
//! segment-local candidate to its global id. A time-travel query passes a *frozen*
//! deletion vector, which is what gives snapshot isolation.

use super::id_map::IdMap;
use crate::distance::DistanceKernel;
use crate::id::GlobalId;
use crate::index::{CardinalityEstimate, FilterContext, scan_topk};
use crate::payload::FilterQuery;
use crate::vector::VectorStorage;
use crate::version::DeletionVector;

/// A [`FilterContext`] that admits a segment-local point iff it isn't tombstoned and
/// passes the (optional) payload filter.
pub(crate) struct SegmentLiveFilter<'a> {
    ids: &'a IdMap,
    deletions: &'a DeletionVector,
    filter: Option<&'a FilterQuery<'a>>,
}

impl<'a> SegmentLiveFilter<'a> {
    pub(crate) fn new(
        ids: &'a IdMap,
        deletions: &'a DeletionVector,
        filter: Option<&'a FilterQuery<'a>>,
    ) -> Self {
        Self {
            ids,
            deletions,
            filter,
        }
    }
}

impl FilterContext for SegmentLiveFilter<'_> {
    fn matches(&self, local: crate::id::PointId) -> bool {
        let global = self.ids.global_at(local.to_local());
        if self.deletions.contains(global) {
            return false;
        }
        self.filter.is_none_or(|f| f.matches(global.get()))
    }

    fn estimate_cardinality(&self) -> CardinalityEstimate {
        // Conservative: assume nothing is filtered (upper bound never under-counts).
        CardinalityEstimate {
            min: 0,
            expected: self.ids.len(),
            max: self.ids.len(),
        }
    }
}

/// Flat (exact) per-segment search mapped to global ids, applying the live filter.
pub(crate) fn search_local(
    vectors: &VectorStorage,
    kernel: &DistanceKernel,
    ids: &IdMap,
    deletions: &DeletionVector,
    query: &[f32],
    k: usize,
    filter: Option<&FilterQuery>,
) -> Vec<(GlobalId, f32)> {
    let live = SegmentLiveFilter::new(ids, deletions, filter);
    scan_topk(vectors, kernel, query, k, None, Some(&live))
        .into_iter()
        .map(|sp| (ids.global_at(sp.id.to_local()), sp.score))
        .collect()
}
