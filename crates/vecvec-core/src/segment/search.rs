//! Shared per-segment search helpers and the version-aware live filter.
//!
//! Deletions live at the collection level (a [`DeletionVector`] over global ids),
//! not inside segments, so segment search takes the active deletion vector as an
//! argument and filters by it via [`SegmentLiveFilter`], which maps each candidate's
//! segment-local id to its global id. This is what gives snapshot isolation: a
//! time-travel query passes a *frozen* deletion vector, unaffected by later deletes.

use super::id_map::IdMap;
use crate::distance::DistanceKernel;
use crate::id::GlobalId;
use crate::index::{CardinalityEstimate, FilterContext, scan_topk};
use crate::vector::VectorStorage;
use crate::version::DeletionVector;

/// A [`FilterContext`] that admits a segment-local point iff it isn't tombstoned by
/// the (collection-level) deletion vector and passes an optional payload filter.
pub(crate) struct SegmentLiveFilter<'a> {
    ids: &'a IdMap,
    deletions: &'a DeletionVector,
    payload: Option<&'a dyn FilterContext>,
}

impl<'a> SegmentLiveFilter<'a> {
    pub(crate) fn new(
        ids: &'a IdMap,
        deletions: &'a DeletionVector,
        payload: Option<&'a dyn FilterContext>,
    ) -> Self {
        Self {
            ids,
            deletions,
            payload,
        }
    }
}

impl FilterContext for SegmentLiveFilter<'_> {
    fn matches(&self, local: crate::id::PointId) -> bool {
        let global = self.ids.global_at(local.to_local());
        if self.deletions.contains(global) {
            return false;
        }
        self.payload.is_none_or(|p| p.matches(local))
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
    payload: Option<&dyn FilterContext>,
) -> Vec<(GlobalId, f32)> {
    let filter = SegmentLiveFilter::new(ids, deletions, payload);
    scan_topk(vectors, kernel, query, k, None, Some(&filter))
        .into_iter()
        .map(|sp| (ids.global_at(sp.id.to_local()), sp.score))
        .collect()
}
