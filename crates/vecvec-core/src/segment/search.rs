//! Shared per-segment search: an exact local scan mapped to global ids.

use super::id_map::IdMap;
use crate::distance::DistanceKernel;
use crate::id::GlobalId;
use crate::index::{FilterContext, SoftDeleteSet, scan_topk};
use crate::vector::VectorStorage;

/// Runs the benched [`scan_topk`] over a segment's vectors and translates the
/// segment-local results into `(global_id, score)` pairs.
pub(crate) fn search_local(
    vectors: &VectorStorage,
    kernel: &DistanceKernel,
    ids: &IdMap,
    deleted: &SoftDeleteSet,
    query: &[f32],
    k: usize,
    filter: Option<&dyn FilterContext>,
) -> Vec<(GlobalId, f32)> {
    let tombstones = deleted.snapshot();
    scan_topk(vectors, kernel, query, k, Some(&tombstones), filter)
        .into_iter()
        .map(|sp| (ids.global_at(sp.id.to_local()), sp.score))
        .collect()
}
