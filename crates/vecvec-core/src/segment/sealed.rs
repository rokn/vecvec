//! Immutable sealed segments.
//!
//! A sealed segment is the immutable, `Arc`-shared building block that versions
//! reference by id. It is now **fully immutable**: deletions are recorded at the
//! collection level (a [`DeletionVector`]) and passed into search, never mutated
//! here. That is exactly what makes structural-sharing snapshots and time-travel
//! correct — an old version searches the same physical segment with its own frozen
//! deletion vector.

use std::sync::Arc;

use super::id_map::IdMap;
use super::search::SegmentLiveFilter;
use crate::distance::{DistanceKernel, Metric};
use crate::id::{GlobalId, LocalId, PointId, SegmentId};
use crate::index::{HnswIndex, Index, SearchParams, scan_topk};
use crate::payload::FilterQuery;
use crate::version::DeletionVector;

/// An immutable, HNSW-backed segment.
pub struct SealedSegment {
    id: SegmentId,
    index: Arc<HnswIndex>,
    ids: IdMap,
}

impl SealedSegment {
    /// Assembles a sealed segment from a built index and its id map.
    pub(crate) fn from_index(id: SegmentId, index: Arc<HnswIndex>, ids: IdMap) -> Self {
        Self { id, index, ids }
    }

    /// The segment's stable id.
    #[inline]
    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// The metric.
    #[inline]
    pub fn metric(&self) -> Metric {
        self.index.vectors().metric()
    }

    /// The number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.index.capacity()
    }

    /// Whether the segment has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.capacity() == 0
    }

    /// The underlying index (for serialization).
    pub(crate) fn index(&self) -> &Arc<HnswIndex> {
        &self.index
    }

    /// The id map (for serialization).
    pub(crate) fn id_map(&self) -> &IdMap {
        &self.ids
    }

    /// Whether `global` is mapped by this segment.
    #[inline]
    pub fn contains(&self, global: GlobalId) -> bool {
        self.ids.to_local(global).is_some()
    }

    /// The f32 vector for `global`, if present (used by recommend-by-example).
    pub(crate) fn vector_of(&self, global: GlobalId) -> Option<&[f32]> {
        self.ids
            .to_local(global)
            .map(|local| self.index.vectors().get(local.to_point()))
    }

    /// Iterates `(global_id, vector)` for every row (used by compaction).
    pub(crate) fn iter_points(&self) -> impl Iterator<Item = (GlobalId, &[f32])> {
        let vectors = self.index.vectors();
        (0..self.len() as u32).map(move |local| {
            (
                self.ids.global_at(LocalId::new(local)),
                vectors.get(PointId::new(local)),
            )
        })
    }

    /// The inclusive global-id range this segment spans, or `None` if empty.
    pub fn global_id_range(&self) -> Option<(u64, u64)> {
        let ids = self.ids.global_ids();
        match (ids.first(), ids.last()) {
            (Some(lo), Some(hi)) => Some((lo.get().min(hi.get()), lo.get().max(hi.get()))),
            _ => None,
        }
    }

    /// HNSW top-k search excluding `deletions` and applying the (optional) payload
    /// `filter`, returning `(global_id, score)`.
    ///
    /// A selective filter can starve the HNSW beam (too few matches reachable), so
    /// if the filtered graph search under-fills we fall back to an exact filtered
    /// scan over the segment's f32 vectors — correct at the cost of an O(n) pass,
    /// taken only when the filter is highly selective.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        deletions: &DeletionVector,
        filter: Option<&FilterQuery>,
    ) -> Vec<(GlobalId, f32)> {
        let live = SegmentLiveFilter::new(&self.ids, deletions, filter);
        let graph_results = self
            .index
            .search(query, k, SearchParams::default(), Some(&live));
        let results = if filter.is_some() && graph_results.len() < k {
            let kernel = DistanceKernel::new(self.metric(), self.index.vectors().dim());
            scan_topk(self.index.vectors(), &kernel, query, k, None, Some(&live))
        } else {
            graph_results
        };
        results
            .into_iter()
            .map(|sp| (self.ids.global_at(sp.id.to_local()), sp.score))
            .collect()
    }
}
