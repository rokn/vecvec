//! Immutable sealed segments.
//!
//! A sealed segment is the immutable, `Arc`-shared building block that versioning
//! references by id — once sealed it is never mutated except for soft-delete
//! tombstones layered on top, which is what makes cheap structural-sharing
//! snapshots possible (M7).
//!
//! As of M5 a sealed segment is backed by a built **HNSW** index (so sealed search
//! is approximate-but-fast and matches the flat oracle within recall tolerance) and
//! can be persisted to / loaded from disk (see [`codec`](super::codec) and
//! [`SegmentStore`](super::SegmentStore)). True zero-copy mmap residency of the
//! vector block is a later hardening step; today a loaded segment owns its data.

use std::sync::Arc;

use super::id_map::IdMap;
use crate::distance::Metric;
use crate::id::{GlobalId, SegmentId};
use crate::index::{FilterContext, HnswIndex, Index, SearchParams};

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

    /// The total number of rows (including tombstones).
    #[inline]
    pub fn len(&self) -> usize {
        self.index.capacity()
    }

    /// Whether the segment has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.capacity() == 0
    }

    /// The number of live (non-tombstoned) rows.
    #[inline]
    pub fn live_len(&self) -> usize {
        self.index.live_len()
    }

    /// The underlying index (for serialization).
    pub(crate) fn index(&self) -> &Arc<HnswIndex> {
        &self.index
    }

    /// The id map (for serialization).
    pub(crate) fn id_map(&self) -> &IdMap {
        &self.ids
    }

    /// Tombstones the row for `global`, if present. Returns whether it was newly
    /// deleted. (Interior-mutable: tombstones are the one thing that can change.)
    pub fn delete_global(&self, global: GlobalId) -> bool {
        match self.ids.to_local(global) {
            Some(local) => self.index.delete(local.to_point()),
            None => false,
        }
    }

    /// Whether `global` is mapped by this segment (whether or not it's tombstoned).
    #[inline]
    pub fn contains(&self, global: GlobalId) -> bool {
        self.ids.to_local(global).is_some()
    }

    /// Whether `global` is present and live in this segment.
    pub fn contains_live(&self, global: GlobalId) -> bool {
        match self.ids.to_local(global) {
            Some(local) => !self.index.is_deleted(local.to_point()),
            None => false,
        }
    }

    /// Exact-ish (HNSW) top-k search, returning `(global_id, score)` best-first.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&dyn FilterContext>,
    ) -> Vec<(GlobalId, f32)> {
        self.index
            .search(query, k, SearchParams::default(), filter)
            .into_iter()
            .map(|sp| (self.ids.global_at(sp.id.to_local()), sp.score))
            .collect()
    }
}
