//! Immutable sealed segments.
//!
//! A sealed segment is the immutable, `Arc`-shared building block that versioning
//! references by id — once sealed it is never mutated (except for soft-delete
//! tombstones layered on top), which is what makes cheap structural-sharing
//! snapshots possible (M7).
//!
//! At M3 a sealed segment still serves search by exact flat scan over its
//! `Arc<VectorStorage>`. M5 replaces that with a built HNSW sub-index + `u8`
//! quantization and an mmap-backed on-disk form, behind this same interface.

use std::sync::Arc;

use super::id_map::IdMap;
use super::search::search_local;
use crate::distance::{DistanceKernel, Metric};
use crate::id::{GlobalId, SegmentId};
use crate::index::{FilterContext, SoftDeleteSet};
use crate::vector::VectorStorage;

/// An immutable segment: vectors + id map are fixed; only tombstones can change.
pub struct SealedSegment {
    id: SegmentId,
    vectors: Arc<VectorStorage>,
    ids: IdMap,
    deleted: SoftDeleteSet,
    kernel: DistanceKernel,
}

impl SealedSegment {
    /// Assembles a sealed segment from its parts (used by
    /// [`AppendableSegment::seal`](super::AppendableSegment::seal)).
    pub(crate) fn from_parts(
        id: SegmentId,
        vectors: VectorStorage,
        ids: IdMap,
        deleted: SoftDeleteSet,
        kernel: DistanceKernel,
    ) -> Self {
        Self {
            id,
            vectors: Arc::new(vectors),
            ids,
            deleted,
            kernel,
        }
    }

    /// The segment's stable id.
    #[inline]
    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// The metric.
    #[inline]
    pub fn metric(&self) -> Metric {
        self.vectors.metric()
    }

    /// The total number of rows (including tombstones).
    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the segment has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The number of live (non-tombstoned) rows.
    #[inline]
    pub fn live_len(&self) -> usize {
        self.vectors
            .len()
            .saturating_sub(self.deleted.deleted_count())
    }

    /// Tombstones the row for `global`, if present. Returns whether it was newly
    /// deleted. (Interior-mutable: a sealed segment's tombstones are the one thing
    /// that can change.)
    pub fn delete_global(&self, global: GlobalId) -> bool {
        match self.ids.to_local(global) {
            Some(local) => self.deleted.delete(local.to_point()),
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
            Some(local) => !self.deleted.is_deleted(local.to_point()),
            None => false,
        }
    }

    /// Exact top-k search, returning `(global_id, score)` best-first.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&dyn FilterContext>,
    ) -> Vec<(GlobalId, f32)> {
        search_local(
            &self.vectors,
            &self.kernel,
            &self.ids,
            &self.deleted,
            query,
            k,
            filter,
        )
    }
}
