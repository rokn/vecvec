//! The mutable appendable segment.
//!
//! All hot writes land in one small appendable segment per collection: vectors are
//! pushed into a contiguous [`VectorStorage`] and searched by exact flat scan (the
//! benched [`scan_topk`](crate::index::scan_topk) path). A background optimizer
//! later *seals* it — [`AppendableSegment::seal`] — into an immutable
//! [`SealedSegment`](super::SealedSegment); from M5 sealing also builds an HNSW and
//! quantizes. Until then a sealed segment simply keeps flat-scanning.

use super::id_map::IdMap;
use super::sealed::SealedSegment;
use super::search::search_local;
use crate::distance::{DistanceKernel, Metric};
use crate::id::{GlobalId, LocalId, SegmentId};
use crate::index::{FilterContext, SoftDeleteSet};
use crate::vector::VectorStorage;

/// A growable, in-RAM segment that accepts appends and serves exact search.
pub struct AppendableSegment {
    vectors: VectorStorage,
    ids: IdMap,
    deleted: SoftDeleteSet,
    kernel: DistanceKernel,
}

impl AppendableSegment {
    /// Creates an empty appendable segment for `dim`-dimensional vectors.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            vectors: VectorStorage::new(dim, metric),
            ids: IdMap::new(),
            deleted: SoftDeleteSet::new(),
            kernel: DistanceKernel::new(metric, dim),
        }
    }

    /// The vector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.vectors.dim()
    }

    /// The metric.
    #[inline]
    pub fn metric(&self) -> Metric {
        self.vectors.metric()
    }

    /// The total number of appended rows (including tombstones).
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

    /// Appends `vector` under collection-global id `global`, returning its local id.
    ///
    /// # Panics
    /// Panics if `vector.len() != self.dim()`.
    pub fn append(&mut self, global: GlobalId, vector: &[f32]) -> LocalId {
        let pid = self.vectors.push(vector);
        let lid = self.ids.push(global);
        debug_assert_eq!(pid.to_local(), lid, "vector/id append out of lockstep");
        lid
    }

    /// Tombstones the row for `global`, if present. Returns whether it was newly
    /// deleted.
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

    /// Seals this segment into an immutable [`SealedSegment`] with id `id`,
    /// consuming it.
    pub fn seal(self, id: SegmentId) -> SealedSegment {
        SealedSegment::from_parts(id, self.vectors, self.ids, self.deleted, self.kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_search_delete() {
        let mut seg = AppendableSegment::new(3, Metric::Dot);
        let g0 = GlobalId::new(10);
        let g1 = GlobalId::new(11);
        seg.append(g0, &[1.0, 0.0, 0.0]);
        seg.append(g1, &[0.0, 1.0, 0.0]);
        assert_eq!(seg.len(), 2);
        assert_eq!(seg.live_len(), 2);

        let res = seg.search(&[1.0, 0.0, 0.0], 2, None);
        assert_eq!(res[0].0, g0); // best match is the aligned vector

        assert!(seg.delete_global(g0));
        assert!(!seg.delete_global(g0));
        assert!(!seg.contains_live(g0));
        assert_eq!(seg.live_len(), 1);
        let res = seg.search(&[1.0, 0.0, 0.0], 2, None);
        assert!(res.iter().all(|(g, _)| *g != g0));
    }
}
