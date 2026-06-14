//! The mutable appendable segment.
//!
//! All hot writes land in one small appendable segment per collection: vectors are
//! pushed into a contiguous [`VectorStorage`] and searched by exact flat scan. It is
//! purely additive — **deletions are collection-level**, not stored here — so the
//! segment is trivially snapshot-able. A background optimizer later *seals* it into
//! an immutable [`SealedSegment`](super::SealedSegment) with a built HNSW.

use std::sync::Arc;

use super::id_map::IdMap;
use super::sealed::SealedSegment;
use super::search::search_local;
use crate::distance::{DistanceKernel, Metric};
use crate::id::SegmentId;
use crate::id::{GlobalId, LocalId};
use crate::index::{HnswConfig, HnswIndex};
use crate::payload::FilterQuery;
use crate::vector::VectorStorage;
use crate::version::DeletionVector;

/// A growable, in-RAM segment that accepts appends and serves exact search.
pub struct AppendableSegment {
    vectors: VectorStorage,
    ids: IdMap,
    kernel: DistanceKernel,
}

impl AppendableSegment {
    /// Creates an empty appendable segment for `dim`-dimensional vectors.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            vectors: VectorStorage::new(dim, metric),
            ids: IdMap::new(),
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

    /// The number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the segment has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Appends `vector` under collection-global id `global`, returning its local id.
    pub fn append(&mut self, global: GlobalId, vector: &[f32]) -> LocalId {
        let pid = self.vectors.push(vector);
        let lid = self.ids.push(global);
        debug_assert_eq!(pid.to_local(), lid, "vector/id append out of lockstep");
        lid
    }

    /// Whether `global` is in this segment.
    #[inline]
    pub fn contains(&self, global: GlobalId) -> bool {
        self.ids.to_local(global).is_some()
    }

    /// The f32 vector for `global`, if present (used by recommend-by-example).
    pub(crate) fn vector_of(&self, global: GlobalId) -> Option<&[f32]> {
        self.ids
            .to_local(global)
            .map(|local| self.vectors.get(local.to_point()))
    }

    /// Iterates `(global_id, vector)` for every row (used by the explorer scroll).
    pub(crate) fn iter_points(&self) -> impl Iterator<Item = (GlobalId, &[f32])> {
        (0..self.vectors.len() as u32).map(move |local| {
            let lid = LocalId::new(local);
            (self.ids.global_at(lid), self.vectors.get(lid.to_point()))
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

    /// Exact top-k search excluding `deletions` and applying the optional `filter`,
    /// returning `(global_id, score)`.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        deletions: &DeletionVector,
        filter: Option<&FilterQuery>,
    ) -> Vec<(GlobalId, f32)> {
        search_local(
            &self.vectors,
            &self.kernel,
            &self.ids,
            deletions,
            query,
            k,
            filter,
        )
    }

    /// Seals this segment into an immutable, HNSW-backed [`SealedSegment`].
    pub fn seal(self, id: SegmentId, config: HnswConfig) -> SealedSegment {
        let vectors = Arc::new(self.vectors);
        let index = HnswIndex::build(vectors, config);
        SealedSegment::from_index(id, Arc::new(index), self.ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_search() {
        let mut seg = AppendableSegment::new(3, Metric::Dot);
        let g0 = GlobalId::new(10);
        let g1 = GlobalId::new(11);
        seg.append(g0, &[1.0, 0.0, 0.0]);
        seg.append(g1, &[0.0, 1.0, 0.0]);
        assert_eq!(seg.len(), 2);
        assert_eq!(seg.global_id_range(), Some((10, 11)));

        let empty = DeletionVector::new();
        let res = seg.search(&[1.0, 0.0, 0.0], 2, &empty, None);
        assert_eq!(res[0].0, g0);

        // A deletion vector excludes the point from results.
        let mut dv = DeletionVector::new();
        dv.insert(g0);
        let res = seg.search(&[1.0, 0.0, 0.0], 2, &dv, None);
        assert!(res.iter().all(|(g, _)| *g != g0));
    }
}
