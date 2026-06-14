//! The sealed, immutable HNSW graph.
//!
//! [`GraphLayers`] is the read-optimized form produced by sealing a builder. Layer
//! 0 (where almost all edges live and the ef-search spends its time) is stored as a
//! flat CSR arena for cache locality; the sparse upper layers are kept per
//! high-level point. The structure derives `PartialEq` so two builds from the same
//! seed can be asserted identical (the determinism guarantee; M5 swaps this for the
//! rkyv on-disk form).

use roaring::RoaringBitmap;

use crate::distance::DistanceKernel;
use crate::id::PointId;
use crate::index::ScoredPoint;
use crate::index::filter::FilterContext;
use crate::vector::VectorStorage;

use super::search::{Graph, search_layer};
use super::visited::VisitedList;

/// An immutable HNSW graph.
#[derive(Clone, PartialEq)]
pub struct GraphLayers {
    pub(crate) entry: Option<u32>,
    pub(crate) max_level: usize,
    /// Per-point assigned level.
    pub(crate) levels: Vec<u8>,
    /// Layer-0 CSR: neighbors of point `p` are `l0_links[l0_offsets[p]..l0_offsets[p+1]]`.
    pub(crate) l0_offsets: Vec<u32>,
    pub(crate) l0_links: Vec<u32>,
    /// Sparse upper layers: `upper[upper_index[p]][layer-1]` for points with level≥1.
    pub(crate) upper: Vec<Vec<Vec<u32>>>,
    /// Maps a point to its slot in `upper`, or `u32::MAX` if it has no upper layers.
    pub(crate) upper_index: Vec<u32>,
}

impl Graph for GraphLayers {
    #[inline]
    fn neighbors(&self, point: u32, layer: usize) -> &[u32] {
        if layer == 0 {
            let p = point as usize;
            let start = self.l0_offsets[p] as usize;
            let end = self.l0_offsets[p + 1] as usize;
            &self.l0_links[start..end]
        } else {
            let slot = self.upper_index[point as usize];
            if slot == u32::MAX {
                &[]
            } else {
                self.upper[slot as usize]
                    .get(layer - 1)
                    .map_or(&[], |v| v.as_slice())
            }
        }
    }
}

impl GraphLayers {
    /// The number of points in the graph.
    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    /// Searches for the best `k` admitted points for `query` with beam width `ef`.
    /// Points in `deleted` or rejected by `filter` are traversed (for connectivity)
    /// but never returned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search(
        &self,
        vectors: &VectorStorage,
        kernel: &DistanceKernel,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn FilterContext>,
        deleted: Option<&RoaringBitmap>,
        visited: &mut VisitedList,
    ) -> Vec<ScoredPoint> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let higher = kernel.metric().higher_is_better();
        let dist = |id: u32| {
            let s = kernel.score_f32(query, vectors.get(PointId::new(id)));
            if higher { -s } else { s }
        };
        let admit_all = |_: u32| true;

        // Greedy descent through the upper layers (beam width 1).
        let mut ep = vec![entry];
        for layer in (1..=self.max_level).rev() {
            let w = search_layer(self, layer, &ep, 1, &dist, &admit_all, visited);
            if let Some(&(_, best)) = w.first() {
                ep = vec![best];
            }
        }

        // Layer 0: full ef-search, admitting only live, filter-passing points.
        let admit = |id: u32| {
            if let Some(d) = deleted
                && d.contains(id)
            {
                return false;
            }
            if let Some(f) = filter
                && !f.matches(PointId::new(id))
            {
                return false;
            }
            true
        };
        let w = search_layer(self, 0, &ep, ef, &dist, &admit, visited);
        w.into_iter()
            .take(k)
            .map(|(b, id)| ScoredPoint {
                id: PointId::new(id),
                score: if higher { -b } else { b },
            })
            .collect()
    }
}
