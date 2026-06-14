//! HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor index.
//!
//! A multi-layer proximity graph: upper layers are sparse "express lanes" for
//! coarse navigation, layer 0 holds everyone. Search greedily descends the upper
//! layers then runs a width-`ef` beam search on layer 0. Build wires each new point
//! to a diverse neighbor set chosen by the Algorithm-4 heuristic (see [`heuristic`]).
//!
//! This module implements the graph from scratch (the rationale, in `BuildPlan.md`:
//! no Rust crate jointly gives us snapshot control, soft-delete, and filtered
//! search). Construction is sequential-deterministic ([`builder`]); the sealed graph
//! ([`graph::GraphLayers`]) is immutable and searched lock-free.

mod builder;
mod graph;
mod heuristic;
mod rng;
mod search;
mod visited;

pub use graph::GraphLayers;

use std::sync::Arc;

use builder::GraphLayersBuilder;
use visited::{VisitedList, VisitedPool};

use super::{Index, ScoredPoint, SearchParams, SoftDeleteSet};
use crate::distance::DistanceKernel;
use crate::id::PointId;
use crate::index::filter::FilterContext;
use crate::vector::VectorStorage;

/// HNSW construction and search parameters. Defaults follow the values validated by
/// the literature and Qdrant (see `BuildPlan.md`).
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Target out-degree on the upper layers.
    pub m: usize,
    /// Maximum out-degree on layer 0 (conventionally `2*m`).
    pub m_max0: usize,
    /// Beam width while building.
    pub ef_construction: usize,
    /// Default beam width while searching (always raised to at least `k`).
    pub ef_search: usize,
    /// Seed for the deterministic level assignment.
    pub seed: u64,
    /// Whether to top up pruned candidates to the degree bound (keepPrunedConnections).
    pub keep_pruned: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 128,
            ef_search: 64,
            seed: 0x5EED_1234_5678_9ABC,
            keep_pruned: true,
        }
    }
}

impl HnswConfig {
    /// The level-distribution constant `mL = 1/ln(M)`.
    #[inline]
    pub(crate) fn ml(&self) -> f64 {
        1.0 / (self.m as f64).ln()
    }
}

/// An HNSW index over a shared [`VectorStorage`].
pub struct HnswIndex {
    vectors: Arc<VectorStorage>,
    kernel: DistanceKernel,
    graph: GraphLayers,
    deleted: SoftDeleteSet,
    config: HnswConfig,
    pool: VisitedPool,
}

impl HnswIndex {
    /// Builds an HNSW index over `vectors` with the given config. Deterministic in
    /// `(vectors, config)`.
    pub fn build(vectors: Arc<VectorStorage>, config: HnswConfig) -> Self {
        let n = vectors.len();
        let kernel = DistanceKernel::new(vectors.metric(), vectors.dim());
        let mut builder = GraphLayersBuilder::new(vectors.clone(), kernel, config);
        let mut visited = VisitedList::new(n.max(1));
        for id in 0..n as u32 {
            builder.insert(id, &mut visited);
        }
        let graph = builder.into_graph_layers();
        Self {
            vectors,
            kernel,
            graph,
            deleted: SoftDeleteSet::new(),
            config,
            pool: VisitedPool::new(n.max(1)),
        }
    }

    /// The sealed graph (exposed for determinism/seal tests).
    #[cfg(test)]
    pub(crate) fn graph(&self) -> &GraphLayers {
        &self.graph
    }
}

impl Index for HnswIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: SearchParams,
        filter: Option<&dyn FilterContext>,
    ) -> Vec<ScoredPoint> {
        if k == 0 || self.graph.len() == 0 {
            return Vec::new();
        }
        // Enforce ef >= k (and honor an explicit override / the config default).
        let ef = params.ef.max(self.config.ef_search).max(k);
        let deleted = self.deleted.snapshot();
        let mut visited = self.pool.get();
        let results = self.graph.search(
            &self.vectors,
            &self.kernel,
            query,
            k,
            ef,
            filter,
            Some(&deleted),
            &mut visited,
        );
        self.pool.put(visited);
        results
    }

    fn delete(&self, id: PointId) -> bool {
        self.deleted.delete(id)
    }

    fn is_deleted(&self, id: PointId) -> bool {
        self.deleted.is_deleted(id)
    }

    fn live_len(&self) -> usize {
        self.vectors
            .len()
            .saturating_sub(self.deleted.deleted_count())
    }

    fn capacity(&self) -> usize {
        self.vectors.len()
    }

    fn iter_live(&self) -> Box<dyn Iterator<Item = PointId> + '_> {
        let tombstones = self.deleted.snapshot();
        let n = self.vectors.len() as u32;
        Box::new(
            (0..n)
                .map(PointId::new)
                .filter(move |id| !tombstones.contains(id.get())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::index::brute_force_topk;

    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        // A few independent hash streams give less structured (more realistic) vectors.
        (0..dim)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503))
                    .wrapping_add(i as u32 * seed);
                ((x % 10_000) as f32 / 5_000.0) - 1.0
            })
            .collect()
    }

    fn storage(dim: usize, n: usize, metric: Metric) -> Arc<VectorStorage> {
        let mut s = VectorStorage::with_capacity(dim, metric, n);
        for i in 0..n {
            s.push(&vec_of(dim, i as u32 + 1));
        }
        Arc::new(s)
    }

    fn recall_at(got: &[ScoredPoint], truth: &[ScoredPoint]) -> f32 {
        let truth_ids: std::collections::HashSet<u32> = truth.iter().map(|s| s.id.get()).collect();
        let hit = got
            .iter()
            .filter(|s| truth_ids.contains(&s.id.get()))
            .count();
        hit as f32 / truth.len() as f32
    }

    #[test]
    fn recall_meets_target() {
        let dim = 24;
        let n = 2_000;
        let queries = 30;
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            let store = storage(dim, n, metric);
            let kernel = DistanceKernel::new(metric, dim);
            let index = HnswIndex::build(store.clone(), HnswConfig::default());

            let mut total = 0.0f32;
            for q in 0..queries {
                let query = vec_of(dim, 100_000 + q);
                let got = index.search(&query, 10, SearchParams::default(), None);
                let truth = brute_force_topk(&store, &kernel, &query, 10, None, None);
                assert_eq!(got.len(), 10);
                total += recall_at(&got, &truth);
            }
            let recall = total / queries as f32;
            assert!(recall >= 0.95, "metric={metric}: recall@10 {recall} < 0.95");
        }
    }

    #[test]
    fn build_is_deterministic() {
        let store = storage(32, 600, Metric::Cosine);
        let a = HnswIndex::build(store.clone(), HnswConfig::default());
        let b = HnswIndex::build(store, HnswConfig::default());
        // Same seed + same vectors => byte-identical sealed graph.
        assert!(a.graph() == b.graph());
    }

    #[test]
    fn ef_is_clamped_to_at_least_k() {
        let store = storage(16, 500, Metric::Dot);
        let index = HnswIndex::build(store, HnswConfig::default());
        // ef explicitly set below k must still return k results.
        let params = SearchParams {
            ef: 1,
            exact: false,
        };
        let got = index.search(&vec_of(16, 7), 50, params, None);
        assert_eq!(got.len(), 50);
    }

    #[test]
    fn deleted_points_are_never_returned() {
        let dim = 16;
        let n = 800;
        let store = storage(dim, n, Metric::Cosine);
        let index = HnswIndex::build(store, HnswConfig::default());
        for id in 0..50u32 {
            index.delete(PointId::new(id));
        }
        let got = index.search(&vec_of(dim, 5), 20, SearchParams::default(), None);
        assert!(got.iter().all(|s| s.id.get() >= 50));
        assert_eq!(index.live_len(), n - 50);
    }

    #[test]
    fn seal_preserves_adjacency() {
        // The sealed graph's neighbors must match what the builder produced.
        let store = storage(16, 300, Metric::Euclidean);
        let kernel = DistanceKernel::new(Metric::Euclidean, 16);
        let cfg = HnswConfig::default();
        let mut builder = GraphLayersBuilder::new(store.clone(), kernel, cfg);
        let mut visited = VisitedList::new(300);
        for id in 0..300u32 {
            builder.insert(id, &mut visited);
        }
        // Snapshot builder layer-0 adjacency before sealing.
        let before: Vec<Vec<u32>> = (0..300)
            .map(|p| {
                use super::search::Graph;
                builder.neighbors(p, 0).to_vec()
            })
            .collect();
        let sealed = builder.into_graph_layers();
        for (p, expected) in before.iter().enumerate() {
            use super::search::Graph;
            assert_eq!(sealed.neighbors(p as u32, 0), expected.as_slice());
        }
    }
}
