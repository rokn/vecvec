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
mod parallel;
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
use crate::ordered::OrderedF32;
use crate::quantization::QuantizedVectorBlock;
use crate::vector::VectorStorage;

/// HNSW construction and search parameters. Defaults follow the values validated by
/// the literature and Qdrant (see `BuildPlan.md`).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
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
    /// Whether to store an int8-quantized copy and rank by quantized distance with
    /// f32 rescore (~4× less memory; the f32 path is exact rescored).
    #[serde(default = "default_quantization")]
    pub quantization: bool,
}

fn default_quantization() -> bool {
    true
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
            quantization: true,
        }
    }
}

/// Candidate over-fetch factor for quantized search before f32 rescore.
const RESCORE_OVERSAMPLE: usize = 4;

/// Below this point count, [`HnswIndex::build_parallel`] uses the sequential build:
/// rayon's fan-out overhead isn't worth it and small graphs build sub-millisecond.
const PARALLEL_MIN: usize = 4096;

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
    /// Optional int8 copy for low-memory ranking (rescored with f32).
    quantized: Option<QuantizedVectorBlock>,
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
        let quantized = config
            .quantization
            .then(|| QuantizedVectorBlock::build(&vectors));
        Self {
            vectors,
            kernel,
            graph,
            quantized,
            deleted: SoftDeleteSet::new(),
            config,
            pool: VisitedPool::new(n.max(1)),
        }
    }

    /// Builds an HNSW index, parallelizing graph construction across the rayon pool
    /// for large inputs. Unlike [`HnswIndex::build`] the resulting graph is **not**
    /// byte-deterministic — concurrent insertion order varies — so it is validated by
    /// recall and structural invariants, not equality. Inputs below
    /// [`PARALLEL_MIN`] delegate to the deterministic sequential build (pool overhead
    /// isn't worth it, and tiny graphs build in microseconds).
    pub fn build_parallel(vectors: Arc<VectorStorage>, config: HnswConfig) -> Self {
        if vectors.len() < PARALLEL_MIN {
            return Self::build(vectors, config);
        }
        Self::build_concurrent(vectors, config)
    }

    /// Always-concurrent build (no size threshold). Separated out so tests can drive
    /// the parallel path at small `n` to cover its edge cases directly.
    fn build_concurrent(vectors: Arc<VectorStorage>, config: HnswConfig) -> Self {
        let kernel = DistanceKernel::new(vectors.metric(), vectors.dim());
        let n = vectors.len();
        // Build the int8 block first so construction can use it for its (memory-bound)
        // distance comparisons; it's then kept for search-time rescore.
        let quantized = config
            .quantization
            .then(|| QuantizedVectorBlock::build(&vectors));
        let graph =
            parallel::build_concurrent_graph(vectors.clone(), kernel, config, quantized.as_ref());
        Self {
            vectors,
            kernel,
            graph,
            quantized,
            deleted: SoftDeleteSet::new(),
            config,
            pool: VisitedPool::new(n.max(1)),
        }
    }

    /// Reconstructs an index from an already-built graph (e.g. loaded from disk),
    /// without rebuilding. `deleted_locals` re-applies tombstones.
    pub(crate) fn from_parts(
        vectors: Arc<VectorStorage>,
        config: HnswConfig,
        graph: GraphLayers,
        deleted_locals: &[u32],
    ) -> Self {
        let kernel = DistanceKernel::new(vectors.metric(), vectors.dim());
        let n = vectors.len();
        let deleted = SoftDeleteSet::new();
        for &local in deleted_locals {
            deleted.delete(PointId::new(local));
        }
        let quantized = config
            .quantization
            .then(|| QuantizedVectorBlock::build(&vectors));
        Self {
            vectors,
            kernel,
            graph,
            quantized,
            deleted,
            config,
            pool: VisitedPool::new(n.max(1)),
        }
    }

    /// The sealed graph (for serialization and determinism/seal tests).
    pub(crate) fn graph(&self) -> &GraphLayers {
        &self.graph
    }

    /// The shared vector storage.
    pub(crate) fn vectors(&self) -> &Arc<VectorStorage> {
        &self.vectors
    }

    /// The construction config.
    pub(crate) fn config(&self) -> HnswConfig {
        self.config
    }

    /// The tombstone set.
    pub(crate) fn deleted(&self) -> &SoftDeleteSet {
        &self.deleted
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
        let higher = self.kernel.metric().higher_is_better();
        let deleted = self.deleted.snapshot();
        let mut visited = self.pool.get();

        let candidates = if let Some(qb) = &self.quantized {
            // Rank by quantized distance over an over-fetched beam, then rescore.
            let quantized_query = qb.quantizer().encode(query);
            let dist = |id: u32| qb.badness(&quantized_query, id);
            let fetch = (k * RESCORE_OVERSAMPLE).max(k);
            self.graph.search_ids(
                &dist,
                fetch,
                ef.max(fetch),
                filter,
                Some(&deleted),
                &mut visited,
            )
        } else {
            let dist = |id: u32| {
                let s = self
                    .kernel
                    .score_f32(query, self.vectors.get(PointId::new(id)));
                if higher { -s } else { s }
            };
            self.graph
                .search_ids(&dist, k, ef, filter, Some(&deleted), &mut visited)
        };
        self.pool.put(visited);

        // Rescore candidates with the exact f32 kernel and take the true top-k.
        let mut scored: Vec<(OrderedF32, ScoredPoint)> = candidates
            .into_iter()
            .map(|(_, id)| {
                let score = self
                    .kernel
                    .score_f32(query, self.vectors.get(PointId::new(id)));
                let badness = if higher { -score } else { score };
                (
                    OrderedF32::new(badness),
                    ScoredPoint {
                        id: PointId::new(id),
                        score,
                    },
                )
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.id.cmp(&b.1.id)));
        scored.truncate(k);
        scored.into_iter().map(|(_, sp)| sp).collect()
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
    use crate::index::BitmapFilter;
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
    fn concurrent_searches_match_serial_baseline() {
        // VisitedPool hands a reused VisitedList to each search; a clear()/generation
        // bug or a leaked list would corrupt a concurrent query's traversal. Proof:
        // every result from many threads hammering one shared index must exactly
        // equal its single-threaded baseline (search is deterministic per query).
        let dim = 24;
        let n = 1_500;
        let store = storage(dim, n, Metric::Cosine);
        let index = HnswIndex::build(store, HnswConfig::default());

        let queries: Vec<Vec<f32>> = (0..64).map(|q| vec_of(dim, 200_000 + q)).collect();
        let baseline: Vec<Vec<(u32, f32)>> = queries
            .iter()
            .map(|q| {
                index
                    .search(q, 10, SearchParams::default(), None)
                    .into_iter()
                    .map(|s| (s.id.get(), s.score))
                    .collect()
            })
            .collect();

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let index = &index;
                let queries = &queries;
                let baseline = &baseline;
                scope.spawn(move || {
                    for _ in 0..20 {
                        for (qi, q) in queries.iter().enumerate() {
                            let got: Vec<(u32, f32)> = index
                                .search(q, 10, SearchParams::default(), None)
                                .into_iter()
                                .map(|s| (s.id.get(), s.score))
                                .collect();
                            assert_eq!(
                                got, baseline[qi],
                                "concurrent search diverged from serial baseline for query {qi}"
                            );
                        }
                    }
                });
            }
        });
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

    // ----- parallel (concurrent) build -----

    /// Structural invariants every valid HNSW graph must satisfy regardless of how it
    /// was built: degree bounds, no self-loops, in-range + present neighbors, no
    /// duplicate edges, the entry sits at the top level, and (for n≥2) no point is
    /// isolated on layer 0.
    fn assert_valid_graph(index: &HnswIndex, n: usize) {
        use super::search::Graph;
        let g = index.graph();
        let cfg = index.config();
        assert_eq!(g.levels.len(), n);
        if n == 0 {
            assert!(g.entry.is_none());
            return;
        }
        let entry = g.entry.expect("non-empty graph must have an entry");
        assert_eq!(
            g.levels[entry as usize] as usize, g.max_level,
            "entry must sit at the top level"
        );
        for p in 0..n as u32 {
            let level = g.levels[p as usize] as usize;
            for layer in 0..=level {
                let nbrs = g.neighbors(p, layer);
                let max_conn = if layer == 0 { cfg.m_max0 } else { cfg.m };
                assert!(
                    nbrs.len() <= max_conn,
                    "point {p} layer {layer}: degree {} exceeds {max_conn}",
                    nbrs.len()
                );
                let mut seen = std::collections::HashSet::new();
                for &nb in nbrs {
                    assert_ne!(nb, p, "self-loop at point {p} layer {layer}");
                    assert!((nb as usize) < n, "neighbor {nb} out of range at point {p}");
                    assert!(
                        g.levels[nb as usize] as usize >= layer,
                        "neighbor {nb} absent from layer {layer}"
                    );
                    assert!(seen.insert(nb), "duplicate edge {p}->{nb} at layer {layer}");
                }
            }
            if n >= 2 {
                assert!(
                    !g.neighbors(p, 0).is_empty(),
                    "point {p} is isolated on layer 0"
                );
            }
        }
    }

    #[test]
    fn concurrent_build_edge_cases() {
        // Tiny counts exercise entry seeding, the empty/degenerate paths, and the
        // small-n branches of insert without the size threshold.
        for n in [0usize, 1, 2, 3, 5, 16, 100] {
            for metric in [Metric::Cosine, Metric::Euclidean] {
                let store = storage(8, n, metric);
                let index = HnswIndex::build_concurrent(store, HnswConfig::default());
                assert_valid_graph(&index, n);
                if n > 0 {
                    let got = index.search(&vec_of(8, 1), 5, SearchParams::default(), None);
                    assert_eq!(got.len(), n.min(5));
                }
            }
        }
    }

    #[test]
    fn concurrent_build_handles_duplicate_vectors() {
        // All-identical vectors: zero distances, lots of ties — must stay valid and
        // still return k results.
        let dim = 12;
        let n = 500;
        let mut s = VectorStorage::with_capacity(dim, Metric::Cosine, n);
        for _ in 0..n {
            s.push(&vec_of(dim, 42));
        }
        let index = HnswIndex::build_concurrent(Arc::new(s), HnswConfig::default());
        assert_valid_graph(&index, n);
        let got = index.search(&vec_of(dim, 42), 10, SearchParams::default(), None);
        assert_eq!(got.len(), 10);
    }

    #[test]
    fn concurrent_build_is_valid_and_matches_levels() {
        // The concurrent build must assign the *same* per-point levels as the
        // deterministic builder (levels are a pure function of seed^id), even though
        // the adjacency differs.
        let store = storage(24, 5_000, Metric::Cosine);
        let seq = HnswIndex::build(store.clone(), HnswConfig::default());
        let par = HnswIndex::build_concurrent(store, HnswConfig::default());
        assert_valid_graph(&par, 5_000);
        assert_eq!(seq.graph().levels, par.graph().levels, "levels must match");
    }

    #[test]
    fn parallel_build_preserves_quality() {
        // Validates the *parallelism* itself: with f32 construction (quantization
        // off), the concurrent build must not materially degrade recall vs the
        // deterministic sequential build, on a graph large enough to cross
        // PARALLEL_MIN. The parallel graph differs (nondeterministic insertion order)
        // and may even beat sequential, so the comparison is one-sided. int8
        // construction is an orthogonal feature, validated by
        // `int8_construction_preserves_recall` (and SIFT1M on real data).
        let dim = 24;
        let n = 6_000;
        let queries = 50;
        let cfg = HnswConfig {
            quantization: false,
            ..HnswConfig::default()
        };
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            let store = storage(dim, n, metric);
            let kernel = DistanceKernel::new(metric, dim);
            let seq = HnswIndex::build(store.clone(), cfg);
            let par = HnswIndex::build_parallel(store.clone(), cfg);
            assert_valid_graph(&par, n);

            let (mut seq_total, mut par_total) = (0.0f32, 0.0f32);
            for q in 0..queries {
                let query = vec_of(dim, 200_000 + q);
                let truth = brute_force_topk(&store, &kernel, &query, 10, None, None);
                seq_total += recall_at(
                    &seq.search(&query, 10, SearchParams::default(), None),
                    &truth,
                );
                par_total += recall_at(
                    &par.search(&query, 10, SearchParams::default(), None),
                    &truth,
                );
            }
            let (seq_recall, par_recall) = (seq_total / queries as f32, par_total / queries as f32);
            assert!(
                par_recall >= seq_recall - 0.05,
                "metric={metric}: parallel recall {par_recall:.3} materially below sequential {seq_recall:.3}"
            );
            if matches!(metric, Metric::Cosine | Metric::Dot) {
                assert!(
                    par_recall >= 0.95,
                    "metric={metric}: parallel recall@10 {par_recall:.3} < 0.95"
                );
            }
        }
    }

    #[test]
    fn concurrent_build_is_stable_across_repeats() {
        // The concurrent builder merges own-links and pushes reverse-links under
        // per-node locks. A lost-update / overwrite-instead-of-merge race would
        // surface non-deterministically as a dropped node or a recall dip on some
        // runs. Repeat the multi-threaded build many times; every run must keep all
        // nodes and stay above the recall floor.
        let dim = 24;
        let n = 1_200;
        let store = storage(dim, n, Metric::Cosine);
        let kernel = DistanceKernel::new(Metric::Cosine, dim);
        let queries: Vec<Vec<f32>> = (0..20).map(|q| vec_of(dim, 300_000 + q)).collect();
        let truths: Vec<Vec<ScoredPoint>> = queries
            .iter()
            .map(|q| brute_force_topk(&store, &kernel, q, 10, None, None))
            .collect();

        for run in 0..16 {
            let index = HnswIndex::build_concurrent(store.clone(), HnswConfig::default());
            assert_eq!(
                index.graph().levels.len(),
                n,
                "run {run}: a node was lost during concurrent build"
            );
            let mut total = 0.0f32;
            for (q, truth) in queries.iter().zip(&truths) {
                let got = index.search(q, 10, SearchParams::default(), None);
                assert_eq!(got.len(), 10);
                total += recall_at(&got, truth);
            }
            let recall = total / queries.len() as f32;
            assert!(
                recall >= 0.90,
                "run {run}: concurrent-build recall {recall:.3} < 0.90"
            );
        }
    }

    #[test]
    fn int8_construction_preserves_recall() {
        // The default parallel build uses int8 codes for its (memory-bound)
        // construction distances. That must not materially hurt recall vs f32
        // construction, on data where int8 quantization is representative (≥64-dim;
        // very low dims are lossy for int8 and unrepresentative). Compares the two
        // parallel builds directly, so it's independent of the data's absolute recall.
        let dim = 64;
        let n = 5_000;
        let queries = 60;
        let f32_cfg = HnswConfig {
            quantization: false,
            ..HnswConfig::default()
        };
        let int8_cfg = HnswConfig::default(); // quantization on
        for metric in [Metric::Cosine, Metric::Euclidean] {
            let store = storage(dim, n, metric);
            let kernel = DistanceKernel::new(metric, dim);
            let f32_idx = HnswIndex::build_parallel(store.clone(), f32_cfg);
            let int8_idx = HnswIndex::build_parallel(store.clone(), int8_cfg);

            let (mut f32_total, mut int8_total) = (0.0f32, 0.0f32);
            for q in 0..queries {
                let query = vec_of(dim, 300_000 + q);
                let truth = brute_force_topk(&store, &kernel, &query, 10, None, None);
                f32_total += recall_at(
                    &f32_idx.search(&query, 10, SearchParams::default(), None),
                    &truth,
                );
                int8_total += recall_at(
                    &int8_idx.search(&query, 10, SearchParams::default(), None),
                    &truth,
                );
            }
            let (f32_recall, int8_recall) =
                (f32_total / queries as f32, int8_total / queries as f32);
            // Coarse "int8 construction isn't broken" gate: the synthetic euclidean
            // set is noisy run-to-run for *both* builds (f32 itself swings ~±0.05), so
            // the margin is loose. SIFT1M validates int8 euclidean precisely (~0.99).
            assert!(
                int8_recall >= f32_recall - 0.10,
                "metric={metric}: int8-construction recall {int8_recall:.3} far below f32 {f32_recall:.3}"
            );
            // Cosine is stable here, so hold it to a tight absolute bar.
            if metric == Metric::Cosine {
                assert!(
                    int8_recall >= 0.95,
                    "metric={metric}: int8 recall {int8_recall:.3} < 0.95"
                );
            }
        }
    }

    #[test]
    fn concurrent_build_preserves_back_links() {
        // The parallel builder MERGES a point's own links with concurrently-pushed
        // reverse-links rather than overwriting them. HNSW wires edges both ways, so
        // the layer-0 mutual-link ratio is the cheap proxy that back-links survived.
        // Pruning legitimately breaks *some* symmetry, so rather than a fragile
        // absolute we anchor to the SEQUENTIAL build (same merge, no races) as the
        // baseline: an `*list = selected` overwrite regression — which only races
        // under concurrency — would collapse the concurrent ratio well below it.
        let dim = 24;
        let n = 3_000;
        let cfg = HnswConfig {
            quantization: false,
            ..HnswConfig::default()
        };
        let store = storage(dim, n, Metric::Cosine);

        use super::search::Graph;
        let mutual_ratio = |g: &GraphLayers| {
            let (mut total, mut mutual) = (0usize, 0usize);
            for p in 0..n as u32 {
                for &q in g.neighbors(p, 0) {
                    total += 1;
                    if g.neighbors(q, 0).contains(&p) {
                        mutual += 1;
                    }
                }
            }
            mutual as f32 / total.max(1) as f32
        };

        let seq = HnswIndex::build(store.clone(), cfg);
        let par = HnswIndex::build_concurrent(store, cfg);
        assert_valid_graph(&par, n);
        let seq_ratio = mutual_ratio(seq.graph());
        let par_ratio = mutual_ratio(par.graph());

        assert!(
            seq_ratio >= 0.5,
            "sanity: sequential mutual-link ratio {seq_ratio:.3} unexpectedly low"
        );
        assert!(
            par_ratio >= seq_ratio - 0.12,
            "concurrent mutual-link ratio {par_ratio:.3} materially below sequential \
             {seq_ratio:.3} — back-links being dropped (overwrite instead of merge)?"
        );
    }

    #[test]
    fn filtered_search_traverses_rejected_nodes_for_connectivity() {
        // Filtered HNSW must keep TRAVERSING through rejected nodes (excluding them
        // only from results), so matches reachable only by hopping through rejected
        // nodes are still found. HnswIndex::search has no exact-scan fallback (that
        // lives in SealedSegment), so this exercises the admit/traverse split directly:
        // if `admit` pruned traversal, filtered recall would collapse.
        let dim = 24;
        let n = 1_000;
        let metric = Metric::Cosine;
        let store = storage(dim, n, metric);
        let kernel = DistanceKernel::new(metric, dim);
        let index = HnswIndex::build(store.clone(), HnswConfig::default());

        // Admit a scattered ~5% of ids (only multiples of 20).
        let filter =
            BitmapFilter::from_ids((0..n as u32).filter(|i| i % 20 == 0).map(PointId::new));

        let queries = 20;
        let mut total = 0.0f32;
        for q in 0..queries {
            let query = vec_of(dim, 400_000 + q);
            let got = index.search(&query, 10, SearchParams::default(), Some(&filter));
            assert_eq!(got.len(), 10, "filtered search under-filled (q={q})");
            assert!(
                got.iter().all(|s| s.id.get() % 20 == 0),
                "filtered search returned a rejected id"
            );
            let truth = brute_force_topk(&store, &kernel, &query, 10, None, Some(&filter));
            total += recall_at(&got, &truth);
        }
        let recall = total / queries as f32;
        assert!(
            recall >= 0.9,
            "filtered recall {recall:.3} < 0.9 — admit may be pruning traversal"
        );
    }

    #[test]
    fn quantized_search_returns_exact_rescored_topk() {
        // The quantized path over-fetches by int8 badness then rescores with the exact
        // f32 kernel. Assert the rescore actually (a) attaches the TRUE f32 score (not
        // leftover quantized badness), (b) orders best-first with the correct sign per
        // metric, and (c) for queries the graph nails (recall 1.0) returns exactly the
        // exact top-10 ids in exact order. A wrong rescore sort key would break (b)/(c)
        // and a missing rescore would break (a) — none caught by set-membership recall.
        let dim = 24;
        let n = 2_000;
        let queries = 50;
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            let store = storage(dim, n, metric);
            let kernel = DistanceKernel::new(metric, dim);
            let index = HnswIndex::build(store.clone(), HnswConfig::default()); // quantization on
            let higher = metric.higher_is_better();

            let mut exact_hits = 0;
            for q in 0..queries {
                let query = vec_of(dim, 500_000 + q);
                let got = index.search(&query, 10, SearchParams::default(), None);
                assert_eq!(got.len(), 10);

                // (a) Each attached score is the exact f32 score of that point.
                for sp in &got {
                    let exact = kernel.score_f32(&query, store.get(sp.id));
                    assert!(
                        (sp.score - exact).abs() < 1e-4,
                        "metric={metric}: attached score {} != exact f32 score {exact}",
                        sp.score
                    );
                }
                // (b) Ordered best-first by exact score with the correct polarity.
                for w in got.windows(2) {
                    if higher {
                        assert!(w[0].score >= w[1].score - 1e-6, "metric={metric}: not desc");
                    } else {
                        assert!(w[0].score <= w[1].score + 1e-6, "metric={metric}: not asc");
                    }
                }
                // (c) When the graph nails it, ids+order match the exact truth.
                let truth = brute_force_topk(&store, &kernel, &query, 10, None, None);
                if recall_at(&got, &truth) == 1.0 {
                    let got_ids: Vec<u32> = got.iter().map(|s| s.id.get()).collect();
                    let truth_ids: Vec<u32> = truth.iter().map(|s| s.id.get()).collect();
                    assert_eq!(
                        got_ids, truth_ids,
                        "metric={metric}: rescored order != exact"
                    );
                    exact_hits += 1;
                }
            }
            assert!(
                exact_hits > 0,
                "metric={metric}: no query hit recall 1.0; cannot verify exact order"
            );
        }
    }
}
