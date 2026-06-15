//! Concurrent HNSW construction.
//!
//! [`build_concurrent_graph`] inserts points in parallel across the rayon pool. It
//! produces a *valid* HNSW graph — degree bounds respected, Alg-4 neighbor
//! selection, every point linked — but, unlike the sequential [`super::builder`], it
//! is **not** byte-identical: insertion order is nondeterministic, so the result is
//! validated by recall and structural invariants rather than equality.
//!
//! Synchronization:
//! - Each point's adjacency (`Vec<Vec<u32>>`, indexed by layer) sits behind its own
//!   [`Mutex`]. Searches copy a node's neighbor list out under that lock, so no
//!   borrow ever outlives the guard (no data race on the `Vec`).
//! - The global entry point + top level sit behind an [`RwLock`]: read once per
//!   insert, written only when a point reaches a new top layer (≈ `log n` times).
//! - When a point writes its *own* links we **merge** with whatever back-links a
//!   concurrent insert already pushed, instead of overwriting — otherwise the race
//!   `own = selected` vs. `neighbor.push(point)` would silently drop edges.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use parking_lot::RwLock;
use rayon::prelude::*;

use crate::distance::DistanceKernel;
use crate::id::PointId;
use crate::ordered::OrderedF32;
use crate::quantization::QuantizedVectorBlock;
use crate::vector::VectorStorage;

use super::HnswConfig;
use super::graph::GraphLayers;
use super::rng::level_for;
use super::visited::VisitedList;

/// The global navigable-entry state, mutated only when a point raises the top level.
struct EntryState {
    entry: Option<u32>,
    max_level: usize,
}

/// A builder that inserts points concurrently, guarding per-point adjacency.
///
/// All construction distances go through [`ConcurrentBuilder::dist_ids`]: when an
/// int8 [`QuantizedVectorBlock`] is supplied, comparisons read 4×-smaller codes,
/// which is a large win because the build is memory-latency/bandwidth-bound. The f32
/// rescore at *search* time recovers the precision lost during construction.
struct ConcurrentBuilder<'q> {
    vectors: Arc<VectorStorage>,
    kernel: DistanceKernel,
    config: HnswConfig,
    higher: bool,
    /// Optional int8 codes used for the (memory-bound) construction distances.
    quant: Option<&'q QuantizedVectorBlock>,
    levels: Vec<u8>,
    /// `links[point]` (locked) holds the point's neighbor lists, indexed by layer.
    links: Vec<RwLock<Vec<Vec<u32>>>>,
    entry: RwLock<EntryState>,
}

impl<'q> ConcurrentBuilder<'q> {
    fn new(
        vectors: Arc<VectorStorage>,
        kernel: DistanceKernel,
        config: HnswConfig,
        quant: Option<&'q QuantizedVectorBlock>,
    ) -> Self {
        let n = vectors.len();
        let ml = config.ml();
        let levels: Vec<u8> = (0..n)
            .map(|i| level_for(i as u32, config.seed, ml) as u8)
            .collect();
        let links: Vec<RwLock<Vec<Vec<u32>>>> = levels
            .iter()
            .map(|&lv| RwLock::new(vec![Vec::new(); lv as usize + 1]))
            .collect();
        let higher = kernel.metric().higher_is_better();
        Self {
            vectors,
            kernel,
            config,
            higher,
            quant,
            levels,
            links,
            entry: RwLock::new(EntryState {
                entry: None,
                max_level: 0,
            }),
        }
    }

    #[inline]
    fn max_conn(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max0
        } else {
            self.config.m
        }
    }

    /// Construction "badness" (smaller = closer) between two stored points by id,
    /// using int8 codes when available, else the f32 kernel.
    #[inline]
    fn dist_ids(&self, a: u32, b: u32) -> f32 {
        if let Some(q) = self.quant {
            // SAFETY: `a`/`b` are existing graph rows (`< len`).
            unsafe { q.badness_between(a, b) }
        } else {
            // SAFETY: same in-range invariant.
            let (va, vb) = unsafe {
                (
                    self.vectors.get_unchecked(PointId::new(a)),
                    self.vectors.get_unchecked(PointId::new(b)),
                )
            };
            let s = self.kernel.score_f32(va, vb);
            if self.higher { -s } else { s }
        }
    }

    /// Alg-4 neighbor selection over `candidates` (`(badness_to_base, id)`), using
    /// id-based distances (so it honors `dist_ids`'s int8/f32 choice). Mirrors
    /// [`super::heuristic::select_neighbors`].
    fn select_ids(&self, base: u32, candidates: &[(f32, u32)], m: usize) -> Vec<u32> {
        let mut sorted: Vec<(OrderedF32, u32)> = candidates
            .iter()
            .filter(|&&(_, id)| id != base)
            .map(|&(b, id)| (OrderedF32::new(b), id))
            .collect();
        sorted.sort_unstable();

        let mut result: Vec<u32> = Vec::with_capacity(m);
        let mut discarded: Vec<u32> = Vec::new();
        for (dist_to_base, e) in sorted {
            if result.len() >= m {
                break;
            }
            let mut keep = true;
            for &r in &result {
                if self.dist_ids(e, r) < dist_to_base.into_inner() {
                    keep = false;
                    break;
                }
            }
            if keep {
                result.push(e);
            } else if self.config.keep_pruned {
                discarded.push(e);
            }
        }
        if self.config.keep_pruned {
            for e in discarded {
                if result.len() >= m {
                    break;
                }
                result.push(e);
            }
        }
        result
    }

    /// Best-first beam search of one `layer` over the (concurrently mutating) graph.
    /// Mirrors [`super::search::search_layer`] but copies each visited node's
    /// neighbor list out under its lock (into the reusable `nbuf`). Admits all nodes
    /// — construction never has deletes or filters.
    fn search_layer<D: Fn(u32) -> f32>(
        &self,
        layer: usize,
        entry_points: &[u32],
        ef: usize,
        dist: &D,
        visited: &mut VisitedList,
        nbuf: &mut Vec<u32>,
    ) -> Vec<(f32, u32)> {
        visited.clear();
        let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::new();

        for &e in entry_points {
            if visited.visit(e) {
                let od = OrderedF32::new(dist(e));
                candidates.push(Reverse((od, e)));
                results.push((od, e));
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(Reverse((cand_d, c))) = candidates.pop() {
            if let Some(&(worst, _)) = results.peek()
                && results.len() >= ef
                && cand_d > worst
            {
                break;
            }
            nbuf.clear();
            {
                let g = self.links[c as usize].read();
                if let Some(v) = g.get(layer) {
                    nbuf.extend_from_slice(v);
                }
            }
            for &nn in nbuf.iter() {
                if visited.visit(nn) {
                    let od = OrderedF32::new(dist(nn));
                    let explore = match results.peek() {
                        Some(&(worst, _)) => results.len() < ef || od < worst,
                        None => true,
                    };
                    if explore {
                        candidates.push(Reverse((od, nn)));
                        results.push((od, nn));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(OrderedF32, u32)> = results.into_vec();
        out.sort_unstable();
        out.into_iter()
            .map(|(b, id)| (b.into_inner(), id))
            .collect()
    }

    /// Re-applies the degree bound to `list` (a node's neighbor list at `layer`),
    /// re-running Alg-4 selection over its current members if it overflowed.
    fn enforce_bound(&self, base: u32, layer: usize, list: &mut Vec<u32>) {
        let max_conn = self.max_conn(layer);
        if list.len() > max_conn {
            let cand: Vec<(f32, u32)> = list
                .iter()
                .map(|&id| (self.dist_ids(base, id), id))
                .collect();
            *list = self.select_ids(base, &cand, max_conn);
        }
    }

    /// Inserts `point`. Requires the entry to already be seeded (see
    /// [`build_concurrent_graph`]).
    fn insert(&self, point: u32, visited: &mut VisitedList, nbuf: &mut Vec<u32>) {
        let level = self.levels[point as usize] as usize;
        let (entry, start_level) = {
            let g = self.entry.read();
            match g.entry {
                Some(e) => (e, g.max_level),
                None => return, // entry not seeded yet — caller guarantees it is
            }
        };

        // Distance closure: all construction distances go through `dist_ids`, which
        // uses int8 codes when available (the memory-bound win).
        let dist = |id: u32| self.dist_ids(point, id);

        // Phase 1: greedy descent from the (snapshotted) top to just above `level`.
        let mut ep = vec![entry];
        for layer in ((level + 1)..=start_level).rev() {
            let w = self.search_layer(layer, &ep, 1, &dist, visited, nbuf);
            if let Some(&(_, best)) = w.first() {
                ep = vec![best];
            }
        }

        // Phase 2: from min(level, top) down to 0, find neighbors and wire up.
        let top = level.min(start_level);
        for layer in (0..=top).rev() {
            let w = self.search_layer(
                layer,
                &ep,
                self.config.ef_construction,
                &dist,
                visited,
                nbuf,
            );
            let max_conn = self.max_conn(layer);
            let selected = self.select_ids(point, &w, max_conn);

            // Our own links: merge with any back-links already pushed concurrently,
            // then re-bound. Merging (not overwriting) is what makes the parallel
            // build lose no edges.
            {
                let mut own = self.links[point as usize].write();
                let list = &mut own[layer];
                for &s in &selected {
                    if !list.contains(&s) {
                        list.push(s);
                    }
                }
                self.enforce_bound(point, layer, list);
            }

            // Reverse links: push us onto each selected neighbor, re-bounding it with
            // the full Alg-4 heuristic (its diversity is essential for recall).
            for &nb in &selected {
                let mut g = self.links[nb as usize].write();
                if !g[layer].contains(&point) {
                    g[layer].push(point);
                }
                let mut list = std::mem::take(&mut g[layer]);
                self.enforce_bound(nb, layer, &mut list);
                g[layer] = list;
            }

            ep = w.iter().map(|&(_, id)| id).collect();
        }

        // Raise the global entry if this point reaches a new top layer.
        if level > start_level {
            let mut g = self.entry.write();
            if level > g.max_level {
                g.max_level = level;
                g.entry = Some(point);
            }
        }
    }

    /// Flattens the locked adjacency into the read-optimized [`GraphLayers`]
    /// (identical CSR layout to the sequential builder's seal).
    fn into_graph_layers(self) -> GraphLayers {
        let n = self.levels.len();
        let entry_state = self.entry.into_inner();
        let links: Vec<Vec<Vec<u32>>> = self.links.into_iter().map(|m| m.into_inner()).collect();

        let mut l0_offsets = Vec::with_capacity(n + 1);
        let mut l0_links = Vec::new();
        l0_offsets.push(0u32);
        for point_links in &links {
            if let Some(layer0) = point_links.first() {
                l0_links.extend_from_slice(layer0);
            }
            l0_offsets.push(l0_links.len() as u32);
        }

        let mut upper: Vec<Vec<Vec<u32>>> = Vec::new();
        let mut upper_index = vec![u32::MAX; n];
        for (p, point_links) in links.iter().enumerate() {
            if point_links.len() > 1 {
                upper_index[p] = upper.len() as u32;
                upper.push(point_links[1..].to_vec());
            }
        }

        GraphLayers {
            entry: entry_state.entry,
            max_level: entry_state.max_level,
            levels: self.levels,
            l0_offsets,
            l0_links,
            upper,
            upper_index,
        }
    }
}

/// Builds an HNSW [`GraphLayers`] by parallel insertion across the rayon pool.
/// Point 0 seeds the entry sequentially (it has no links, like the sequential
/// builder's first insert); the rest are inserted concurrently. When `quant` is
/// supplied, construction distances use its int8 codes (memory-bound speedup).
pub(crate) fn build_concurrent_graph(
    vectors: Arc<VectorStorage>,
    kernel: DistanceKernel,
    config: HnswConfig,
    quant: Option<&QuantizedVectorBlock>,
) -> GraphLayers {
    let n = vectors.len();
    let builder = ConcurrentBuilder::new(vectors, kernel, config, quant);
    if n == 0 {
        return builder.into_graph_layers();
    }
    {
        let mut g = builder.entry.write();
        g.entry = Some(0);
        g.max_level = builder.levels[0] as usize;
    }
    (1..n as u32).into_par_iter().for_each_init(
        || (VisitedList::new(n.max(1)), Vec::<u32>::new()),
        |(visited, nbuf), point| builder.insert(point, visited, nbuf),
    );
    builder.into_graph_layers()
}
