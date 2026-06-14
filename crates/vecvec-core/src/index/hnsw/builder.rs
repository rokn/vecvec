//! HNSW graph construction.
//!
//! [`GraphLayersBuilder`] inserts points one at a time in id order, holding mutable
//! per-point/per-layer adjacency lists. Insertion is **sequential and
//! deterministic**: levels come from the pure [`level_for`] function and every
//! candidate set is sorted before use, so a given `(vectors, seed, config)` always
//! yields the identical graph — which is what makes single- vs multi-threaded
//! builds agree and lets WAL replay reproduce an index. (Concurrent construction
//! with per-point locks is a later optimization; it must preserve this result.)
//!
//! Sealing flattens the adjacency into the read-optimized [`GraphLayers`].

use std::sync::Arc;

use crate::distance::DistanceKernel;
use crate::id::PointId;
use crate::vector::VectorStorage;

use super::HnswConfig;
use super::graph::GraphLayers;
use super::heuristic::{candidates_to, select_neighbors};
use super::rng::level_for;
use super::search::{Graph, search_layer};
use super::visited::VisitedList;

/// Builds an HNSW graph by sequential insertion.
pub(crate) struct GraphLayersBuilder {
    vectors: Arc<VectorStorage>,
    kernel: DistanceKernel,
    config: HnswConfig,
    higher: bool,
    /// Assigned level per point.
    levels: Vec<u8>,
    /// `links[point][layer]` = neighbor ids; `layer` ranges `0..=levels[point]`.
    links: Vec<Vec<Vec<u32>>>,
    entry: Option<u32>,
    max_level: usize,
}

impl Graph for GraphLayersBuilder {
    #[inline]
    fn neighbors(&self, point: u32, layer: usize) -> &[u32] {
        self.links[point as usize]
            .get(layer)
            .map_or(&[], |v| v.as_slice())
    }
}

impl GraphLayersBuilder {
    /// Creates a builder over `vectors`, pre-assigning every point's level.
    pub(crate) fn new(
        vectors: Arc<VectorStorage>,
        kernel: DistanceKernel,
        config: HnswConfig,
    ) -> Self {
        let n = vectors.len();
        let ml = config.ml();
        let levels: Vec<u8> = (0..n)
            .map(|i| level_for(i as u32, config.seed, ml) as u8)
            .collect();
        let links: Vec<Vec<Vec<u32>>> = levels
            .iter()
            .map(|&lv| vec![Vec::new(); lv as usize + 1])
            .collect();
        let higher = kernel.metric().higher_is_better();
        Self {
            vectors,
            kernel,
            config,
            higher,
            levels,
            links,
            entry: None,
            max_level: 0,
        }
    }

    fn max_conn(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max0
        } else {
            self.config.m
        }
    }

    /// Inserts `point` into the graph.
    pub(crate) fn insert(&mut self, point: u32, visited: &mut VisitedList) {
        let level = self.levels[point as usize] as usize;

        let Some(entry) = self.entry else {
            // First point: it becomes the entry.
            self.entry = Some(point);
            self.max_level = level;
            return;
        };

        // Self-contained distance closure (owns its captures: no borrow of `self`,
        // so the adjacency can be mutated afterwards).
        let kernel = self.kernel;
        let vectors = self.vectors.clone();
        let query = vectors.get(PointId::new(point)).to_vec();
        let higher = self.higher;
        let dist = move |id: u32| {
            let s = kernel.score_f32(&query, vectors.get(PointId::new(id)));
            if higher { -s } else { s }
        };
        let admit_all = |_: u32| true;

        let start_level = self.max_level;

        // Phase 1: greedy descent from the top down to just above this point's level.
        let mut ep = vec![entry];
        for layer in ((level + 1)..=start_level).rev() {
            let w = search_layer(self, layer, &ep, 1, &dist, &admit_all, visited);
            if let Some(&(_, best)) = w.first() {
                ep = vec![best];
            }
        }

        // Phase 2: from min(level, top) down to 0, find neighbors and wire up.
        let top = level.min(start_level);
        for layer in (0..=top).rev() {
            let w = search_layer(
                self,
                layer,
                &ep,
                self.config.ef_construction,
                &dist,
                &admit_all,
                visited,
            );
            let max_conn = self.max_conn(layer);
            let selected = select_neighbors(
                &self.vectors,
                &self.kernel,
                point,
                &w,
                max_conn,
                self.config.keep_pruned,
                self.higher,
            );

            // Wire the new point to its selected neighbors.
            self.links[point as usize][layer] = selected.clone();
            for &nb in &selected {
                self.links[nb as usize][layer].push(point);
                // Shrink the neighbor back to its degree bound if it overflowed.
                let nb_max = self.max_conn(layer);
                if self.links[nb as usize][layer].len() > nb_max {
                    let current = self.links[nb as usize][layer].clone();
                    let cand =
                        candidates_to(&self.vectors, &self.kernel, nb, &current, self.higher);
                    let reselected = select_neighbors(
                        &self.vectors,
                        &self.kernel,
                        nb,
                        &cand,
                        nb_max,
                        self.config.keep_pruned,
                        self.higher,
                    );
                    self.links[nb as usize][layer] = reselected;
                }
            }

            // The found set seeds the next (lower) layer's search.
            ep = w.iter().map(|&(_, id)| id).collect();
        }

        // Raise the global entry point if this point reaches a new top layer.
        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(point);
        }
    }

    /// Seals the mutable adjacency into the read-optimized [`GraphLayers`].
    pub(crate) fn into_graph_layers(self) -> GraphLayers {
        let n = self.levels.len();
        let mut l0_offsets = Vec::with_capacity(n + 1);
        let mut l0_links = Vec::new();
        l0_offsets.push(0u32);
        for point_links in &self.links {
            if let Some(layer0) = point_links.first() {
                l0_links.extend_from_slice(layer0);
            }
            l0_offsets.push(l0_links.len() as u32);
        }

        let mut upper: Vec<Vec<Vec<u32>>> = Vec::new();
        let mut upper_index = vec![u32::MAX; n];
        for (p, point_links) in self.links.iter().enumerate() {
            if point_links.len() > 1 {
                upper_index[p] = upper.len() as u32;
                upper.push(point_links[1..].to_vec());
            }
        }

        GraphLayers {
            entry: self.entry,
            max_level: self.max_level,
            levels: self.levels,
            l0_offsets,
            l0_links,
            upper,
            upper_index,
        }
    }
}
