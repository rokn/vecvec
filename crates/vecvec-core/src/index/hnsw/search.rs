//! The layer search routine, shared by graph construction and query.
//!
//! [`search_layer`] is the heart of HNSW: a best-first beam search of width `ef`
//! over a single layer. It works over any [`Graph`] (the mutable builder during
//! construction, the sealed [`GraphLayers`](super::graph::GraphLayers) at query
//! time), parameterized by a distance closure and an `admit` predicate (so deleted
//! / filtered points are traversed for connectivity but not collected).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::ordered::OrderedF32;

use super::visited::VisitedList;

/// A read view of an HNSW graph's adjacency.
pub(crate) trait Graph {
    /// Neighbors of `point` on `layer` (empty if the point isn't on that layer).
    fn neighbors(&self, point: u32, layer: usize) -> &[u32];
}

/// Best-first search of one `layer`, returning the admitted results closest to the
/// query as `(badness, id)` sorted best-first (smallest badness, then id).
///
/// `dist(id)` returns badness (smaller = closer); `admit(id)` decides whether a
/// node may enter the result set (non-admitted nodes are still traversed).
pub(crate) fn search_layer<G, D, A>(
    graph: &G,
    layer: usize,
    entry_points: &[u32],
    ef: usize,
    dist: &D,
    admit: &A,
    visited: &mut VisitedList,
) -> Vec<(f32, u32)>
where
    G: Graph,
    D: Fn(u32) -> f32,
    A: Fn(u32) -> bool,
{
    visited.clear();
    // Exploration frontier: min-heap by badness (closest popped first).
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::new();
    // Result beam: max-heap by badness (farthest at the root, evicted first).
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::new();

    for &e in entry_points {
        if visited.visit(e) {
            let od = OrderedF32::new(dist(e));
            candidates.push(Reverse((od, e)));
            if admit(e) {
                results.push((od, e));
                if results.len() > ef {
                    results.pop();
                }
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
        for &n in graph.neighbors(c, layer) {
            if visited.visit(n) {
                let od = OrderedF32::new(dist(n));
                let explore = match results.peek() {
                    Some(&(worst, _)) => results.len() < ef || od < worst,
                    None => true,
                };
                if explore {
                    candidates.push(Reverse((od, n)));
                    if admit(n) {
                        results.push((od, n));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
    }

    let mut out: Vec<(OrderedF32, u32)> = results.into_vec();
    out.sort_unstable(); // ascending (badness, id) == best-first
    out.into_iter()
        .map(|(b, id)| (b.into_inner(), id))
        .collect()
}
