//! Neighbor selection (Malkov & Yashunin, Algorithm 4).
//!
//! When wiring a point to its candidates, naively taking the `M` closest produces
//! clustered, poorly-navigable graphs. The heuristic instead prefers a *diverse*
//! set: a candidate `e` is kept only if it is closer to the base point than to any
//! already-selected neighbor — spreading connections across directions. With
//! `keep_pruned` we top up from the discarded candidates so the degree target is
//! still met.
//!
//! All inputs are sorted by `(distance, id)` so selection is fully deterministic.

use crate::distance::DistanceKernel;
use crate::id::PointId;
use crate::ordered::OrderedF32;
use crate::vector::VectorStorage;

/// Converts a raw metric score into a "badness" where smaller is closer (negating
/// for higher-is-better metrics), so all of HNSW can reason with one polarity.
#[inline]
pub(crate) fn badness(score: f32, higher_is_better: bool) -> f32 {
    if higher_is_better { -score } else { score }
}

/// Builds a `(badness_to_base, id)` candidate list for `ids` relative to `base`.
pub(crate) fn candidates_to(
    vectors: &VectorStorage,
    kernel: &DistanceKernel,
    base: u32,
    ids: &[u32],
    higher: bool,
) -> Vec<(f32, u32)> {
    // SAFETY: `base` and every `id` are existing graph rows, so `< vectors.len()`.
    let base_vec = unsafe { vectors.get_unchecked(PointId::new(base)) };
    ids.iter()
        .map(|&id| {
            let s = kernel.score_f32(base_vec, unsafe {
                vectors.get_unchecked(PointId::new(id))
            });
            (badness(s, higher), id)
        })
        .collect()
}

/// Selects up to `m` neighbors for `base` from `candidates` (`(badness_to_base,
/// id)` pairs) using Algorithm 4. Deterministic.
pub(crate) fn select_neighbors(
    vectors: &VectorStorage,
    kernel: &DistanceKernel,
    base: u32,
    candidates: &[(f32, u32)],
    m: usize,
    keep_pruned: bool,
    higher: bool,
) -> Vec<u32> {
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
        // SAFETY: `e` and every selected `r` are existing graph rows (`< len`).
        let e_vec = unsafe { vectors.get_unchecked(PointId::new(e)) };
        let mut keep = true;
        for &r in &result {
            let d_er = badness(
                kernel.score_f32(e_vec, unsafe { vectors.get_unchecked(PointId::new(r)) }),
                higher,
            );
            // e is closer to an already-selected neighbor than to the base -> prune.
            if d_er < dist_to_base.into_inner() {
                keep = false;
                break;
            }
        }
        if keep {
            result.push(e);
        } else if keep_pruned {
            discarded.push(e);
        }
    }

    if keep_pruned {
        for e in discarded {
            if result.len() >= m {
                break;
            }
            result.push(e);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn respects_degree_bound_and_subset() {
        // Points on a line; base at origin.
        let mut s = VectorStorage::new(1, Metric::Euclidean);
        for x in [0.0f32, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0] {
            s.push(&[x]);
        }
        let kernel = DistanceKernel::new(Metric::Euclidean, 1);
        let ids: Vec<u32> = (1..7).collect();
        let cand = candidates_to(&s, &kernel, 0, &ids, false);
        let sel = select_neighbors(&s, &kernel, 0, &cand, 3, true, false);
        assert!(sel.len() <= 3);
        assert!(sel.iter().all(|id| ids.contains(id)));
        // The two immediate neighbors (id 1 at +1, id 5 at -1) should be selected:
        // diverse directions beat piling up on one side.
        assert!(sel.contains(&1));
        assert!(sel.contains(&5));
    }

    #[test]
    fn keep_pruned_fills_to_m() {
        // Many candidates clustered on one side: heuristic prunes, keep_pruned refills.
        let mut s = VectorStorage::new(1, Metric::Euclidean);
        for x in [0.0f32, 1.0, 1.1, 1.2, 1.3] {
            s.push(&[x]);
        }
        let kernel = DistanceKernel::new(Metric::Euclidean, 1);
        let ids: Vec<u32> = vec![1, 2, 3, 4];
        let cand = candidates_to(&s, &kernel, 0, &ids, false);
        let with = select_neighbors(&s, &kernel, 0, &cand, 4, true, false);
        let without = select_neighbors(&s, &kernel, 0, &cand, 4, false, false);
        assert_eq!(with.len(), 4); // refilled to m
        assert!(without.len() <= with.len());
    }
}
