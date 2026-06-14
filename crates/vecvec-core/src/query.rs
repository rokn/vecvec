//! Result fusion for multi-stage / hybrid queries.
//!
//! Reciprocal Rank Fusion (RRF) combines several ranked result lists into one by
//! summing `1 / (k + rank)` across lists — robust because it uses ranks, not raw
//! scores, so lists with incomparable score scales (e.g. dense vs sparse) fuse
//! cleanly. This is the combiner a `prefetch` DAG (deferred) would use; exposed now
//! so callers can fuse multiple `query`/`recommend` result sets.

use std::collections::HashMap;

/// The conventional RRF constant.
pub const DEFAULT_RRF_K: f64 = 60.0;

/// Fuses ranked `(id, score)` lists via Reciprocal Rank Fusion, returning the top
/// `top` ids by fused score. `k` damps the contribution of low ranks.
pub fn reciprocal_rank_fusion(lists: &[Vec<(u64, f32)>], k: f64, top: usize) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f64> = HashMap::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (k + (rank + 1) as f64);
        }
    }
    let mut fused: Vec<(u64, f32)> = scores.into_iter().map(|(id, s)| (id, s as f32)).collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    fused.truncate(top);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuses_by_rank() {
        // Id 2 is top of list A and second of list B -> should win overall.
        let a = vec![(2, 0.9), (1, 0.8), (3, 0.1)];
        let b = vec![(5, 0.9), (2, 0.7), (1, 0.6)];
        let fused = reciprocal_rank_fusion(&[a, b], DEFAULT_RRF_K, 3);
        assert_eq!(fused[0].0, 2);
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn single_list_preserves_order() {
        let a = vec![(7, 0.5), (8, 0.4), (9, 0.3)];
        let fused = reciprocal_rank_fusion(&[a], DEFAULT_RRF_K, 10);
        assert_eq!(fused.iter().map(|x| x.0).collect::<Vec<_>>(), vec![7, 8, 9]);
    }
}
