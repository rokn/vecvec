//! Deterministic per-point level assignment.
//!
//! HNSW assigns each point a random maximum layer drawn from an exponential
//! distribution with parameter `mL = 1/ln(M)`. For reproducible builds (so a graph
//! can be rebuilt identically during WAL replay and so single- vs multi-threaded
//! builds agree) the "randomness" must be a **pure function of the point id and a
//! seed**, never a shared RNG advanced during insertion.

/// SplitMix64 — a fast, well-distributed integer hash used to derive a uniform
/// value from `(seed, id)`.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The maximum layer level we ever assign, a safety clamp on the exponential tail.
pub(crate) const MAX_LEVEL: usize = 16;

/// Deterministically assigns the layer level for `id` given `seed` and `ml`.
///
/// `level = floor(-ln(u) * ml)` with `u` uniform in `(0, 1]`, clamped to
/// [`MAX_LEVEL`]. Pure: depends only on `(id, seed, ml)`.
#[inline]
pub(crate) fn level_for(id: u32, seed: u64, ml: f64) -> usize {
    let h = splitmix64(seed ^ (u64::from(id).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // Top 53 bits -> uniform double in [0, 1).
    let mut u = ((h >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0); // 2^53
    if u <= 0.0 {
        u = f64::MIN_POSITIVE;
    }
    let level = (-u.ln() * ml).floor() as usize;
    level.min(MAX_LEVEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_pure_function_of_id_and_seed() {
        let ml = 1.0 / (16f64).ln();
        for id in 0..1000u32 {
            assert_eq!(level_for(id, 42, ml), level_for(id, 42, ml));
        }
        // Different seeds generally produce a different level somewhere.
        let a: Vec<_> = (0..200u32).map(|i| level_for(i, 1, ml)).collect();
        let b: Vec<_> = (0..200u32).map(|i| level_for(i, 2, ml)).collect();
        assert_ne!(a, b);
    }

    #[test]
    fn distribution_is_mostly_level_zero() {
        let ml = 1.0 / (16f64).ln();
        let n = 10_000u32;
        let zeros = (0..n).filter(|&i| level_for(i, 7, ml) == 0).count();
        // ~1 - 1/M = ~93.75% expected at level 0; allow a wide band.
        assert!(
            zeros as f64 / n as f64 > 0.88,
            "too few level-0 points: {zeros}"
        );
        assert!((0..n).all(|i| level_for(i, 7, ml) <= MAX_LEVEL));
    }
}
