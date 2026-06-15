//! NEON (aarch64) f32 distance kernels.
//!
//! NEON is baseline on aarch64, so these are selected on every Apple-silicon /
//! ARMv8 machine. **Four independent accumulators** (16 lanes/iter) hide FMA latency,
//! with a horizontal reduce and a scalar tail. They must match
//! [`scalar`](super::scalar) within fp tolerance (the distance tests + simsimd oracle
//! enforce this).
#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32,
};

#[target_feature(enable = "neon")]
fn reduce(acc: [float32x4_t; 4]) -> f32 {
    let s01 = vaddq_f32(acc[0], acc[1]);
    let s23 = vaddq_f32(acc[2], acc[3]);
    vaddvq_f32(vaddq_f32(s01, s23))
}

/// Dot product (NEON). `unsafe` only because of `#[target_feature]`; dispatch
/// guarantees the `neon` feature (baseline on aarch64).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..4 {
            // SAFETY: the 16-element chunk fully covers the four 4-wide loads.
            let (va, vb) = unsafe {
                (
                    vld1q_f32(x[j * 4..].as_ptr()),
                    vld1q_f32(y[j * 4..].as_ptr()),
                )
            };
            acc[j] = vfmaq_f32(acc[j], va, vb);
        }
    }
    let mut sum = reduce(acc);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Squared Euclidean distance (NEON).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sq_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..4 {
            // SAFETY: the 16-element chunk fully covers the four 4-wide loads.
            let (va, vb) = unsafe {
                (
                    vld1q_f32(x[j * 4..].as_ptr()),
                    vld1q_f32(y[j * 4..].as_ptr()),
                )
            };
            let d = vsubq_f32(va, vb);
            acc[j] = vfmaq_f32(acc[j], d, d);
        }
    }
    let mut sum = reduce(acc);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    /// The NEON kernels run on every Apple-silicon / ARMv8 machine but are never
    /// compiled by the x86-only CI runners, so validate them in-file here. Covers the
    /// 16-wide main loop plus every tail remainder 0..16 (and a few large dims), each
    /// asserted equal to the scalar reference within fp tolerance.
    #[test]
    fn neon_matches_scalar_across_dims() {
        // 1..=15 are pure-tail (no full 16-chunk); 16..=33 give a full chunk with
        // every tail remainder 0..15; the rest stress multiple chunks + long tails.
        let dims = (1usize..=33).chain([48, 64, 96, 127, 128, 256, 769]);
        for dim in dims {
            let a = vec_of(dim, 11);
            let b = vec_of(dim, 13);
            // SAFETY: neon is baseline on aarch64; this file only compiles there.
            let (n_dot, n_l2) = unsafe { (dot_f32(&a, &b), sq_l2_f32(&a, &b)) };
            let s_dot = super::super::scalar::dot_f32(&a, &b);
            let s_l2 = super::super::scalar::sq_l2_f32(&a, &b);
            assert!(
                (n_dot - s_dot).abs() < 1e-3,
                "neon dot dim {dim}: {n_dot} vs {s_dot}"
            );
            assert!(
                (n_l2 - s_l2).abs() < 1e-3,
                "neon sq_l2 dim {dim}: {n_l2} vs {s_l2}"
            );
        }
    }
}
