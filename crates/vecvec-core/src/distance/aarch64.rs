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
