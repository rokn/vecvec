//! AVX2+FMA (x86_64) f32 distance kernels.
//!
//! Selected at runtime when AVX2+FMA are detected. **Four independent accumulators**
//! (32 lanes/iter) hide FMA latency, with a horizontal reduce and a scalar tail.
//! Must match [`scalar`](super::scalar) within fp tolerance (the distance tests +
//! simsimd oracle enforce this).
#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};

#[target_feature(enable = "avx2")]
fn reduce(acc: [__m256; 4]) -> f32 {
    let s01 = _mm256_add_ps(acc[0], acc[1]);
    let s23 = _mm256_add_ps(acc[2], acc[3]);
    let s = _mm256_add_ps(s01, s23);
    let mut tmp = [0.0f32; 8];
    // SAFETY: `tmp` holds exactly 8 f32, matching the 256-bit store.
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), s) };
    tmp.iter().sum()
}

/// Dot product (AVX2+FMA). `unsafe` only because of `#[target_feature]`; dispatch
/// guarantees `avx2` + `fma`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut ca = a.chunks_exact(32);
    let mut cb = b.chunks_exact(32);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..4 {
            // SAFETY: the 32-element chunk fully covers the four 8-wide loads.
            let (va, vb) = unsafe {
                (
                    _mm256_loadu_ps(x[j * 8..].as_ptr()),
                    _mm256_loadu_ps(y[j * 8..].as_ptr()),
                )
            };
            acc[j] = _mm256_fmadd_ps(va, vb, acc[j]);
        }
    }
    let mut sum = reduce(acc);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Squared Euclidean distance (AVX2+FMA).
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn sq_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut ca = a.chunks_exact(32);
    let mut cb = b.chunks_exact(32);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for j in 0..4 {
            // SAFETY: the 32-element chunk fully covers the four 8-wide loads.
            let (va, vb) = unsafe {
                (
                    _mm256_loadu_ps(x[j * 8..].as_ptr()),
                    _mm256_loadu_ps(y[j * 8..].as_ptr()),
                )
            };
            let d = _mm256_sub_ps(va, vb);
            acc[j] = _mm256_fmadd_ps(d, d, acc[j]);
        }
    }
    let mut sum = reduce(acc);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}
