//! Portable scalar distance kernels.
//!
//! These are **always compiled** and define the ground-truth semantics of every
//! metric. The SIMD-dispatched fast paths added in M14 must match them bit-for-bit
//! modulo floating-point reassociation, and the test suite uses them (plus the
//! optional `simsimd` oracle) as a reference.
//!
//! Conventions:
//! - Dot product (used for [`Metric::Dot`] and, over L2-normalized inputs, for
//!   [`Metric::Cosine`]): **higher is more similar**.
//! - Squared Euclidean (used for [`Metric::Euclidean`]): **lower is closer**. We
//!   keep it squared because the square root is monotonic and only wastes cycles
//!   for ranking.
//!
//! [`Metric::Dot`]: super::Metric::Dot
//! [`Metric::Cosine`]: super::Metric::Cosine
//! [`Metric::Euclidean`]: super::Metric::Euclidean

/// Dot product of two equal-length `f32` vectors.
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Four independent accumulators expose instruction-level parallelism and let
    // the autovectorizer fold this into vector FMAs.
    let mut acc = [0.0f32; 4];
    let mut ca = a.chunks_exact(4);
    let mut cb = b.chunks_exact(4);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Squared Euclidean distance of two equal-length `f32` vectors.
pub(crate) fn sq_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 4];
    let mut ca = a.chunks_exact(4);
    let mut cb = b.chunks_exact(4);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        let d0 = x[0] - y[0];
        let d1 = x[1] - y[1];
        let d2 = x[2] - y[2];
        let d3 = x[3] - y[3];
        acc[0] += d0 * d0;
        acc[1] += d1 * d1;
        acc[2] += d2 * d2;
        acc[3] += d3 * d3;
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Dot product of two equal-length `u8` (scalar-quantized) vectors, accumulated in
/// `i32`. The quantization layer (M8) defines how this raw integer score maps back
/// to an approximate f32 metric and drives the rescore step.
pub(crate) fn dot_u8(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// Squared Euclidean distance of two equal-length `u8` vectors, accumulated in `i32`.
pub(crate) fn sq_l2_u8(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            d * d
        })
        .sum()
}
