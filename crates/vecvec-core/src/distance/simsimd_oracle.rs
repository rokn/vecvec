//! Independent reference kernels from the `simsimd` crate, used **only** to
//! cross-check the hand-written/scalar kernels in tests. Compiled only under the
//! `oracle` feature (`cargo test --features oracle`); never part of a normal build.

use simsimd::SpatialSimilarity;

/// Reference dot product.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).expect("equal-length vectors") as f32
}

/// Reference squared Euclidean distance.
pub(crate) fn sqeuclidean(a: &[f32], b: &[f32]) -> f32 {
    f32::sqeuclidean(a, b).expect("equal-length vectors") as f32
}
