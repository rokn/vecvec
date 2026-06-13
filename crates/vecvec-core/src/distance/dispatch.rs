//! Kernel selection.
//!
//! At M1 this always returns the portable scalar kernels. In M14 it grows runtime
//! CPU-feature detection (`is_x86_feature_detected!` / `is_aarch64_feature_detected!`)
//! and picks AVX-512 / AVX2+FMA / SSE / NEON variants, resolved **exactly once**
//! into the function pointers stored on a [`DistanceKernel`]. Keeping selection
//! behind this single function means the rest of the engine never branches on CPU
//! features on the hot path.
//!
//! [`DistanceKernel`]: super::DistanceKernel

use super::{KernelF32, KernelU8, Metric, scalar};

/// Selects the `(f32, u8)` kernel pair for a metric.
pub(crate) fn select(metric: Metric) -> (KernelF32, KernelU8) {
    // Cosine is dot over L2-normalized inputs, so it shares the dot kernel; the
    // normalization is performed once, at ingest, by the storage layer.
    match metric {
        Metric::Cosine | Metric::Dot => (scalar::dot_f32 as KernelF32, scalar::dot_u8 as KernelU8),
        Metric::Euclidean => (scalar::sq_l2_f32 as KernelF32, scalar::sq_l2_u8 as KernelU8),
    }
}
