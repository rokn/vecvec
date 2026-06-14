//! Kernel selection.
//!
//! Resolves the best available distance kernel **once** (at [`DistanceKernel`]
//! creation) into a stored function pointer, so the hot path never branches on CPU
//! features. SIMD f32 kernels (NEON on aarch64, AVX2+FMA on x86_64) are selected via
//! runtime feature detection, falling back to the always-compiled scalar kernels.
//! The `u8` (quantized) kernels are scalar for now.
//!
//! [`DistanceKernel`]: super::DistanceKernel

use super::{KernelF32, KernelU8, Metric, scalar};

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(target_arch = "x86_64")]
use super::x86;

/// Selects the `(f32, u8)` kernel pair for a metric.
pub(crate) fn select(metric: Metric) -> (KernelF32, KernelU8) {
    match metric {
        // Cosine is dot over L2-normalized inputs, so it shares the dot kernel.
        Metric::Cosine | Metric::Dot => (best_dot_f32(), scalar::dot_u8 as KernelU8),
        Metric::Euclidean => (best_sq_l2_f32(), scalar::sq_l2_u8 as KernelU8),
    }
}

fn best_dot_f32() -> KernelF32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return x86::dot_f32 as KernelF32;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return aarch64::dot_f32 as KernelF32;
        }
    }
    scalar::dot_f32 as KernelF32
}

fn best_sq_l2_f32() -> KernelF32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return x86::sq_l2_f32 as KernelF32;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return aarch64::sq_l2_f32 as KernelF32;
        }
    }
    scalar::sq_l2_f32 as KernelF32
}
