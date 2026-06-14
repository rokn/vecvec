//! Distance metrics and the runtime-dispatched distance kernel.
//!
//! A [`DistanceKernel`] is the hot-path object every search holds: it bundles the
//! chosen `f32` and `u8` kernel function pointers for a metric so a comparison is a
//! single indirect call with no per-call feature branch. At M1 the kernels are the
//! portable scalar implementations in [`scalar`]; SIMD variants are selected by
//! [`dispatch`] in M14 without changing this interface.
//!
//! Score polarity differs by metric, so the kernel returns the *raw* metric value
//! and callers order results via [`Metric::higher_is_better`]:
//! - [`Metric::Cosine`] / [`Metric::Dot`] — dot product, **higher is better**
//!   (cosine assumes L2-normalized inputs; see [`l2_normalize`]).
//! - [`Metric::Euclidean`] — squared L2 distance, **lower is better**.

mod dispatch;
mod scalar;

#[cfg(feature = "oracle")]
mod simsimd_oracle;

use std::fmt;

/// Signature of an `f32` distance kernel. `unsafe` so SIMD variants (which require
/// `#[target_feature]`) share one type with the scalar fns (which coerce in safely).
pub(crate) type KernelF32 = unsafe fn(&[f32], &[f32]) -> f32;
/// Signature of a `u8` (quantized) distance kernel.
pub(crate) type KernelU8 = unsafe fn(&[u8], &[u8]) -> i32;

/// A vector-space distance/similarity metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Cosine similarity, computed as a dot product over L2-normalized vectors.
    Cosine,
    /// Raw inner (dot) product.
    Dot,
    /// (Squared) Euclidean distance.
    Euclidean,
}

impl Metric {
    /// Whether a larger raw score means a *better* (closer) match. True for cosine
    /// and dot; false for Euclidean (where smaller distance is better).
    #[inline]
    pub const fn higher_is_better(self) -> bool {
        matches!(self, Metric::Cosine | Metric::Dot)
    }

    /// The sentinel "worst possible" score for this metric, suitable for seeding a
    /// top-k heap: `-inf` when higher-is-better, `+inf` otherwise.
    #[inline]
    pub const fn worst_score(self) -> f32 {
        if self.higher_is_better() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    }

    /// Whether inputs must be L2-normalized at ingest for this metric to be correct.
    /// Only true for [`Metric::Cosine`].
    #[inline]
    pub const fn requires_normalization(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// The stable lowercase name used on the wire / in configs.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
            Metric::Euclidean => "euclidean",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parses a metric from its lowercase wire name.
impl std::str::FromStr for Metric {
    type Err = UnknownMetric;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosine" => Ok(Metric::Cosine),
            "dot" => Ok(Metric::Dot),
            "euclidean" => Ok(Metric::Euclidean),
            other => Err(UnknownMetric(other.to_owned())),
        }
    }
}

/// Error returned when parsing an unrecognized [`Metric`] name.
#[derive(Debug, thiserror::Error)]
#[error("unknown metric {0:?} (expected one of: cosine, dot, euclidean)")]
pub struct UnknownMetric(pub String);

/// The hot-path distance object: a metric plus its resolved kernel function
/// pointers and the (fixed) vector dimensionality they operate on.
#[derive(Clone, Copy)]
pub struct DistanceKernel {
    metric: Metric,
    dim: usize,
    f32_fn: KernelF32,
    u8_fn: KernelU8,
}

impl DistanceKernel {
    /// Builds a kernel for `metric` over `dim`-dimensional vectors, resolving the
    /// best available kernel implementation once.
    pub fn new(metric: Metric, dim: usize) -> Self {
        let (f32_fn, u8_fn) = dispatch::select(metric);
        Self {
            metric,
            dim,
            f32_fn,
            u8_fn,
        }
    }

    /// The metric this kernel computes.
    #[inline]
    pub const fn metric(&self) -> Metric {
        self.metric
    }

    /// The vector dimensionality this kernel expects.
    #[inline]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Computes the raw `f32` metric value between two vectors. Order results with
    /// [`Metric::higher_is_better`].
    #[inline]
    pub fn score_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), self.dim, "lhs has wrong dimensionality");
        debug_assert_eq!(b.len(), self.dim, "rhs has wrong dimensionality");
        // SAFETY: the selected kernel only reads the two slices, whose lengths are
        // asserted equal to `dim` here (and re-checked inside each kernel). At M1
        // every kernel is a safe scalar fn coerced to the `unsafe fn` pointer type.
        unsafe { (self.f32_fn)(a, b) }
    }

    /// Computes the raw `i32` metric value between two quantized (`u8`) vectors.
    #[inline]
    pub fn score_u8(&self, a: &[u8], b: &[u8]) -> i32 {
        debug_assert_eq!(a.len(), self.dim, "lhs has wrong dimensionality");
        debug_assert_eq!(b.len(), self.dim, "rhs has wrong dimensionality");
        // SAFETY: see `score_f32`.
        unsafe { (self.u8_fn)(a, b) }
    }
}

impl fmt::Debug for DistanceKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DistanceKernel")
            .field("metric", &self.metric)
            .field("dim", &self.dim)
            .finish()
    }
}

/// The L2 (Euclidean) norm of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    scalar::dot_f32(v, v).sqrt()
}

/// L2-normalizes a vector in place and returns its original norm. A zero vector is
/// left unchanged (returns `0.0`).
#[inline]
pub fn l2_normalize(v: &mut [f32]) -> f32 {
    let norm = l2_norm(v);
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIMS: &[usize] = &[1, 3, 7, 8, 15, 16, 128, 769];

    /// Deterministic pseudo-vector, no RNG dependency.
    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                // map to roughly [-1, 1)
                ((x % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn naive_sq_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[test]
    fn hand_computed_values() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let dot = DistanceKernel::new(Metric::Dot, 3);
        assert_eq!(dot.score_f32(&a, &b), 32.0); // 4+10+18
        let euc = DistanceKernel::new(Metric::Euclidean, 3);
        assert_eq!(euc.score_f32(&a, &b), 27.0); // 9+9+9
    }

    #[test]
    fn kernels_match_naive_across_dims() {
        for &dim in DIMS {
            let a = vec_of(dim, 1);
            let b = vec_of(dim, 2);
            let dot = DistanceKernel::new(Metric::Dot, dim);
            let euc = DistanceKernel::new(Metric::Euclidean, dim);
            let got_dot = dot.score_f32(&a, &b);
            let got_euc = euc.score_f32(&a, &b);
            assert!(
                (got_dot - naive_dot(&a, &b)).abs() < 1e-3,
                "dim {dim}: dot {got_dot} vs {}",
                naive_dot(&a, &b)
            );
            assert!(
                (got_euc - naive_sq_l2(&a, &b)).abs() < 1e-3,
                "dim {dim}: sq_l2 {got_euc} vs {}",
                naive_sq_l2(&a, &b)
            );
        }
    }

    #[test]
    fn cosine_is_normalized_dot() {
        // After L2-normalizing both vectors, the dot kernel yields cosine similarity.
        let mut a = vec_of(64, 7);
        let mut b = vec_of(64, 9);
        let raw_cos = naive_dot(&a, &b) / (l2_norm(&a) * l2_norm(&b));
        let na = l2_normalize(&mut a);
        let nb = l2_normalize(&mut b);
        assert!(na > 0.0 && nb > 0.0);
        assert!((l2_norm(&a) - 1.0).abs() < 1e-5);
        let k = DistanceKernel::new(Metric::Cosine, 64);
        assert!((k.score_f32(&a, &b) - raw_cos).abs() < 1e-5);
        assert!(k.score_f32(&a, &b) <= 1.0 + 1e-5 && k.score_f32(&a, &b) >= -1.0 - 1e-5);
    }

    #[test]
    fn u8_kernels() {
        let a = [1u8, 2, 3, 255];
        let b = [4u8, 5, 6, 1];
        let dot = DistanceKernel::new(Metric::Dot, 4);
        assert_eq!(dot.score_u8(&a, &b), 4 + 10 + 18 + 255);
        let euc = DistanceKernel::new(Metric::Euclidean, 4);
        assert_eq!(euc.score_u8(&a, &b), 9 + 9 + 9 + 254 * 254);
    }

    #[test]
    fn metric_properties() {
        assert!(Metric::Cosine.higher_is_better());
        assert!(Metric::Dot.higher_is_better());
        assert!(!Metric::Euclidean.higher_is_better());
        assert_eq!(Metric::Dot.worst_score(), f32::NEG_INFINITY);
        assert_eq!(Metric::Euclidean.worst_score(), f32::INFINITY);
        assert!(Metric::Cosine.requires_normalization());
        assert!(!Metric::Dot.requires_normalization());
        assert_eq!(Metric::Euclidean.to_string(), "euclidean");
        assert_eq!("cosine".parse::<Metric>().unwrap(), Metric::Cosine);
        assert!("bogus".parse::<Metric>().is_err());
    }

    #[test]
    fn zero_vector_normalizes_to_itself() {
        let mut z = vec![0.0f32; 8];
        assert_eq!(l2_normalize(&mut z), 0.0);
        assert!(z.iter().all(|&x| x == 0.0));
    }

    #[cfg(feature = "oracle")]
    #[test]
    fn matches_simsimd_oracle() {
        use super::simsimd_oracle;
        for &dim in DIMS {
            let a = vec_of(dim, 11);
            let b = vec_of(dim, 13);
            let dot = DistanceKernel::new(Metric::Dot, dim);
            let euc = DistanceKernel::new(Metric::Euclidean, dim);
            assert!((dot.score_f32(&a, &b) - simsimd_oracle::dot(&a, &b)).abs() < 1e-4);
            assert!((euc.score_f32(&a, &b) - simsimd_oracle::sqeuclidean(&a, &b)).abs() < 1e-4);
        }
    }
}
