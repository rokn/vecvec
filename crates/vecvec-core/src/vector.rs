//! Contiguous in-memory `f32` vector storage.
//!
//! [`VectorStorage`] is a flat, row-major `dim`-strided buffer of `f32` vectors —
//! the cache-friendly layout the distance kernels want. It owns the cosine
//! invariant: when its [`Metric`] is [`Metric::Cosine`], vectors are L2-normalized
//! **once, at ingest**, so search reduces to a plain dot product.
//!
//! At M2 this backs the [`FlatIndex`](crate::index::FlatIndex) and the appendable
//! segment. M5's sealed-segment vector block reuses the same layout (and adds the
//! `u8` quantized companion).

use crate::distance::{self, Metric};
use crate::id::PointId;

/// A growable, contiguous store of equal-length `f32` vectors.
#[derive(Clone)]
pub struct VectorStorage {
    data: Vec<f32>,
    dim: usize,
    metric: Metric,
    count: usize,
}

impl VectorStorage {
    /// Creates an empty store for `dim`-dimensional vectors under `metric`.
    ///
    /// # Panics
    /// Panics if `dim == 0`.
    pub fn new(dim: usize, metric: Metric) -> Self {
        assert!(dim > 0, "vector dimensionality must be non-zero");
        Self {
            data: Vec::new(),
            dim,
            metric,
            count: 0,
        }
    }

    /// Like [`VectorStorage::new`] but preallocates room for `capacity` vectors.
    pub fn with_capacity(dim: usize, metric: Metric, capacity: usize) -> Self {
        assert!(dim > 0, "vector dimensionality must be non-zero");
        Self {
            data: Vec::with_capacity(dim * capacity),
            dim,
            metric,
            count: 0,
        }
    }

    /// The vector dimensionality.
    #[inline]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// The metric this store normalizes for.
    #[inline]
    pub const fn metric(&self) -> Metric {
        self.metric
    }

    /// The number of stored vectors.
    #[inline]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether the store is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Appends a vector, returning its assigned segment-local [`PointId`]. For a
    /// cosine store the stored copy is L2-normalized; the input slice is untouched.
    ///
    /// # Panics
    /// Panics if `vector.len() != self.dim()`.
    pub fn push(&mut self, vector: &[f32]) -> PointId {
        assert_eq!(
            vector.len(),
            self.dim,
            "vector has wrong dimensionality (expected {}, got {})",
            self.dim,
            vector.len()
        );
        let id = PointId::new(self.count as u32);
        let start = self.data.len();
        self.data.extend_from_slice(vector);
        if self.metric.requires_normalization() {
            distance::l2_normalize(&mut self.data[start..start + self.dim]);
        }
        self.count += 1;
        id
    }

    /// Returns the stored vector for `id`.
    ///
    /// # Panics
    /// Panics if `id` is out of range.
    #[inline]
    pub fn get(&self, id: PointId) -> &[f32] {
        let start = id.get() as usize * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Iterates over `(id, vector)` pairs in id order.
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &[f32])> {
        self.data
            .chunks_exact(self.dim)
            .enumerate()
            .map(|(i, row)| (PointId::new(i as u32), row))
    }

    /// The raw backing buffer (row-major, `len() * dim()` long).
    #[inline]
    pub fn as_flat(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_get_roundtrip() {
        let mut s = VectorStorage::new(3, Metric::Dot);
        let a = s.push(&[1.0, 2.0, 3.0]);
        let b = s.push(&[4.0, 5.0, 6.0]);
        assert_eq!(a, PointId::new(0));
        assert_eq!(b, PointId::new(1));
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(a), &[1.0, 2.0, 3.0]);
        assert_eq!(s.get(b), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn cosine_normalizes_on_ingest() {
        let mut s = VectorStorage::new(2, Metric::Cosine);
        let id = s.push(&[3.0, 4.0]); // norm 5
        let stored = s.get(id);
        assert!((stored[0] - 0.6).abs() < 1e-6);
        assert!((stored[1] - 0.8).abs() < 1e-6);
        assert!((distance::l2_norm(stored) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_does_not_normalize() {
        let mut s = VectorStorage::new(2, Metric::Dot);
        let id = s.push(&[3.0, 4.0]);
        assert_eq!(s.get(id), &[3.0, 4.0]);
    }

    #[test]
    fn iter_yields_rows_in_order() {
        let mut s = VectorStorage::new(2, Metric::Euclidean);
        s.push(&[1.0, 1.0]);
        s.push(&[2.0, 2.0]);
        let rows: Vec<_> = s.iter().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, PointId::new(0));
        assert_eq!(rows[1].1, &[2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "wrong dimensionality")]
    fn push_wrong_dim_panics() {
        let mut s = VectorStorage::new(3, Metric::Dot);
        s.push(&[1.0, 2.0]);
    }
}
