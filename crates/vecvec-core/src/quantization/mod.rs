//! Scalar (int8) quantization with f32 rescoring.
//!
//! Each sealed segment stores, alongside its f32 vectors, a compact `i8` copy at
//! ~4× less memory. Search ranks candidates with cheap integer distances over the
//! `i8` block, then **rescores** a small over-fetched set with the exact f32 vectors
//! to recover the true order. We use **symmetric** quantization (a single per-segment
//! scale, zero offset): `q = clamp(round(v / scale), -127, 127)`. With no offset,
//! both the int dot product and the int squared-L2 are *monotonic* with their f32
//! counterparts, so ranking is preserved and the rescore fixes the final top-k.

use rayon::prelude::*;

use crate::distance::{DistanceKernel, Metric};
use crate::id::PointId;
use crate::index::ScoredPoint;
use crate::ordered::OrderedF32;
use crate::vector::VectorStorage;

/// A fitted symmetric int8 quantizer.
#[derive(Debug, Clone, Copy)]
pub struct ScalarQuantizer {
    scale: f32,
}

impl ScalarQuantizer {
    /// Fits a quantizer to a vector store (scale = max |value| / 127).
    pub fn fit(storage: &VectorStorage) -> Self {
        let max_abs = storage
            .as_flat()
            .par_iter()
            .fold(|| 0.0f32, |m, &x| m.max(x.abs()))
            .reduce(|| 0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        Self { scale }
    }

    /// The quantization scale.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Encodes a vector into `out` (cleared first).
    pub fn encode_into(&self, vector: &[f32], out: &mut Vec<i8>) {
        out.clear();
        out.reserve(vector.len());
        let inv = 1.0 / self.scale;
        for &x in vector {
            out.push((x * inv).round().clamp(-127.0, 127.0) as i8);
        }
    }

    /// Encodes a vector into a fresh buffer.
    pub fn encode(&self, vector: &[f32]) -> Vec<i8> {
        let mut out = Vec::new();
        self.encode_into(vector, &mut out);
        out
    }

    /// Encodes `vector` into the `out` slice (same length).
    #[inline]
    pub fn encode_slice(&self, vector: &[f32], out: &mut [i8]) {
        let inv = 1.0 / self.scale;
        for (o, &x) in out.iter_mut().zip(vector) {
            *o = (x * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// A contiguous block of int8-quantized vectors.
pub struct QuantizedVectorBlock {
    data: Vec<i8>,
    dim: usize,
    metric: Metric,
    quantizer: ScalarQuantizer,
}

impl QuantizedVectorBlock {
    /// Builds a quantized block from an f32 store.
    pub fn build(storage: &VectorStorage) -> Self {
        let quantizer = ScalarQuantizer::fit(storage);
        let dim = storage.dim();
        let mut data = vec![0i8; storage.len() * dim];
        // Encode rows in parallel (this is on the index-build critical path).
        data.par_chunks_mut(dim)
            .zip(storage.as_flat().par_chunks(dim))
            .for_each(|(out, v)| quantizer.encode_slice(v, out));
        Self {
            data,
            dim,
            metric: storage.metric(),
            quantizer,
        }
    }

    /// The number of quantized vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    /// Whether the block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The number of bytes used (for the ~4× memory comparison).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// The fitted quantizer (to encode queries).
    #[inline]
    pub fn quantizer(&self) -> &ScalarQuantizer {
        &self.quantizer
    }

    #[inline]
    fn get(&self, id: u32) -> &[i8] {
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Badness (smaller = closer) between a quantized query and stored point `id`,
    /// using the quantized metric. Monotonic with the f32 metric, so suitable for
    /// ranking candidates prior to rescoring.
    #[inline]
    pub fn badness(&self, quantized_query: &[i8], id: u32) -> f32 {
        let stored = self.get(id);
        match self.metric {
            Metric::Cosine | Metric::Dot => -(dot_i8(quantized_query, stored) as f32),
            Metric::Euclidean => sq_l2_i8(quantized_query, stored) as f32,
        }
    }

    /// Badness between two *stored* points by id, in the quantized metric. Reads two
    /// int8 codes (4× smaller than the f32 rows), used to keep graph construction's
    /// memory-bound distance loop in cache.
    ///
    /// # Safety
    /// `a` and `b` must be `< self.len()`.
    #[inline]
    pub(crate) unsafe fn badness_between(&self, a: u32, b: u32) -> f32 {
        let da = a as usize * self.dim;
        let db = b as usize * self.dim;
        // SAFETY: caller guarantees both ids are in range.
        let (va, vb) = unsafe {
            (
                self.data.get_unchecked(da..da + self.dim),
                self.data.get_unchecked(db..db + self.dim),
            )
        };
        match self.metric {
            Metric::Cosine | Metric::Dot => -(dot_i8(va, vb) as f32),
            Metric::Euclidean => sq_l2_i8(va, vb) as f32,
        }
    }
}

// The simple scalar reductions auto-vectorize well (LLVM beats a hand-rolled NEON
// version here, which was measurably slower — see the Opt8 experiment).
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

#[inline]
fn sq_l2_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            d * d
        })
        .sum()
}

/// Exact-rescore quantized search over a flat block: rank all points by quantized
/// distance, take the best `k * oversample`, then rescore those with the f32 kernel
/// and return the true top-k. (The standalone reference used to validate the
/// quantized path; the HNSW index applies the same quantize-then-rescore idea over
/// its graph traversal.)
pub fn quantized_rescore_search(
    block: &QuantizedVectorBlock,
    storage: &VectorStorage,
    kernel: &DistanceKernel,
    query: &[f32],
    k: usize,
    oversample: usize,
) -> Vec<ScoredPoint> {
    if k == 0 || block.is_empty() {
        return Vec::new();
    }
    let quantized_query = block.quantizer().encode(query);
    let fetch = (k * oversample.max(1)).max(k);

    let mut ranked: Vec<(OrderedF32, u32)> = (0..block.len() as u32)
        .map(|id| (OrderedF32::new(block.badness(&quantized_query, id)), id))
        .collect();
    let cut = fetch.min(ranked.len());
    ranked.select_nth_unstable(cut - 1);
    ranked.truncate(cut);

    let higher = kernel.metric().higher_is_better();
    let mut rescored: Vec<(OrderedF32, ScoredPoint)> = ranked
        .into_iter()
        .map(|(_, id)| {
            let score = kernel.score_f32(query, storage.get(PointId::new(id)));
            let badness = if higher { -score } else { score };
            (
                OrderedF32::new(badness),
                ScoredPoint {
                    id: PointId::new(id),
                    score,
                },
            )
        })
        .collect();
    rescored.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.id.cmp(&b.1.id)));
    rescored.truncate(k);
    rescored.into_iter().map(|(_, sp)| sp).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute_force_topk;

    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503))
                    .wrapping_add(i as u32 * seed);
                ((x % 10_000) as f32 / 5_000.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn rescore_recall_and_memory() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            let dim = 48;
            let n = 1500;
            let mut storage = VectorStorage::with_capacity(dim, metric, n);
            for i in 0..n {
                storage.push(&vec_of(dim, i as u32 + 1));
            }
            let kernel = DistanceKernel::new(metric, dim);
            let block = QuantizedVectorBlock::build(&storage);

            // ~4x memory reduction vs f32.
            assert_eq!(block.byte_len(), n * dim);
            assert_eq!(block.byte_len() * 4, storage.as_flat().len() * 4);

            let mut hits = 0;
            let trials = 20;
            for q in 0..trials {
                let query = vec_of(dim, 900_000 + q);
                let got: std::collections::HashSet<u32> =
                    quantized_rescore_search(&block, &storage, &kernel, &query, 10, 4)
                        .into_iter()
                        .map(|sp| sp.id.get())
                        .collect();
                let truth = brute_force_topk(&storage, &kernel, &query, 10, None, None);
                hits += truth.iter().filter(|sp| got.contains(&sp.id.get())).count();
            }
            let recall = hits as f32 / (10 * trials) as f32;
            assert!(
                recall >= 0.90,
                "metric={metric}: quantized recall@10 {recall} < 0.90"
            );
        }
    }

    #[test]
    fn rescore_beats_no_rescore() {
        // With oversample=1 (no extra fetch) recall is lower than with oversample=8.
        let dim = 48;
        let n = 1500;
        let metric = Metric::Cosine;
        let mut storage = VectorStorage::with_capacity(dim, metric, n);
        for i in 0..n {
            storage.push(&vec_of(dim, i as u32 + 1));
        }
        let kernel = DistanceKernel::new(metric, dim);
        let block = QuantizedVectorBlock::build(&storage);

        let recall_for = |oversample: usize| {
            let mut hits = 0;
            for q in 0..20 {
                let query = vec_of(dim, 700_000 + q);
                let got: std::collections::HashSet<u32> =
                    quantized_rescore_search(&block, &storage, &kernel, &query, 10, oversample)
                        .into_iter()
                        .map(|sp| sp.id.get())
                        .collect();
                let truth = brute_force_topk(&storage, &kernel, &query, 10, None, None);
                hits += truth.iter().filter(|sp| got.contains(&sp.id.get())).count();
            }
            hits as f32 / 200.0
        };
        assert!(recall_for(8) >= recall_for(1));
    }

    #[test]
    fn encode_clamps_rounds_and_is_symmetric() {
        // Fit on a store whose max abs is a known value (12.7) so scale is checkable.
        let dim = 4;
        let mut storage = VectorStorage::new(dim, Metric::Dot);
        storage.push(&[12.7, -5.0, 0.0, 1.0]); // max abs = 12.7
        storage.push(&[3.0, -2.0, 6.35, -1.0]);
        let q = ScalarQuantizer::fit(&storage);
        assert!(
            (q.scale() - 12.7 / 127.0).abs() < 1e-6,
            "scale must be max|value|/127, got {}",
            q.scale()
        );

        // The max-magnitude value encodes to exactly +/-127 (not an i8-cast overflow).
        assert_eq!(q.encode(&[12.7, 0.0, 0.0, 0.0])[0], 127);
        assert_eq!(q.encode(&[-12.7, 0.0, 0.0, 0.0])[0], -127);

        // A value far beyond the fitted max saturates via clamp (no `as i8` wrap:
        // 1e9 would wrap to a small/negative number without the clamp).
        assert_eq!(q.encode(&[1e9, 0.0, 0.0, 0.0])[0], 127);
        assert_eq!(q.encode(&[-1e9, 0.0, 0.0, 0.0])[0], -127);

        // Symmetric, zero-offset scheme: encode(v) == -encode(-v) componentwise.
        for seed in [1u32, 2, 3] {
            let v = vec_of(dim, seed);
            let pos = q.encode(&v);
            let neg = q.encode(&v.iter().map(|x| -x).collect::<Vec<_>>());
            for (i, (p, n)) in pos.iter().zip(&neg).enumerate() {
                assert_eq!(*p, -*n, "encode not symmetric at seed {seed} component {i}");
            }
        }
    }

    #[test]
    fn fit_all_zero_storage_yields_unit_scale() {
        // The single guard against 1.0/0.0 = +inf inv (which would turn every code
        // into NaN-as-i8 / garbage): an all-zero store forces scale to 1.0.
        let dim = 4;
        let mut storage = VectorStorage::new(dim, Metric::Dot);
        for _ in 0..8 {
            storage.push(&[0.0; 4]);
        }
        let q = ScalarQuantizer::fit(&storage);
        assert_eq!(q.scale(), 1.0, "all-zero store must force scale to 1.0");
        assert_eq!(q.encode(&[0.0; 4]), vec![0i8; 4]);

        // An all-zero query against a quantizer fitted on normal (nonzero) data also
        // encodes to all zeros.
        let mut nonzero = VectorStorage::new(dim, Metric::Dot);
        nonzero.push(&[1.0, -2.0, 3.0, 0.5]);
        let q2 = ScalarQuantizer::fit(&nonzero);
        assert!(q2.scale() > 0.0);
        assert_eq!(q2.encode(&[0.0; 4]), vec![0i8; 4]);
    }
}
