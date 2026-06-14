//! HNSW search benchmark (M4). Builds a 10k×128 graph once, then measures top-10
//! query latency at several `ef` widths. Compare against the `flat_search_top10`
//! baseline (≈211 µs for 128d×10k): HNSW should be ~10–50× faster while keeping
//! recall@10 ≥ 0.95 (validated in the unit tests).

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use vecvec_core::{HnswConfig, HnswIndex, Index, Metric, SearchParams, VectorStorage};

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

fn bench_hnsw_search(c: &mut Criterion) {
    let dim = 128usize;
    let n = 10_000usize;
    let mut storage = VectorStorage::with_capacity(dim, Metric::Cosine, n);
    for i in 0..n {
        storage.push(&vec_of(dim, i as u32 + 1));
    }
    let index = HnswIndex::build(Arc::new(storage), HnswConfig::default());
    let query = vec_of(dim, 999_999);

    let mut group = c.benchmark_group("hnsw_search_top10/128d_10k");
    for &ef in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, &ef| {
            let params = SearchParams { ef, exact: false };
            b.iter(|| black_box(index.search(black_box(&query), 10, params, None)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hnsw_search);
criterion_main!(benches);
