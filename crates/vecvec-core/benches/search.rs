//! Micro-benchmarks for exact `FlatIndex` top-k search (M2). This is the O(n)
//! brute-force baseline every approximate index (HNSW, M4) must beat; tracking it
//! also validates the per-segment fan-out cost model. Throughput is in vectors
//! scanned per second.

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vecvec_core::{FlatIndex, Index, Metric, SearchParams, VectorStorage};

fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn storage(dim: usize, n: usize, metric: Metric) -> Arc<VectorStorage> {
    let mut s = VectorStorage::with_capacity(dim, metric, n);
    for i in 0..n {
        s.push(&vec_of(dim, i as u32 + 1));
    }
    Arc::new(s)
}

fn bench_flat_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_search_top10");
    let cases = [
        (128usize, 1_000usize),
        (128, 10_000),
        (128, 100_000),
        (768, 10_000),
    ];
    for &(dim, n) in &cases {
        let flat = FlatIndex::new(storage(dim, n, Metric::Cosine));
        let query = vec_of(dim, 9_999);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("dim{dim}_n{n}")),
            &(),
            |bn, ()| {
                bn.iter(|| {
                    black_box(flat.search(black_box(&query), 10, SearchParams::default(), None))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_flat_search);
criterion_main!(benches);
