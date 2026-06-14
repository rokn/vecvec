//! HNSW search benchmark (M4). Builds a graph once, then measures top-10 query
//! latency at several `ef` widths and reports recall@10 against a brute-force oracle.
//!
//! Two scales are benched:
//!   - `128d_10k`  — quick signal, comparable to the `flat_search_top10` baseline.
//!   - `128d_100k` — a Qdrant-comparable single-node scale (SIFT1M is 1M; 100k is a
//!     tractable subset). Single-thread query latency here ⇒ per-core QPS ≈ 1/latency;
//!     a server fans queries across cores, so cluster QPS ≈ cores × that.
//!
//! NOTE on recall: `vec_of` is a deterministic *synthetic* generator, not a real
//! embedding distribution, so the recall number is a sanity check on the index, not a
//! dataset-faithful figure. A rigorous Qdrant-style comparison needs SIFT1M/GloVe.

use std::collections::HashSet;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use vecvec_core::{HnswConfig, HnswIndex, Index, Metric, PointId, SearchParams, VectorStorage};

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

/// Brute-force top-10 ids for one query (recall oracle). Cosine ranking ≡ ranking by
/// `dot(query, stored)` since stored vectors are L2-normalized and the query is a
/// fixed positive scale across all candidates.
fn brute_force_top10(storage: &VectorStorage, query: &[f32]) -> Vec<PointId> {
    let mut scored: Vec<(f32, u32)> = (0..storage.len() as u32)
        .map(|i| {
            let v = storage.get(PointId::new(i));
            let dot: f32 = query.iter().zip(v).map(|(x, y)| x * y).sum();
            (dot, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(10).map(|(_, i)| PointId::new(i)).collect()
}

/// Mean recall@10 of the HNSW index against brute force over `n_queries` queries.
fn measure_recall(index: &HnswIndex, storage: &VectorStorage, ef: usize, n_queries: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..n_queries {
        let query = vec_of(storage.dim(), 7_000_000 + q as u32);
        let truth = brute_force_top10(storage, &query);
        let got = index.search(&query, 10, SearchParams { ef, exact: false }, None);
        let got_ids: HashSet<PointId> = got.into_iter().map(|r| r.id).collect();
        hits += truth.iter().filter(|t| got_ids.contains(t)).count();
        total += truth.len();
    }
    hits as f64 / total as f64
}

fn bench_scale(c: &mut Criterion, dim: usize, n: usize) {
    let mut storage = VectorStorage::with_capacity(dim, Metric::Cosine, n);
    for i in 0..n {
        storage.push(&vec_of(dim, i as u32 + 1));
    }
    let storage = Arc::new(storage);
    let index = HnswIndex::build(storage.clone(), HnswConfig::default());
    let query = vec_of(dim, 999_999);

    let label = format!("{dim}d_{}k", n / 1000);
    // Print recall@10 once per ef so the latency numbers are interpretable.
    for &ef in &[64usize, 128, 256] {
        let recall = measure_recall(&index, &storage, ef, 100);
        println!("recall@10 [{label} ef={ef}] = {recall:.4}");
    }

    let mut group = c.benchmark_group(format!("hnsw_search_top10/{label}"));
    for &ef in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, &ef| {
            let params = SearchParams { ef, exact: false };
            b.iter(|| black_box(index.search(black_box(&query), 10, params, None)));
        });
    }
    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    bench_scale(c, 128, 10_000);
    bench_scale(c, 128, 100_000);
}

criterion_group!(benches, bench_hnsw_search);
criterion_main!(benches);
