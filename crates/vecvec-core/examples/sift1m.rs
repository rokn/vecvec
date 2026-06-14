//! SIFT1M head-to-head benchmark — real data, **exact** ground truth.
//!
//! Loads the standard ANN_SIFT1M dataset (1M × 128 base, 10k queries, 100-NN exact
//! ground truth, L2/Euclidean) and reports recall@10, mean/p50/p99 single-thread
//! query latency, and QPS across an `ef` sweep — the same axes ann-benchmarks and
//! Qdrant publish, so the numbers are directly comparable (unlike the synthetic
//! `benches/hnsw.rs` figures).
//!
//! ```sh
//! # dataset: http://corpus-texmex.irisa.fr/  (ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)
//! SIFT_DIR=~/.cache/vecvec-sift/sift cargo run --release -p vecvec-core --example sift1m
//! ```
//! `EFC` overrides ef_construction (default 128); `MAXQ` caps the query count.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use vecvec_core::{HnswConfig, HnswIndex, Index, Metric, SearchParams, VectorStorage};

/// Reads a `.fvecs` file: a sequence of `[i32 dim][dim × f32]` records (little-endian).
/// Returns `(dim, flat_row_major)`.
fn read_fvecs(path: &Path) -> (usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut out = Vec::new();
    let mut dim = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let d = i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        dim = d;
        i += 4;
        let floats = &bytes[i..i + d * 4];
        out.extend(floats.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
        i += d * 4;
    }
    (dim, out)
}

/// Reads an `.ivecs` file: a sequence of `[i32 dim][dim × i32]` records. Returns rows.
fn read_ivecs(path: &Path) -> Vec<Vec<i32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut rows = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let d = i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        let row = bytes[i..i + d * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        rows.push(row);
        i += d * 4;
    }
    rows
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx]
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let dir: PathBuf = std::env::var("SIFT_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/vecvec-sift/sift")
    });

    eprintln!("loading SIFT1M from {}", dir.display());
    let (dim, base) = read_fvecs(&dir.join("sift_base.fvecs"));
    let (qdim, queries) = read_fvecs(&dir.join("sift_query.fvecs"));
    assert_eq!(dim, qdim, "base/query dim mismatch");
    let gt = read_ivecs(&dir.join("sift_groundtruth.ivecs"));
    let n = base.len() / dim;
    let nq = (queries.len() / dim).min(env_usize("MAXQ", usize::MAX));
    eprintln!("base = {n} × {dim}, queries = {nq}, gt rows = {}", gt.len());

    // SIFT uses exact L2 — Euclidean metric, no normalization.
    let t = Instant::now();
    let mut storage = VectorStorage::with_capacity(dim, Metric::Euclidean, n);
    for v in base.chunks_exact(dim) {
        storage.push(v);
    }
    eprintln!("ingested {n} vectors in {:.1}s", t.elapsed().as_secs_f64());

    let cfg = HnswConfig {
        ef_construction: env_usize("EFC", 128),
        // Lower the search-time floor so the per-query `ef` below fully drives the
        // beam width (`HnswIndex::search` raises ef to `max(params.ef, ef_search, k)`).
        // With the default ef_search=64 the cheap end of the Pareto curve is hidden.
        ef_search: 1,
        ..HnswConfig::default()
    };
    eprintln!(
        "building HNSW (m={}, m_max0={}, ef_c={}, int8_quantized={}) ...",
        cfg.m, cfg.m_max0, cfg.ef_construction, cfg.quantization
    );
    let t = Instant::now();
    let index = HnswIndex::build_parallel(Arc::new(storage), cfg);
    eprintln!("built in {:.1}s", t.elapsed().as_secs_f64());

    let k = 10usize;
    println!("\n  ef | recall@{k} | mean µs |  p50 µs |  p99 µs |  QPS/core");
    println!("-----+-----------+---------+---------+---------+-----------");
    for &ef in &[10usize, 16, 24, 32, 48, 64, 96, 128, 192, 256] {
        let params = SearchParams { ef, exact: false };
        let mut lat_us: Vec<f64> = Vec::with_capacity(nq);
        let mut hits = 0usize;
        for (qi, q) in queries.chunks_exact(dim).take(nq).enumerate() {
            let start = Instant::now();
            let res = index.search(q, k, params, None);
            lat_us.push(start.elapsed().as_nanos() as f64 / 1000.0);
            let truth: HashSet<u32> = gt[qi].iter().take(k).map(|&x| x as u32).collect();
            hits += res.iter().filter(|r| truth.contains(&r.id.get())).count();
            std::hint::black_box(&res);
        }
        let total_us: f64 = lat_us.iter().sum();
        lat_us.sort_by(|a, b| a.total_cmp(b));
        let recall = hits as f64 / (nq * k) as f64;
        let mean = total_us / nq as f64;
        println!(
            "{ef:4} |   {recall:.4}  | {mean:7.1} | {:7.1} | {:7.1} | {:9.0}",
            percentile(&lat_us, 50.0),
            percentile(&lat_us, 99.0),
            1_000_000.0 / mean,
        );
    }
}
