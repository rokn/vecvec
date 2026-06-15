//! Fast build-time optimization harness (subset of SIFT1M).
//!
//! Builds `MAXN` (default 200k) SIFT base vectors and reports build time, build
//! throughput, and recall@10 against a brute-force oracle computed over the *same*
//! subset (so recall stays valid at any `MAXN`, unlike the full-1M ground truth).
//! Much faster to iterate than `examples/sift1m.rs`.
//!
//! ```sh
//! MAXN=200000 QSAMPLE=200 cargo run --release -p vecvec-core --example sift_build
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use vecvec_core::{HnswConfig, HnswIndex, Index, Metric, PointId, SearchParams, VectorStorage};

fn read_fvecs(path: &Path) -> (usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut out = Vec::new();
    let mut dim = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let d = i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        dim = d;
        i += 4;
        out.extend(
            bytes[i..i + d * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
        );
        i += d * 4;
    }
    (dim, out)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Brute-force top-10 (squared L2) over `base[0..n]`, run in parallel over queries.
fn brute_truth(base: &[f32], dim: usize, n: usize, queries: &[f32], nq: usize) -> Vec<Vec<u32>> {
    (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..qi * dim + dim];
            let mut best: Vec<(f32, u32)> = Vec::with_capacity(n);
            for j in 0..n {
                let v = &base[j * dim..j * dim + dim];
                let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                best.push((d, j as u32));
            }
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
            best.truncate(10);
            best.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

fn recall_at(
    index: &HnswIndex,
    dim: usize,
    queries: &[f32],
    nq: usize,
    truth: &[Vec<u32>],
    ef: usize,
) -> f64 {
    let mut hits = 0usize;
    for qi in 0..nq {
        let q = &queries[qi * dim..qi * dim + dim];
        let got = index.search(q, 10, SearchParams { ef, exact: false }, None);
        let t: std::collections::HashSet<u32> = truth[qi].iter().copied().collect();
        hits += got.iter().filter(|r| t.contains(&r.id.get())).count();
    }
    hits as f64 / (nq * 10) as f64
}

fn main() {
    let _ = PointId::new(0); // keep the import used regardless of cfg
    // Set the rayon pool size BEFORE any parallel work (brute_truth below also uses
    // rayon, and the first par call locks in the global pool size).
    let threads = env_usize("THREADS", 0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }
    let dir: PathBuf = std::env::var("SIFT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/vecvec-sift/sift")
        });
    let (dim, base_all) = read_fvecs(&dir.join("sift_base.fvecs"));
    let (_, queries_all) = read_fvecs(&dir.join("sift_query.fvecs"));
    let n = env_usize("MAXN", 200_000).min(base_all.len() / dim);
    let nq = env_usize("QSAMPLE", 200).min(queries_all.len() / dim);

    let truth = brute_truth(&base_all, dim, n, &queries_all, nq);

    let efc = env_usize("EFC", 128);
    let quant = env_usize("QUANT", 1) != 0;
    let mut storage = VectorStorage::with_capacity(dim, Metric::Euclidean, n);
    for v in base_all[..n * dim].chunks_exact(dim) {
        storage.push(v);
    }
    let cfg = HnswConfig {
        ef_construction: efc,
        ef_search: 1,
        quantization: quant,
        ..HnswConfig::default()
    };

    let t = Instant::now();
    let index = HnswIndex::build_parallel(Arc::new(storage), cfg);
    let secs = t.elapsed().as_secs_f64();

    let r64 = recall_at(&index, dim, &queries_all, nq, &truth, 64);
    let r128 = recall_at(&index, dim, &queries_all, nq, &truth, 128);
    let thr = if threads > 0 {
        threads
    } else {
        rayon::current_num_threads()
    };
    println!(
        "n={n} efc={efc} quant={} thr={thr} | build {secs:6.2}s | {:8.0} vec/s | recall@10 ef64={r64:.4} ef128={r128:.4}",
        quant as u8,
        n as f64 / secs
    );
}
