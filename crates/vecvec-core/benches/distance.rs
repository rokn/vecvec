//! Micro-benchmarks for the distance kernels (M1). Reported per-call latency at
//! representative embedding dimensionalities; throughput is in vector elements/s.
//! The SIMD kernels (M14) will be compared against these scalar baselines here.

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use vecvec_core::Metric;
use vecvec_core::distance::{DistanceKernel, l2_normalize};

fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

const DIMS: &[usize] = &[8, 128, 384, 768, 1536];

fn bench_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_f32");
    for &dim in DIMS {
        let a = vec_of(dim, 1);
        let b = vec_of(dim, 2);
        group.throughput(Throughput::Elements(dim as u64));
        for metric in [Metric::Dot, Metric::Cosine, Metric::Euclidean] {
            let kernel = DistanceKernel::new(metric, dim);
            group.bench_with_input(BenchmarkId::new(metric.as_str(), dim), &dim, |bn, _| {
                bn.iter(|| black_box(kernel.score_f32(black_box(&a), black_box(&b))));
            });
        }
    }
    group.finish();
}

fn bench_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_u8");
    for &dim in DIMS {
        let a: Vec<u8> = (0..dim).map(|i| (i as u32 % 251) as u8).collect();
        let b: Vec<u8> = (0..dim).map(|i| ((i as u32 * 7 + 3) % 251) as u8).collect();
        group.throughput(Throughput::Elements(dim as u64));
        for metric in [Metric::Dot, Metric::Euclidean] {
            let kernel = DistanceKernel::new(metric, dim);
            group.bench_with_input(BenchmarkId::new(metric.as_str(), dim), &dim, |bn, _| {
                bn.iter(|| black_box(kernel.score_u8(black_box(&a), black_box(&b))));
            });
        }
    }
    group.finish();
}

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_normalize");
    for &dim in DIMS {
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bn, &dim| {
            bn.iter_batched(
                || vec_of(dim, 3),
                |mut v| black_box(l2_normalize(black_box(&mut v))),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_f32, bench_u8, bench_normalize);
criterion_main!(benches);
