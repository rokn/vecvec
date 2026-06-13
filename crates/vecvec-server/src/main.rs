//! `vecvec-server` — the database server binary.
//!
//! Scaffold only at M0: the tokio runtime, tonic gRPC + axum REST surfaces, the
//! collection registry, and background tasks are introduced from M3 onward (see
//! `BuildPlan.md`).

fn main() {
    println!(
        "vecvec-server {} — scaffold (networking lands in M3)",
        vecvec_core::VERSION
    );
}
