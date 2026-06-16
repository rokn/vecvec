# vecvec

A super-fast, in-memory-primary vector database in Rust with **automatic, git-like
versioning** of collections (immutable snapshots, time-travel query, branch, diff,
restore) on top of an HNSW index. See [BuildPlan.md](./BuildPlan.md) for the full
architecture and milestone roadmap.

```sh
cargo run -p vecvec-server          # gRPC :6334 + REST :6333
grpcurl -plaintext 127.0.0.1:6334 list
curl localhost:6333/healthz
cargo run -p vecvec-cli -- --help
```

## Features

- **HNSW** index from scratch (Alg-4 heuristic, deterministic builds, lock-free
  sealed search) with **int8 scalar quantization + f32 rescore** (~4× memory, faster
  search) and hand-written **NEON/AVX2 SIMD** distance kernels.
- **Durable & crash-safe**: WAL-first writes (fsync before ack), generation-switched
  checkpoints, recovery; segment store with mmap loading + CRC framing.
- **Git-like versioning** (the differentiator): immutable manifest commits over
  shared segments → snapshot-isolated **time-travel** query, branch, diff, restore;
  rule-based auto-commit. Survives restart.
- **Payload + filtered search** (Qdrant-style `must`/`should`/`must_not`),
  **recommend-by-example**, **compaction + GC**, **export/import** (tar backup).
- **Two transports**: gRPC (tonic) + REST/JSON (axum) over one shared core.

## Development

The toolchain and all dev dependencies are provided by a Nix flake.

```sh
# with direnv (recommended):
direnv allow

# or directly:
nix develop            # stable shell
nix develop .#nightly  # nightly + miri, for the unsafe-module UB tests
```

Then:

```sh
cargo build --workspace
cargo test  --workspace      # or: cargo nextest run
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo deny check
```

## Workspace

| Crate | Role |
|-------|------|
| `vecvec-core` | Pure, runtime-agnostic engine: index, segments, versioning, payload, persistence. No network deps. |
| `vecvec-proto` | gRPC `.proto` definitions + generated stubs (codegen lands in M3). |
| `vecvec-server` | The server binary (tokio + tonic + axum). |
| `vecvec-cli` | Admin/client CLI. |
| `vecvec-client` | Reusable Rust client library. |

Licensed under MIT OR Apache-2.0.
