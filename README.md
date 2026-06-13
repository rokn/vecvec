# vecvec

A super-fast, in-memory-primary vector database in Rust with **automatic, git-like
versioning** of collections (immutable snapshots, time-travel query, branch, diff,
restore) on top of an HNSW index. See [BuildPlan.md](./BuildPlan.md) for the full
architecture and milestone roadmap.

> Status: **M3** — runnable gRPC vertical slice. The server starts and serves
> `Collections.Create`, `Points.Upsert` (client-streaming), and `Query.Query`, with
> health + reflection. HNSW, durability, versioning, payload/filter, and REST land
> in later milestones.

Run it:

```sh
cargo run -p vecvec-server          # serves grpc://127.0.0.1:6334
grpcurl -plaintext 127.0.0.1:6334 list
```

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
