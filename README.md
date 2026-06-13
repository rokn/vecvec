# vecvec

A super-fast, in-memory-primary vector database in Rust with **automatic, git-like
versioning** of collections (immutable snapshots, time-travel query, branch, diff,
restore) on top of an HNSW index. See [BuildPlan.md](./BuildPlan.md) for the full
architecture and milestone roadmap.

> Status: **M0** — workspace scaffold and shared primitives. Not yet runnable as a server
> (networking lands in M3).

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
