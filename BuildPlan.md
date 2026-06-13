# vecvec — a super-fast, in-memory Rust vector DB with automatic git-like versioning

## Context

`/Users/antoniomindov/programming/AI/vecvec` is an empty directory. We're building a
brand-new vector database, **vecvec**, from scratch. The need: a high-performance store
for embeddings that (a) runs primarily from RAM, (b) supports pluggable indexes (HNSW
first), (c) attaches arbitrary metadata to each entry with filtered search, and — the
**differentiator** — (d) performs **automatic version control**: git-like, immutable,
whole-collection snapshots triggered by configurable rules, with time-travel query,
branch, diff, and restore.

The hard technical problem is reconciling a *mutable* HNSW graph with *cheap, immutable*
versioning. The resolution (validated against how Lucene, Qdrant, Lance/LanceDB, and
Iceberg actually work) is a **segment-based architecture**: a collection is many immutable
*sealed* segments plus one small mutable *appendable* segment; a "commit" is a tiny
manifest that lists sealed segment IDs by reference. Taking a snapshot is `O(#segments)`
`Arc` refcount bumps with **zero vector copies** — structural sharing, exactly like git
content-addressing realized through `Arc`.

### Locked decisions (confirmed with the user)

| Decision | Choice |
|---|---|
| Language | **Rust** |
| Deployment | **Standalone server**, gRPC (tonic 0.14) **+** HTTP/REST (axum 0.8) |
| Versioning model | **Global git-like snapshots** (branch / diff / restore / time-travel) |
| Persistence | **RAM-primary** + WAL (durability) + rkyv snapshot files (versions) |
| Scope | **Exhaustive** — full Qdrant-class feature surface, specified up front |
| Vector storage | **f32 originals + uint8 scalar quantization** with f32 rescore (≈4× RAM saving) |
| API | **Own clean schema**, Qdrant-inspired (not wire-compatible) |
| Auto-commit triggers (v1) | **every-N-writes + time-interval** (drift-based trigger deferred) |
| Topology (v1) | **single-node** (distribution/sharding deferred) |

---

## Architecture overview

```
                 ┌──────────────── vecvec-server (binary) ────────────────┐
   gRPC :6334 ──▶│  tonic services ─┐                                      │
   REST :6333 ──▶│  axum router ────┴─▶ VecVecService (one core facade)    │
                 │                          │                              │
                 │                          ▼  BlockingBridge              │
                 │              (tokio Semaphore + rayon pool + oneshot)    │
                 │   background tasks: versioning-trigger · optimizer ·     │
                 │                     checkpointer · GC                    │
                 └──────────────────────────┼──────────────────────────────┘
                                            ▼
        ┌──────────────────────── vecvec-core (pure logic, no net deps) ───────────┐
        │  Collection                                                              │
        │   ├─ SegmentSet (ArcSwap)  ── lock-free point-in-time reader             │
        │   │    ├─ [sealed Segment]·····immutable {VectorBlock f32+u8,            │
        │   │    │                          own HNSW, PayloadBlock, IdMap, deleted}│
        │   │    └─ AppendableSegment····mutable, flat brute-force search          │
        │   ├─ VersionStore  ── commit DAG, HEAD/branches/tags, ReadView, GC       │
        │   └─ Durability   ── WAL-first, checkpoint, recovery                     │
        └──────────────────────────────────────────────────────────────────────────┘
```

**Write path:** request → WAL (assign LSN, fsync, *then* ack) → apply to AppendableSegment
(via the single `replay::apply` path shared by live-apply and recovery). A background
optimizer **seals** a full appendable segment into an immutable segment (builds HNSW +
quantizes + writes the rkyv `.seg` file), and merges/compacts small segments.

**Read path:** `query` resolves the `at:` selector → a `ReadView` (an `Arc`-cloned segment
set + per-version deletion bitmap), fans out across live segments on the rayon pool
(quantized ANN → f32 rescore), and k-way-merges per-segment top-k.

**Commit:** seal the working head → write the new `Manifest` (lists sealed segment refs +
deletion vectors) → atomic HEAD swap. Auto-fires when the `VersioningPolicy` trigger
(every-N-writes / interval) is met.

---

## The differentiator: versioning design

- **Commit = an immutable `Manifest`** `{version: u64, parent: Option<u64>, created_at,
  trigger, message, tag?, segments: Vec<SegmentRef>, deletions: Vec<DeletionVectorRef>,
  schema, index_meta}`. This is git's commit semantics (parent pointer + a *full* segment
  list, never a diff) and the Lance/Iceberg manifest model.
- **Segment = the "tree/blob"**: immutable, `Arc`-held, identified by a monotonic
  `SegmentId`, owning a **disjoint, contiguous** global-id range `[id_lo..=id_hi]` (a hard
  invariant the seal/merge paths must enforce — diff/restore/GC depend on it).
- **Cheap snapshot via structural sharing**: a snapshot `Arc`-clones `Vec<Arc<Segment>>`
  (`O(#segments)` refcount bumps, zero vector/graph copies). Unchanged segments are
  physically shared by every version that references them.
- **Updates/deletes never rewrite sealed segments**: an update = soft-delete old id
  (tombstone) + append a new row to the appendable segment; each version carries a
  `roaring` **DeletionVector** over global ids. `live_set(v) = ⋃ segment ranges − ⋃ dvs`.
- **Time-travel** = a *stateless* per-request `at:<version|tag|branch>` parameter on the
  query endpoint (never a global "checkout mode" — that's a multi-client footgun).
- **Branch / tag** = named movable pointers (`branches`, `tags` maps); a tag also protects
  its version from GC. **Diff(A,B)** = roaring set-difference of live-id sets, pruned by
  per-segment id-range/count so untouched segments are skipped. **Restore(N)** = a *new
  forward commit* re-pointing at version N's segment set (history preserved, auditable).
- **Auto-commit** = `VersioningPolicy { every_n_writes: Option<u64>, interval:
  Option<Duration> }`; writes accumulate in a `WorkingHead` (git staging) and a
  `TriggerEvaluator` seals it on the rule. Never commit-per-upsert (version explosion).
- **GC/retention** = mark-and-sweep refcount over *retained* manifests with a grace window
  (guards the commit-vs-GC race, ref. Lance #3718); rules: keep last K / newer than T /
  always keep tagged+branch heads / keep any version with an open reader.

---

## Workspace layout

A Cargo **workspace** with a shared dependency table and single `Cargo.lock`.
`vecvec-core` has **zero network deps** so its large test matrix builds without tonic/axum.

```
vecvec/
├── Cargo.toml                      # [workspace] members + shared [workspace.dependencies]
├── rust-toolchain.toml             # pinned toolchain
├── deny.toml                       # cargo-deny licenses/advisories/bans
├── crates/
│   ├── vecvec-core/                # pure logic: index, segments, versioning, payload, persist
│   ├── vecvec-proto/               # .proto + tonic-build codegen + wire<->core convert
│   ├── vecvec-server/              # the binary: tokio + tonic + axum + tasks + observability
│   ├── vecvec-cli/                 # clap admin/client CLI
│   └── vecvec-client/              # (optional) reusable Rust client crate
└── benches/                        # criterion + recall@k harness (own crate)
```

### `vecvec-core` source tree (the heart)

```
src/
├── distance/   mod·kernel·scalar·x86·aarch64·dispatch·simsimd_oracle   # Metric + SIMD, runtime-dispatched once
├── quantization/ mod·scalar_u8·block·kernel·rescore                    # uint8 SQ + f32 rescore
├── index/      mod·flat·deleted·filter
│   └── hnsw/   mod·builder·graph·links·search·heuristic·visited·rng·entry_points
├── segment/    mod·segment·vector_block·payload_block·id_map·appendable·seal·set·search·optimizer·stats
├── storage/    mod·seg_file·store·mmap·atomic_io·gc
├── version/    mod·manifest·deletion·store·working·policy·view·refs·diff·restore·gc·persist
├── payload/    mod·value·path·block·archive
│   └── index/  mod·keyword·numeric·bool·geo·text·histogram
├── filter/     mod·condition·eval·estimate·compile
├── query/      mod·model·planner·execute·prefetch·fusion·reco·readview
└── persist/    mod·atomic·io_pool·checkpoint·snapshot·manifest·recovery·layout·export·replay
    └── wal/    mod·record·fsync
```

### Stable trait seams (define early; everything plugs into these)

- `Index` — `build/insert/search(query,k,ef,filter)/delete/len/iter`. HNSW + Flat impls;
  future IVF/DiskANN slot in unchanged. `PointId` is **segment-local `u32`**.
- `Distance` / `DistanceKernel` — `Metric{Cosine,Dot,Euclidean}` resolved **once** into
  stored `f32`/`u8` fn pointers (AVX-512F/AVX2+FMA/SSE/NEON/scalar); `simsimd` = test oracle.
- `FieldIndex` — keyword/numeric/bool/geo/text/uuid backends behind one trait.
- `FilterContext` / `CompiledFilter` — visitor + cardinality estimator for the planner.
- `VersionStore` — commit DAG + `ReadView` resolution (`at:`), pinnable open readers.
- `Clock` — injectable so interval triggers and tests are deterministic.
- `VecVecService` — the single core facade both gRPC and REST call (no self-RPC).

### Key HNSW parameters (locked defaults)

`M=16`, `M_max0=32`, `ef_construction=128`, `ef_search=64–128` (enforced `≥ k`),
`mL=1/ln(16)`, seeded **deterministic** level RNG (`seed ^ point_id`, reproducible for
replay), **SELECT-NEIGHBORS-HEURISTIC** (Malkov & Yashunin Alg. 4 + `keepPrunedConnections`),
level-0 links in a flat contiguous arena / CSR upper layers (rkyv-friendly), soft-delete via
atomic roaring bitset (rebuild segment when `deleted_ratio > 0.30`), filtered search via
filter-aware traversal + additional per-payload-value filter edges, `full_scan_threshold ≈
10_000` for the exact-vs-filterable planner switch.

---

## Build roadmap (15 milestones)

Build a **runnable vertical slice first** (M0–M3), then deepen one subsystem at a time
behind the stable seams. Each milestone is independently verifiable by its own exit tests;
tests are written **alongside** code, and brute-force/reference **oracles are built before
the optimized paths they validate**.

| # | Milestone | Deliverable (what works) | Deps |
|---|---|---|---|
| **M0** | Workspace + CI scaffold + shared primitives | 5 crates compile & wire together; pinned toolchain; `atomic_write` (temp→fsync→rename→fsync-dir) w/ crc32 framing; `OrderedF32`, id newtypes, `thiserror` errors; green fmt+clippy+test CI on empty workspace | — |
| **M1** | Distance layer (scalar) + dispatch | `Metric` + `DistanceKernel` w/ fn-pointers; correct scalar dot/cosine/sq-euclid (f32 + u8); SIMD slots stubbed; `simsimd` oracle behind `oracle` feature | M0 |
| **M2** | `Index` trait + `VectorStorage` + `FlatIndex` + soft-delete + `FilterContext` | the pluggability seam; `brute_force_topk` oracle reused everywhere; L2-normalize-on-ingest for cosine; `SoftDeleteSet`; filter visitor | M1 |
| **M3** | **Vertical slice**: minimal Collection + `SegmentSet(ArcSwap)` + gRPC upsert/search | end-to-end runnable server; flat-search appendable segment; `BlockingBridge`; minimal proto (Create, streaming Upsert, Query); health+reflection | M2 |
| **M4** | Real HNSW (concurrent + deterministic builder → sealed lock-free graph) | from-scratch HNSW: `GraphLinks` layout, Alg.4 heuristic, rayon-parallel build w/ `ready` bitset, seal conversion, `HnswIndex` | M2 |
| **M5** | Segment seal + rkyv `.seg` + zero-copy mmap + `SegmentStore` + fan-out | immutable `Segment`; mmap cast-and-go load; cross-segment parallel top-k + global k-way merge; Weak cache + refcount table | M3, M4 |
| **M6** | WAL-first durability + recovery + checkpoint ordering | `WalManager` (qdrant `wal`), `WalOp`/bincode, `replay::apply` (single path), fsync modes + group-commit, `CheckpointCoordinator`, `recover_all` w/ torn-tail + bad-collection quarantine | M5 |
| **M7** | **Versioning engine — the differentiator** | `Manifest`/`SegmentRef`/`DeletionVector`, `VersionStore` (HEAD/branches/tags/working head), `VersioningPolicy`+`TriggerEvaluator`, atomic commit, `ReadView`+`at:` resolution, diff/restore, mark-and-sweep GC | M6 |
| **M8** | uint8 scalar quantization + rescore at seal | `ScalarQuantizer` (per-segment/per-dim affine), `QuantizedVectorBlock` + i32-accum kernel, `quantized_rescore_search` (oversample → f32 rescore) | M5 |
| **M9** | Payload model + field indexes + Filter DSL + estimator + cost planner | `Value`/`JsonPath`/`PayloadBlock`; keyword/numeric/bool/geo/text indexes; `Filter{must/should/must_not/min_should}`; cardinality estimator; `plan_segment` exact-vs-filterable switch | M4, M5 |
| **M10** | Optimizer/compactor + GC + checkpoint scheduling | seal-by-size/age, merge-small bin-packing, rebuild-on-deleted_ratio; background tasks (trigger·optimizer·checkpointer·gc) cancellable; file-level mark-and-sweep | M7, M8 |
| **M11** | Full polymorphic `/query`: recommend/discover/context/order_by/fusion + prefetch DAG | full `Query` enum + nestable `Prefetch` (hybrid/rerank), `execute_query` over `ReadView`, RRF/DBSF fusion, recommend/discover scoring | M9, M7 |
| **M12** | Full API surface + REST gateway + backpressure + observability + multi-tenancy | full Points CRUD/scroll/count, payload & vector verbs, payload-index, full Versioning+Exports services; axum REST mirroring gRPC 1:1 via shared `VecVecService`; tower ConcurrencyLimit/LoadShed; tracing+prometheus; tenant filter injection | M11, M6 |
| **M13** | CLI + exports (tar backup) + client crate | `vecvec-cli` (clap) over tonic client; JSONL streaming upsert; commit/branch/tag/diff/restore/versions/export; `persist/export.rs` tar of a version's segment-set | M12 |
| **M14** | SIMD acceleration + hardening + bench suite + CI regression gates | hand-written AVX-512/AVX2/SSE/NEON kernels behind once-dispatch; criterion + recall@k harness (SIFT/GloVe); miri on unsafe; loom soak; cargo-deny; CI fails on recall/latency regression | M3, M8, M11 |

**Critical path:** `M0→M1→M2→M3→M4→M5→M6→M7→M9→M11→M12→M13`. The single longest serial
constraint is **M5→M6→M7** (durable segments → WAL/recovery → versioning): versioning can't
be correct until segments are durable and seal with disjoint-contiguous id ranges — which is
why the differentiator lands at M7, as early as is reasonable on top of the segment model.

**Parallelizable forks (token cost is not a constraint here — fan these out):**
SIMD kernels (M14's distance part) anytime after **M1** against the scalar oracle · M4 (HNSW)
and M5 storage plumbing concurrently (M5 starts against `FlatIndex` + stubbed `SerGraphLayers`) ·
the serde-pure `Value`/Filter DSL + brute-force reference evaluator (front half of M9) right
after **M2** · M8 quantization fully parallel to M6/M7 (depends only on the sealed Segment) ·
proto `.proto` + `convert.rs` + REST DTOs throughout, against the evolving core public types.

---

## Testing strategy (oracle-driven, layered)

- **Unit** — distance kernels vs hand-computed values across odd dims `{1,3,7,8,15,16,128,769}`;
  Alg.4 on hand-worked geometry; range/histogram estimators; filter `evaluate()` per matcher
  (geo point-on-edge, inclusive range boundaries); RRF/DBSF math; trigger exact-N firing.
- **Oracle/differential** — `brute_force_topk` is recall ground truth for HNSW & quantized
  search; a brute-force filter evaluator is the oracle for `evaluate()` + estimator sanity;
  scalar kernels are the oracle for SIMD; `simsimd` (`oracle` feature) cross-checks the
  hand-written kernels within `1e-4`.
- **Property-based (proptest)** — the cornerstone: **kill-9 crash/recovery** model test
  (random `WalOp` sequences × random crash point × all fsync modes ⇒ every acked op survives,
  no torn op applied, recovered state == deterministic model); plus `convert.rs` pb↔core
  roundtrip, `DeletionVector` u64-boundary roundtrip, filter boolean-combinator equivalence.
- **Recall validation (gated)** — `recall@10 ≥ 0.95` HNSW, `≥ 0.90` quantized+rescore,
  `≥ 0.99` exact-vs-filterable-HNSW on selective filters, on seeded Gaussian-cluster sets in
  CI; SIFT/GloVe in the bench job.
- **Determinism** — same-seed single-thread vs rayon build seal to **byte-identical**
  `GraphLinks`; two independent recoveries produce byte-identical rebuilt state.
- **Concurrency/crash** — loom-style model of the `ready`-bit Release/Acquire handshake and
  ArcSwap snapshots; "no RwLock guard across `.await`" stress; concurrent append-while-seal;
  fault injection at every checkpoint-ordering step and atomic-rename boundary.
- **Integration** — in-process server harness exercising
  `upsert→query→commit→time-travel→branch→diff→restore` over **both gRPC and REST**, asserting
  cross-transport consistency; backpressure; streaming upsert; reflection; health gating.

## Benchmark plan ("super-fast" is measured, not claimed)

criterion micro-benches (per-arch f32/u8 kernels, `VisitedPool` reset, neighbor heuristic) ·
seal/build scaling across rayon threads · search latency p50/p99/p99.9 vs `ef`, cross-segment
fan-out vs `#segments` (validates the `O(#segments)` merge claim), filtered-search across
selectivity (exact↔filterable crossover near `full_scan_threshold`), quantized-rescore vs
f32-only · durability ack p50/p99 for per-batch vs group-commit{1,5,10ms} vs periodic ·
**ann-benchmarks-style recall/QPS Pareto** on SIFT1M (128d) & GloVe (100d, angular), tracked
over time · in-process vs gRPC vs REST overhead (prove `BlockingBridge` adds negligible cost).
CI stores baselines and fails on regression (>5% latency or >0.01 recall drop).

## CI strategy

Workspace-first incremental builds (sccache + registry/target cache keyed on `Cargo.lock`).
Stages: `fmt --check` → `clippy --all-targets -D warnings` → `build` → `test` (+ `oracle`
feature, proptest elevated case counts for the crash test) → **miri** on the unsafe-heavy
modules (mmap cast-and-go, flat arenas, self-referential Segment) → **cargo-deny** → a
separate scheduled **bench-regression** job. Multi-platform matrix x86_64 (AVX2/AVX-512) +
aarch64 (NEON); scalar fallback always compiled & tested. Branch protection requires
fmt+clippy+test+miri+cargo-deny green.

---

## Top risks & mitigations

1. **Zero-copy mmap cast-and-go is the deepest unsafe surface** (rkyv 0.8 needs ≥16-byte
   alignment; `Segment` is self-referential `Mmap`+slices; a layout change silently breaks
   `access_unchecked`). → Choose the self-referential ownership strategy (yoke/ouroboros vs a
   manual unsafe wrapper) **before** writing `vector_block.rs`. Carry `magic`+`format_version`
   in every header and reject mismatches; crc32fast first line + rkyv `bytecheck` `access()`
   (validated) on cold/untrusted loads; `access_unchecked` only for files written this
   process; assert `ptr % 16 == 0`; gate the module under miri + a byte-flip/truncate test.
2. **Concurrent build + lock-free sealed search memory-ordering hazards.** → `ready` bit set
   with **Release after** all links written, read with **Acquire**; bake level assignment into
   a pure `seed ^ point_id` function (never a shared RNG); cover with the determinism test +
   a loom model of the ready-bit handshake; confine RwLock guards inside rayon closures.
3. **Durability ordering contract** (WAL-first, ack-after-fsync; checkpoint
   snapshot-durable → HEAD-swap → `prefix_truncate` **last**). → Encode checkpoint as one
   linear function with no early-truncate branch; gate truncate behind the HEAD-fsync result;
   make `replay::apply` the SAME code for live-apply and recovery; prove with the proptest
   kill-9 test + per-step fault injection at elevated case counts.
4. **Versioning invariants leak into other subsystems** (live-set/diff assume disjoint,
   contiguous global-id ranges; global id owned by the Collection). → Lock the
   disjoint-contiguous invariant in seal **and** merge (assert + test) before `version/diff.rs`;
   centralize global-id allocation in the Collection; feed GC an externally-computed retained
   id set (versioning owns policy, storage owns file deletion).
5. **Cost-based planner mis-estimation** (wrong exact-vs-HNSW choice; filtered-HNSW recall
   collapse under high selectivity). → Keep estimates conservative (primary popcount never
   under-counts; geo uses covering-cell upper bound); validate the switch is *score-correct*
   (exact-vs-filterable recall ≥ 0.99); build per-payload-value filter edges at seal + enlarge
   `ef` under high selectivity + plateau guard → plain HNSW; benchmark the crossover.
6. **Ecosystem version pinning** (tonic 0.14 reorg: `tonic-prost-build`/`tonic-prost`/`prost`
   0.14 pinned together; rkyv 0.8 + bytecheck + memmap2 + qdrant `wal` are a tight on-disk
   contract). → Pin these as sets in the workspace table; cargo-deny blocks duplicates; any
   rkyv-touching bump = a `format_version` bump with a migration/rejection path + roundtrip
   test; verify proto codegen emits `FILE_DESCRIPTOR_SET` and reflection round-trips in CI.
7. **Scope/sequencing stall** (huge surface; wrong order → long non-runnable stretches). →
   Enforce the M0–M3 thin vertical slice first so there's always a runnable, demoable server;
   deepen one subsystem at a time behind stable seams; stub `VersionStore` and use
   Flat-before-HNSW so payload/query/segment work proceeds in parallel.

---

## Verification (end-to-end)

At each milestone, run its exit-criteria tests (table above). The whole-system acceptance
check, runnable from **M12** onward:

1. **Build & boot:** `cargo build --workspace` then run `vecvec-server` with a temp
   `data_dir`; confirm gRPC `:6334` + REST `:6333` up, health flips `SERVING` after recovery,
   `grpcurl` reflection lists services, `/metrics` serves Prometheus.
2. **Core lifecycle (drive via `vecvec-cli` over both transports):**
   - create a collection (cosine, dim N, HNSW params, `every_n_writes` policy);
   - client-streaming upsert of M points with payloads from JSONL;
   - `query` nearest-k → assert it matches a brute-force ground truth (`recall@10 ≥ 0.95`);
   - filtered `query` → assert results satisfy the filter and match the exact oracle.
3. **The differentiator:** keep upserting until the trigger auto-commits (or `commit`
   explicitly) → `versions` shows the commit DAG; mutate/delete points; `query --at v1`
   returns the **pre-commit** set (snapshot isolation); `branch staging` from v1, advance it,
   `diff v1..v2` shows correct added/updated/deleted; `restore v1` lands a new forward commit
   and history is preserved; export a version's tar and re-import into a fresh collection,
   asserting equality.
4. **Durability:** `kill -9` the server mid-write; restart; assert every acked write survived,
   no torn op applied, and HEAD/versions are consistent (also covered by the proptest model).
5. **Performance:** run the criterion + `recall@k` bench harness on SIFT1M/GloVe; record the
   recall@QPS Pareto and p50/p99 latency; confirm gRPC query p99 is within a small constant of
   in-process search p99 (BlockingBridge overhead is negligible); CI guards regressions.

---

## Deferred to later phases (explicitly out of v1 scope)

Drift-based auto-commit trigger · product/binary quantization & on-disk/mmap-resident vectors
beyond uint8 SQ · distribution/sharding/replication (write-ordering, consistency factor) ·
COW-HNSW per-vertex versioning (segmentation supersedes it) · object-store (S3/GCS) backend
for segments/exports · sparse vectors & multi-named-vector schema evolution under time-travel ·
token-pinned branch/tag + tenant simultaneously. Each is a clean extension behind an existing
seam (`Index`, `FieldIndex`, `VersionStore`, `Quantizer`, storage backend).
