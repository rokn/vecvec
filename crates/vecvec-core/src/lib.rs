//! `vecvec-core` — the pure, runtime-agnostic engine behind the vecvec vector
//! database.
//!
//! This crate has **no network dependencies**: it holds the index, segment,
//! versioning, payload, and persistence logic so it can be exhaustively unit- and
//! property-tested without spinning up a server. See `BuildPlan.md` at the repo
//! root for the architecture and milestone roadmap.
//!
//! At milestone **M0** this crate provides only the shared low-level primitives
//! that every later subsystem reuses:
//!
//! - [`id`] — strongly-typed id newtypes ([`PointId`], [`LocalId`], [`GlobalId`],
//!   [`SegmentId`], [`VersionId`]).
//! - [`ordered`] — total-ordering wrappers for floats ([`OrderedF32`],
//!   [`OrderedF64`]) so scores can live in `BTreeMap`/`BinaryHeap`.
//! - [`error`] — the crate-wide [`CoreError`] / [`Result`] types.
//! - [`persist::atomic`] — crash-safe atomic file writes (temp → fsync → rename →
//!   fsync-dir) with magic + version + CRC framing.

pub mod distance;
pub mod error;
pub mod id;
pub mod index;
pub mod ordered;
pub mod persist;
pub mod vector;

pub use distance::{DistanceKernel, Metric};
pub use error::{CoreError, Result};
pub use id::{GlobalId, LocalId, PointId, SegmentId, VersionId};
pub use index::{
    FilterContext, FlatIndex, Index, ScoredPoint, SearchParams, SoftDeleteSet, brute_force_topk,
};
pub use ordered::{OrderedF32, OrderedF64};
pub use vector::VectorStorage;

/// The crate version, surfaced so binaries can report a single source of truth.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
