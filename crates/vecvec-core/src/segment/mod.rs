//! Segments: the immutable-sealed + one-mutable-appendable storage model.
//!
//! A collection's data is one [`AppendableSegment`] (mutable, RAM, flat search)
//! plus a [`SegmentSet`] of immutable [`SealedSegment`]s. This split is what lets
//! versioning take cheap structural-sharing snapshots (M7): a commit references
//! already-sealed segments by `Arc`. See `BuildPlan.md`.

mod appendable;
mod id_map;
mod sealed;
mod search;
mod set;

pub use appendable::AppendableSegment;
pub use id_map::IdMap;
pub use sealed::SealedSegment;
pub use set::SegmentSet;
