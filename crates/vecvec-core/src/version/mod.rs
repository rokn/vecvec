//! Versioning — the git-like commit DAG over immutable segments.
//!
//! A *commit* is a [`Manifest`] listing the sealed segments and a frozen deletion
//! vector; snapshots are cheap because segments are `Arc`-shared, so a version costs
//! a manifest plus refcount bumps, never a vector copy. Time-travel resolves a
//! [`VersionSelector`] to a manifest and searches exactly its segment set;
//! branch/tag are movable pointers; diff is a set operation; restore is a forward
//! commit re-pointing at an old segment set (history is preserved). See
//! `BuildPlan.md`.

pub mod deletion;
pub mod manifest;
pub mod policy;
pub mod store;

pub use deletion::DeletionVector;
pub use manifest::{Manifest, SegmentRef};
pub use policy::{Clock, SystemClock, TriggerEvaluator, VersioningPolicy};
pub use store::{Diff, GcReport, RetentionRules, VersionError, VersionSelector, VersionStore};
