//! Version manifests — the "commit" object.
//!
//! A [`Manifest`] is an immutable record of a committed version: its place in the
//! DAG (`version` + `parent`), the **full** list of sealed segments it references
//! (git-style: a complete snapshot, not a delta), and the deletion vector frozen at
//! commit time. Because segments are immutable and shared by `Arc`, every manifest
//! that lists a segment shares the same physical bytes — the structural sharing that
//! makes snapshots O(#segments) and free of vector copies.

use serde::{Deserialize, Serialize};

use super::deletion::DeletionVector;

/// A reference to a sealed segment within a version, with the disjoint, contiguous
/// global-id range it owns (used to prune diffs cheaply).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRef {
    /// The segment id.
    pub id: u64,
    /// Inclusive lower bound of this segment's global-id range.
    pub id_lo: u64,
    /// Inclusive upper bound of this segment's global-id range.
    pub id_hi: u64,
    /// The number of rows in the segment.
    pub count: u64,
}

/// An immutable committed version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Monotonic version id.
    pub version: u64,
    /// The version this was committed on top of (`None` for the first).
    pub parent: Option<u64>,
    /// Wall-clock commit time (unix millis); informational.
    pub created_at_ms: u64,
    /// What caused the commit (e.g. "manual", "every_n_writes", "interval").
    pub trigger: String,
    /// Optional human message.
    pub message: Option<String>,
    /// The full set of sealed segments this version references.
    pub segments: Vec<SegmentRef>,
    /// Tombstones frozen at commit time.
    pub deletions: DeletionVector,
}

impl Manifest {
    /// The segment ids this version references.
    pub fn segment_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.segments.iter().map(|s| s.id)
    }

    /// The number of live rows in this version (total segment rows minus tombstones
    /// that fall inside this version's segment ranges).
    pub fn live_count(&self) -> u64 {
        let total: u64 = self.segments.iter().map(|s| s.count).sum();
        total.saturating_sub(self.deletions.len())
    }
}
