//! An immutable set of sealed segments — the unit of point-in-time snapshots.
//!
//! A [`SegmentSet`] is held behind an `ArcSwap` in the collection. A reader takes a
//! one-load snapshot (`Arc<SegmentSet>`) and scans it lock-free while writers
//! publish new sets; sealing/compaction produce a new set that shares all unchanged
//! segments by `Arc` (the structural sharing that makes versioning snapshots cheap
//! in M7).

use std::sync::Arc;

use super::sealed::SealedSegment;

/// An immutable list of sealed segments.
#[derive(Default, Clone)]
pub struct SegmentSet {
    sealed: Vec<Arc<SealedSegment>>,
}

impl SegmentSet {
    /// The empty set.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The number of sealed segments.
    #[inline]
    pub fn len(&self) -> usize {
        self.sealed.len()
    }

    /// Whether there are no sealed segments.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sealed.is_empty()
    }

    /// Iterates the sealed segments.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<SealedSegment>> {
        self.sealed.iter()
    }

    /// The total number of live rows across all sealed segments.
    pub fn total_live(&self) -> usize {
        self.sealed.iter().map(|s| s.live_len()).sum()
    }

    /// Returns a new set with `segment` appended, sharing all existing segments by
    /// reference (the basis for cheap snapshots).
    pub(crate) fn with_appended(&self, segment: Arc<SealedSegment>) -> Self {
        let mut sealed = self.sealed.clone();
        sealed.push(segment);
        Self { sealed }
    }
}
