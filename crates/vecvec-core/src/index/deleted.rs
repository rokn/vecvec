//! Soft-delete tracking.
//!
//! Sealed segments are immutable, so a delete can't physically remove a row; it
//! sets a tombstone in a [`SoftDeleteSet`]. Search traverses *through* tombstones
//! (to preserve graph connectivity in HNSW) but never returns them. When the
//! tombstone ratio gets high enough ([`REBUILD_DELETED_RATIO`]) the optimizer
//! rebuilds the segment to reclaim the space (M10).
//!
//! The set is a [`RoaringBitmap`] behind an [`ArcSwap`] so readers can grab a
//! cheap, point-in-time [`snapshot`](SoftDeleteSet::snapshot) (one atomic load)
//! and scan it lock-free while concurrent deletes publish new versions.

use std::sync::Arc;

use arc_swap::ArcSwap;
use roaring::RoaringBitmap;

use crate::id::PointId;

/// The tombstone ratio at or above which a segment should be rebuilt to reclaim
/// space (see M10's optimizer).
pub const REBUILD_DELETED_RATIO: f32 = 0.30;

/// Returns whether a segment with the given live/total counts has accumulated
/// enough tombstones to warrant a rebuild.
#[inline]
pub fn should_rebuild(deleted: usize, total: usize) -> bool {
    deleted_ratio(deleted, total) >= REBUILD_DELETED_RATIO
}

/// The fraction of a segment's rows that are tombstoned (`0.0` when empty).
#[inline]
pub fn deleted_ratio(deleted: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        deleted as f32 / total as f32
    }
}

/// A concurrent, point-in-time-snapshottable set of tombstoned segment-local ids.
#[derive(Default)]
pub struct SoftDeleteSet {
    bitmap: ArcSwap<RoaringBitmap>,
}

impl SoftDeleteSet {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks `id` deleted. Returns `true` if it was newly tombstoned, `false` if it
    /// was already deleted.
    pub fn delete(&self, id: PointId) -> bool {
        let mut newly = false;
        self.bitmap.rcu(|current| {
            let mut next = RoaringBitmap::clone(current);
            newly = next.insert(id.get());
            next
        });
        newly
    }

    /// Whether `id` is currently tombstoned.
    #[inline]
    pub fn is_deleted(&self, id: PointId) -> bool {
        self.bitmap.load().contains(id.get())
    }

    /// The number of tombstoned ids.
    #[inline]
    pub fn deleted_count(&self) -> usize {
        self.bitmap.load().len() as usize
    }

    /// Takes a cheap, immutable, point-in-time snapshot of the tombstone set.
    /// Deletes published after this call do not affect the returned snapshot.
    #[inline]
    pub fn snapshot(&self) -> Arc<RoaringBitmap> {
        self.bitmap.load_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delete_is_idempotent_and_reports_newness() {
        let s = SoftDeleteSet::new();
        assert!(s.delete(PointId::new(5)));
        assert!(!s.delete(PointId::new(5)));
        assert!(s.is_deleted(PointId::new(5)));
        assert!(!s.is_deleted(PointId::new(6)));
        assert_eq!(s.deleted_count(), 1);
    }

    #[test]
    fn snapshot_is_point_in_time() {
        let s = SoftDeleteSet::new();
        s.delete(PointId::new(1));
        let snap = s.snapshot();
        s.delete(PointId::new(2));
        assert!(snap.contains(1));
        assert!(!snap.contains(2)); // snapshot frozen before the second delete
        assert!(s.is_deleted(PointId::new(2))); // live set reflects it
    }

    #[test]
    fn rebuild_threshold_crossing() {
        // 30% is the trigger.
        assert!(!should_rebuild(29, 100));
        assert!(should_rebuild(30, 100));
        assert!(should_rebuild(31, 100));
        assert!(!should_rebuild(0, 0));
        assert!((deleted_ratio(3, 10) - 0.3).abs() < 1e-6);
    }
}
