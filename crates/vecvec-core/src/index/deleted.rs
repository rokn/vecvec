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
    fn concurrent_disjoint_deletes_lose_nothing() {
        // The raison d'être of SoftDeleteSet is lock-free concurrent deletes via
        // ArcSwap::rcu. N threads each tombstone a disjoint range; none may be lost.
        let s = SoftDeleteSet::new();
        let threads = 8u32;
        let per = 500u32;
        std::thread::scope(|scope| {
            for t in 0..threads {
                let s = &s;
                scope.spawn(move || {
                    for i in 0..per {
                        s.delete(PointId::new(t * per + i));
                    }
                });
            }
        });
        assert_eq!(s.deleted_count(), (threads * per) as usize);
        for id in 0..threads * per {
            assert!(s.is_deleted(PointId::new(id)), "id {id} lost");
        }
    }

    #[test]
    fn concurrent_overlapping_deletes_report_newly_exactly_once() {
        // Under contention every id must transition undeleted->deleted exactly once
        // across all threads — a broken rcu CAS would double-count or lose a delete.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let s = SoftDeleteSet::new();
        let newly = AtomicUsize::new(0);
        let range = 1000u32;
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let s = &s;
                let newly = &newly;
                scope.spawn(move || {
                    for i in 0..range {
                        if s.delete(PointId::new(i)) {
                            newly.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        assert_eq!(s.deleted_count(), range as usize);
        assert_eq!(newly.load(Ordering::Relaxed), range as usize);
    }

    #[test]
    fn snapshot_stable_while_writers_publish() {
        let s = SoftDeleteSet::new();
        for i in 0..100u32 {
            s.delete(PointId::new(i));
        }
        let snap = s.snapshot();
        std::thread::scope(|scope| {
            for t in 0..4u32 {
                let s = &s;
                scope.spawn(move || {
                    for i in 0..100u32 {
                        s.delete(PointId::new(1000 + t * 100 + i));
                    }
                });
            }
        });
        // The pre-spawn snapshot is frozen: unaffected by concurrent publishes.
        assert_eq!(snap.len(), 100);
        assert!(snap.iter().all(|x| x < 100));
        // The live set saw every concurrent delete.
        assert_eq!(s.deleted_count(), 100 + 4 * 100);
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
