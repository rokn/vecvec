//! Strongly-typed identifier newtypes.
//!
//! These are deliberately distinct types (rather than bare integers) so the
//! compiler catches mixing id spaces at subsystem boundaries — a real hazard once
//! segment-local ids, collection-global ids, and version ids all flow through the
//! same code paths (the versioning math in particular depends on never confusing
//! a [`LocalId`] with a [`GlobalId`]).
//!
//! Id spaces:
//! - [`PointId`] / [`LocalId`] — **segment-local** `u32` row indices. [`PointId`]
//!   is the space the [`crate`] index trait operates in (HNSW node ids); [`LocalId`]
//!   names the same physical row from the segment/storage side. They share a width
//!   so conversions are explicit and cheap.
//! - [`GlobalId`] — a **collection-global** `u64`, monotonically allocated by the
//!   collection; stable across segment seal/merge and the unit of the versioning
//!   deletion vectors.
//! - [`SegmentId`] — a monotonic `u64` naming an immutable segment.
//! - [`VersionId`] — a monotonic `u64` naming a committed version/manifest.

use std::fmt;

/// Generates a transparent newtype around an integer with the full set of derive
/// traits we need everywhere (copy, total order, hash) plus `Display`, an inherent
/// `new`/`get`, and lossless `From` conversions to/from the inner type.
macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident($inner:ty)) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name($inner);

        impl $name {
            /// Wraps a raw integer in this id type.
            #[inline]
            pub const fn new(raw: $inner) -> Self {
                Self(raw)
            }

            /// Returns the underlying raw integer.
            #[inline]
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }

        impl From<$inner> for $name {
            #[inline]
            fn from(raw: $inner) -> Self {
                Self(raw)
            }
        }

        impl From<$name> for $inner {
            #[inline]
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
}

id_newtype!(
    /// Segment-local node id as seen by the index layer (HNSW). Always `< segment.len()`.
    PointId(u32)
);
id_newtype!(
    /// Segment-local physical row id, as seen by the storage/segment layer.
    LocalId(u32)
);
id_newtype!(
    /// Collection-global, monotonically-allocated point id; the unit of versioning.
    GlobalId(u64)
);
id_newtype!(
    /// Monotonic id of an immutable segment.
    SegmentId(u64)
);
id_newtype!(
    /// Monotonic id of a committed version (manifest).
    VersionId(u64)
);

impl PointId {
    /// Converts to the matching storage-side [`LocalId`] (same physical row).
    #[inline]
    pub const fn to_local(self) -> LocalId {
        LocalId::new(self.0)
    }
}

impl LocalId {
    /// Converts to the matching index-side [`PointId`] (same physical row).
    #[inline]
    pub const fn to_point(self) -> PointId {
        PointId::new(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn roundtrip_and_accessors() {
        assert_eq!(PointId::new(7).get(), 7);
        assert_eq!(GlobalId::from(42u64).get(), 42);
        assert_eq!(u64::from(SegmentId::new(9)), 9);
    }

    #[test]
    fn display_and_debug() {
        assert_eq!(VersionId::new(3).to_string(), "VersionId(3)");
        assert_eq!(format!("{:?}", PointId::new(1)), "PointId(1)");
    }

    #[test]
    fn ordering_is_by_inner_value() {
        let mut v = [GlobalId::new(5), GlobalId::new(1), GlobalId::new(3)];
        v.sort();
        assert_eq!(v, [GlobalId::new(1), GlobalId::new(3), GlobalId::new(5)]);
    }

    #[test]
    fn hashable() {
        let set: HashSet<SegmentId> = [SegmentId::new(1), SegmentId::new(1), SegmentId::new(2)]
            .into_iter()
            .collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn local_point_conversions_preserve_row() {
        assert_eq!(PointId::new(11).to_local(), LocalId::new(11));
        assert_eq!(LocalId::new(11).to_point(), PointId::new(11));
    }
}
