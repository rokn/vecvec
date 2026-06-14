//! Per-version deletion vectors.
//!
//! In the versioned model a delete is **not** a mutation of an (immutable) segment —
//! it's a tombstone recorded in a [`DeletionVector`] over collection-global ids,
//! layered on top of the segments. A version freezes a *clone* of the live deletion
//! vector, so deletes made after a commit can never change what an older version
//! sees (snapshot isolation). A [`roaring::RoaringTreemap`] keeps this compact even
//! for huge id spaces, and gives fast set-difference for diffing two versions.

use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::id::GlobalId;

/// A compressed set of tombstoned global ids.
#[derive(Clone, Default)]
pub struct DeletionVector {
    bitmap: RoaringTreemap,
}

impl std::fmt::Debug for DeletionVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeletionVector({} tombstones)", self.bitmap.len())
    }
}

impl DeletionVector {
    /// An empty deletion vector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks `id` deleted; returns `true` if it was newly added.
    pub fn insert(&mut self, id: GlobalId) -> bool {
        self.bitmap.insert(id.get())
    }

    /// Whether `id` is tombstoned.
    #[inline]
    pub fn contains(&self, id: GlobalId) -> bool {
        self.bitmap.contains(id.get())
    }

    /// The number of tombstoned ids.
    #[inline]
    pub fn len(&self) -> u64 {
        self.bitmap.len()
    }

    /// Whether nothing is tombstoned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    /// Iterates the tombstoned ids in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = GlobalId> + '_ {
        self.bitmap.iter().map(GlobalId::new)
    }

    /// The ids tombstoned in `self` but not in `other` (set difference).
    pub fn difference(&self, other: &DeletionVector) -> impl Iterator<Item = GlobalId> + '_ {
        (&self.bitmap - &other.bitmap)
            .into_iter()
            .map(GlobalId::new)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

// Serialized as the plain list of ids — simple, portable, and small for the typical
// (sparse) case.
impl Serialize for DeletionVector {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.bitmap.iter().collect::<Vec<u64>>().serialize(s)
    }
}

impl<'de> Deserialize<'de> for DeletionVector {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let ids = Vec::<u64>::deserialize(d)?;
        let mut bitmap = RoaringTreemap::new();
        bitmap.extend(ids);
        Ok(Self { bitmap })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains_len() {
        let mut dv = DeletionVector::new();
        assert!(dv.insert(GlobalId::new(5)));
        assert!(!dv.insert(GlobalId::new(5)));
        assert!(dv.contains(GlobalId::new(5)));
        assert!(!dv.contains(GlobalId::new(6)));
        assert_eq!(dv.len(), 1);
    }

    #[test]
    fn difference_is_set_minus() {
        let mut a = DeletionVector::new();
        for i in [1u64, 2, 3, 4] {
            a.insert(GlobalId::new(i));
        }
        let mut b = DeletionVector::new();
        for i in [2u64, 4] {
            b.insert(GlobalId::new(i));
        }
        let only_a: Vec<u64> = a.difference(&b).map(|g| g.get()).collect();
        assert_eq!(only_a, vec![1, 3]);
    }

    #[test]
    fn serde_roundtrip() {
        let mut dv = DeletionVector::new();
        for i in [10u64, 20, 4_000_000_000] {
            dv.insert(GlobalId::new(i));
        }
        let bytes = rmp_serde::to_vec(&dv).unwrap();
        let back: DeletionVector = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.len(), 3);
        assert!(back.contains(GlobalId::new(4_000_000_000)));
    }
}
