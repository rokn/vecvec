//! Local ↔ global id mapping within a segment.
//!
//! Each segment numbers its rows with dense, segment-local [`LocalId`]s (`0..len`),
//! which is what the index and vector storage use. The collection assigns every
//! point a stable, monotonic [`GlobalId`] that survives seal/merge and is the unit
//! of the versioning deletion vectors. [`IdMap`] is the per-segment bridge between
//! the two.

use std::collections::HashMap;

use crate::id::{GlobalId, LocalId};

/// A bijection between a segment's local row ids and collection-global ids.
#[derive(Debug, Default, Clone)]
pub struct IdMap {
    local_to_global: Vec<GlobalId>,
    global_to_local: HashMap<GlobalId, LocalId>,
}

impl IdMap {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty map with room for `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            local_to_global: Vec::with_capacity(capacity),
            global_to_local: HashMap::with_capacity(capacity),
        }
    }

    /// Appends `global`, assigning it the next local id.
    ///
    /// # Panics
    /// Panics in debug builds if `global` is already mapped.
    pub fn push(&mut self, global: GlobalId) -> LocalId {
        let local = LocalId::new(self.local_to_global.len() as u32);
        self.local_to_global.push(global);
        let prev = self.global_to_local.insert(global, local);
        debug_assert!(prev.is_none(), "global id {global} inserted twice");
        local
    }

    /// The global id for a local id, or `None` if out of range.
    #[inline]
    pub fn to_global(&self, local: LocalId) -> Option<GlobalId> {
        self.local_to_global.get(local.get() as usize).copied()
    }

    /// The global id for a local id known to be in range.
    ///
    /// # Panics
    /// Panics if `local` is out of range.
    #[inline]
    pub fn global_at(&self, local: LocalId) -> GlobalId {
        self.local_to_global[local.get() as usize]
    }

    /// The local id for a global id, or `None` if not present.
    #[inline]
    pub fn to_local(&self, global: GlobalId) -> Option<LocalId> {
        self.global_to_local.get(&global).copied()
    }

    /// The local→global mapping as a slice indexed by local id.
    #[inline]
    pub fn global_ids(&self) -> &[GlobalId] {
        &self.local_to_global
    }

    /// Rebuilds an id map from a local→global list (e.g. when loading a segment).
    pub fn from_global_ids(global_ids: Vec<GlobalId>) -> Self {
        let global_to_local = global_ids
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, LocalId::new(i as u32)))
            .collect();
        Self {
            local_to_global: global_ids,
            global_to_local,
        }
    }

    /// The number of mapped points.
    #[inline]
    pub fn len(&self) -> usize {
        self.local_to_global.len()
    }

    /// Whether the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.local_to_global.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_assigns_dense_locals_and_is_bijective() {
        let mut m = IdMap::new();
        let l0 = m.push(GlobalId::new(100));
        let l1 = m.push(GlobalId::new(250));
        assert_eq!(l0, LocalId::new(0));
        assert_eq!(l1, LocalId::new(1));
        assert_eq!(m.len(), 2);
        assert_eq!(m.global_at(l1), GlobalId::new(250));
        assert_eq!(m.to_global(LocalId::new(0)), Some(GlobalId::new(100)));
        assert_eq!(m.to_local(GlobalId::new(250)), Some(l1));
        assert_eq!(m.to_local(GlobalId::new(999)), None);
        assert_eq!(m.to_global(LocalId::new(5)), None);
    }
}
