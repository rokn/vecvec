//! The collection registry.
//!
//! Maps collection names to live, durable collections with lock-light concurrent
//! access via [`DashMap`].

use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use vecvec_core::DurableCollection;

/// A concurrent map of collection name → durable collection.
#[derive(Default)]
pub struct Registry {
    collections: DashMap<String, Arc<DurableCollection>>,
}

impl Registry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up a collection by name.
    pub fn get(&self, name: &str) -> Option<Arc<DurableCollection>> {
        self.collections.get(name).map(|r| r.value().clone())
    }

    /// Inserts a collection if the name is free; returns `false` if it was taken.
    pub fn insert_new(&self, name: String, collection: Arc<DurableCollection>) -> bool {
        match self.collections.entry(name) {
            Entry::Occupied(_) => false,
            Entry::Vacant(slot) => {
                slot.insert(collection);
                true
            }
        }
    }

    /// A snapshot of all collections (for background maintenance).
    pub fn snapshot(&self) -> Vec<Arc<DurableCollection>> {
        self.collections.iter().map(|e| e.value().clone()).collect()
    }

    /// A snapshot of all `(name, collection)` pairs (for listing in the explorer UI).
    pub fn list_all(&self) -> Vec<(String, Arc<DurableCollection>)> {
        self.collections
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect()
    }

    /// Removes a collection from the registry, returning it if it was present.
    pub fn remove(&self, name: &str) -> Option<Arc<DurableCollection>> {
        self.collections.remove(name).map(|(_, v)| v)
    }

    /// The number of collections.
    pub fn len(&self) -> usize {
        self.collections.len()
    }

    /// Whether there are no collections.
    pub fn is_empty(&self) -> bool {
        self.collections.is_empty()
    }
}
