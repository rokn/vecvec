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

#[cfg(test)]
mod tests {
    use super::*;
    use vecvec_core::{CollectionConfig, FsyncMode, Metric};

    fn dc(dir: &std::path::Path) -> Arc<DurableCollection> {
        let cfg = CollectionConfig::new("c", 4, Metric::Dot);
        Arc::new(DurableCollection::open(dir, cfg, FsyncMode::Async).unwrap())
    }

    #[test]
    fn insert_collision_get_remove_and_listing() {
        let d1 = tempfile::tempdir().unwrap();
        let d2 = tempfile::tempdir().unwrap();
        let r = Registry::new();
        assert!(r.is_empty());

        assert!(r.insert_new("a".into(), dc(d1.path())));
        // Same name is rejected (not overwritten).
        assert!(!r.insert_new("a".into(), dc(d2.path())));
        assert!(r.insert_new("b".into(), dc(d2.path())));
        assert_eq!(r.len(), 2);

        assert!(r.get("a").is_some());
        assert!(r.get("missing").is_none());

        let mut names: Vec<String> = r.list_all().into_iter().map(|(n, _)| n).collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(r.snapshot().len(), 2);

        assert!(r.remove("a").is_some());
        assert!(r.remove("a").is_none()); // already gone
        assert_eq!(r.len(), 1);
    }
}
