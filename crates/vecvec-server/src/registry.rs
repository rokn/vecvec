//! The collection registry.
//!
//! Maps collection names to live [`Collection`]s with lock-light concurrent access
//! via [`DashMap`]. The full per-collection durability/versioning wiring layers on
//! in later milestones; at M3 it just owns the in-RAM collections.

use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use vecvec_core::{Collection, CollectionConfig};

/// Returned when creating a collection whose name is already taken.
#[derive(Debug)]
pub struct AlreadyExists(pub String);

/// A concurrent map of collection name → collection.
#[derive(Default)]
pub struct Registry {
    collections: DashMap<String, Arc<Collection>>,
}

impl Registry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a collection, failing if the name already exists.
    pub fn create(&self, config: CollectionConfig) -> Result<Arc<Collection>, AlreadyExists> {
        let name = config.name.clone();
        match self.collections.entry(name.clone()) {
            Entry::Occupied(_) => Err(AlreadyExists(name)),
            Entry::Vacant(slot) => {
                let collection = Arc::new(Collection::create(config));
                slot.insert(collection.clone());
                Ok(collection)
            }
        }
    }

    /// Looks up a collection by name.
    pub fn get(&self, name: &str) -> Option<Arc<Collection>> {
        self.collections.get(name).map(|r| r.value().clone())
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
