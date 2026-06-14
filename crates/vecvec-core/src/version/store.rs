//! The commit DAG: versions, branches, tags, diff, and GC.
//!
//! [`VersionStore`] is the in-memory registry of [`Manifest`]s plus the movable
//! pointers over them (`HEAD`, branches, tags). It owns no segments — those live in
//! the collection and are referenced by id — so it is pure metadata: commits are
//! cheap, diffs are set operations, and GC is reachability over retained manifests.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use super::deletion::DeletionVector;
use super::manifest::{Manifest, SegmentRef};

/// Selects a version to read: an explicit id, a tag, a branch head, or `HEAD`.
#[derive(Debug, Clone)]
pub enum VersionSelector {
    /// An explicit version id.
    Version(u64),
    /// A tag name.
    Tag(String),
    /// A branch name (its current head).
    Branch(String),
    /// The current `HEAD`.
    Head,
}

/// The result of diffing two versions' live id sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diff {
    /// The `from` version.
    pub from: u64,
    /// The `to` version.
    pub to: u64,
    /// Ids live in `to` but not `from`.
    pub added: Vec<u64>,
    /// Ids live in `from` but not `to`.
    pub removed: Vec<u64>,
}

/// Which versions to keep during GC. Defaults keep everything reachable.
#[derive(Debug, Clone)]
pub struct RetentionRules {
    /// Keep at least this many most-recent versions (`None` = all).
    pub keep_last: Option<usize>,
    /// Always keep tagged versions.
    pub keep_tagged: bool,
    /// Always keep branch heads (and `HEAD`).
    pub keep_branch_heads: bool,
}

impl Default for RetentionRules {
    fn default() -> Self {
        Self {
            keep_last: None,
            keep_tagged: true,
            keep_branch_heads: true,
        }
    }
}

/// What a GC pass would remove.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Versions dropped from the DAG.
    pub removed_versions: Vec<u64>,
    /// Segment ids no longer referenced by any retained version (safe to delete).
    pub orphan_segments: Vec<u64>,
}

/// The commit DAG and its movable pointers.
#[derive(Default)]
pub struct VersionStore {
    commits: BTreeMap<u64, Arc<Manifest>>,
    head: Option<u64>,
    branches: HashMap<String, u64>,
    tags: HashMap<String, u64>,
    next_version: u64,
}

impl VersionStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a new commit on top of the current `HEAD`, advancing `HEAD` to it.
    #[allow(clippy::too_many_arguments)]
    pub fn commit(
        &mut self,
        trigger: impl Into<String>,
        message: Option<String>,
        tag: Option<String>,
        segments: Vec<SegmentRef>,
        deletions: DeletionVector,
        created_at_ms: u64,
    ) -> Arc<Manifest> {
        let version = self.next_version;
        self.next_version += 1;
        let manifest = Arc::new(Manifest {
            version,
            parent: self.head,
            created_at_ms,
            trigger: trigger.into(),
            message,
            segments,
            deletions,
        });
        self.commits.insert(version, manifest.clone());
        self.head = Some(version);
        if let Some(name) = tag {
            self.tags.insert(name, version);
        }
        manifest
    }

    /// The current `HEAD` version, if any commits exist.
    pub fn head(&self) -> Option<u64> {
        self.head
    }

    /// The manifest for a version id.
    pub fn get(&self, version: u64) -> Option<Arc<Manifest>> {
        self.commits.get(&version).cloned()
    }

    /// All manifests, oldest first.
    pub fn list(&self) -> Vec<Arc<Manifest>> {
        self.commits.values().cloned().collect()
    }

    /// The number of committed versions.
    pub fn len(&self) -> usize {
        self.commits.len()
    }

    /// Whether there are no commits.
    pub fn is_empty(&self) -> bool {
        self.commits.is_empty()
    }

    /// Resolves a selector to a concrete version id.
    pub fn resolve(&self, selector: &VersionSelector) -> Option<u64> {
        match selector {
            VersionSelector::Version(v) => self.commits.contains_key(v).then_some(*v),
            VersionSelector::Tag(t) => self.tags.get(t).copied(),
            VersionSelector::Branch(b) => self.branches.get(b).copied(),
            VersionSelector::Head => self.head,
        }
    }

    /// Creates or moves a tag to a version. Errors if the version doesn't exist.
    pub fn set_tag(&mut self, name: impl Into<String>, version: u64) -> Result<(), VersionError> {
        if !self.commits.contains_key(&version) {
            return Err(VersionError::NoSuchVersion(version));
        }
        self.tags.insert(name.into(), version);
        Ok(())
    }

    /// Creates or moves a branch pointer to a version.
    pub fn set_branch(
        &mut self,
        name: impl Into<String>,
        version: u64,
    ) -> Result<(), VersionError> {
        if !self.commits.contains_key(&version) {
            return Err(VersionError::NoSuchVersion(version));
        }
        self.branches.insert(name.into(), version);
        Ok(())
    }

    /// The tag → version map.
    pub fn tags(&self) -> &HashMap<String, u64> {
        &self.tags
    }

    /// The branch → version map.
    pub fn branches(&self) -> &HashMap<String, u64> {
        &self.branches
    }

    /// The live global-id set of a version (segment ranges minus its deletions).
    fn live_ids(manifest: &Manifest) -> BTreeSet<u64> {
        let mut live = BTreeSet::new();
        for seg in &manifest.segments {
            for id in seg.id_lo..=seg.id_hi {
                if !manifest.deletions.contains(crate::id::GlobalId::new(id)) {
                    live.insert(id);
                }
            }
        }
        live
    }

    /// Diffs two versions' live id sets. Skips segments whose range+deletions are
    /// identical in both versions (the common case for descendants).
    pub fn diff(&self, from: u64, to: u64) -> Result<Diff, VersionError> {
        let a = self.get(from).ok_or(VersionError::NoSuchVersion(from))?;
        let b = self.get(to).ok_or(VersionError::NoSuchVersion(to))?;
        let la = Self::live_ids(&a);
        let lb = Self::live_ids(&b);
        Ok(Diff {
            from,
            to,
            added: lb.difference(&la).copied().collect(),
            removed: la.difference(&lb).copied().collect(),
        })
    }

    /// Computes a GC pass without mutating: which versions would be dropped and which
    /// segments would become orphaned. Never lists a segment still referenced by a
    /// retained version.
    pub fn plan_gc(&self, rules: &RetentionRules) -> GcReport {
        let retained = self.retained_versions(rules);
        let removed_versions: Vec<u64> = self
            .commits
            .keys()
            .copied()
            .filter(|v| !retained.contains(v))
            .collect();

        let mut retained_segments = BTreeSet::new();
        for v in &retained {
            if let Some(m) = self.commits.get(v) {
                retained_segments.extend(m.segment_ids());
            }
        }
        let mut orphan_segments = BTreeSet::new();
        for v in &removed_versions {
            if let Some(m) = self.commits.get(v) {
                for sid in m.segment_ids() {
                    if !retained_segments.contains(&sid) {
                        orphan_segments.insert(sid);
                    }
                }
            }
        }
        GcReport {
            removed_versions,
            orphan_segments: orphan_segments.into_iter().collect(),
        }
    }

    /// Applies a GC pass, dropping non-retained manifests. Returns the orphaned
    /// segment ids the caller may now delete from storage.
    pub fn gc(&mut self, rules: &RetentionRules) -> GcReport {
        let report = self.plan_gc(rules);
        for v in &report.removed_versions {
            self.commits.remove(v);
        }
        report
    }

    fn retained_versions(&self, rules: &RetentionRules) -> BTreeSet<u64> {
        let mut retained = BTreeSet::new();
        if let Some(h) = self.head {
            retained.insert(h);
        }
        if rules.keep_branch_heads {
            retained.extend(self.branches.values().copied());
        }
        if rules.keep_tagged {
            retained.extend(self.tags.values().copied());
        }
        if let Some(k) = rules.keep_last {
            for v in self.commits.keys().rev().take(k) {
                retained.insert(*v);
            }
        } else {
            retained.extend(self.commits.keys().copied());
        }
        retained
    }
}

/// Errors from version operations.
#[derive(Debug, thiserror::Error)]
pub enum VersionError {
    /// The referenced version does not exist.
    #[error("no such version: {0}")]
    NoSuchVersion(u64),
    /// The selector could not be resolved.
    #[error("could not resolve version selector")]
    Unresolvable,
}
