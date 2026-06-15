//! The commit DAG: versions, branches, tags, diff, and GC.
//!
//! [`VersionStore`] is the in-memory registry of [`Manifest`]s plus the movable
//! pointers over them (`HEAD`, branches, tags). It owns no segments — those live in
//! the collection and are referenced by id — so it is pure metadata: commits are
//! cheap, diffs are set operations, and GC is reachability over retained manifests.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::deletion::DeletionVector;
use super::manifest::{Manifest, SegmentRef};

/// A serializable snapshot of a [`VersionStore`] for durable persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionStoreSnapshot {
    /// All committed manifests.
    pub manifests: Vec<Manifest>,
    /// The current `HEAD`.
    pub head: Option<u64>,
    /// Branch pointers.
    pub branches: HashMap<String, u64>,
    /// Tag pointers.
    pub tags: HashMap<String, u64>,
    /// The next version id to allocate.
    pub next_version: u64,
}

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

    /// A serializable snapshot of the whole store.
    pub fn snapshot(&self) -> VersionStoreSnapshot {
        VersionStoreSnapshot {
            manifests: self.commits.values().map(|m| (**m).clone()).collect(),
            head: self.head,
            branches: self.branches.clone(),
            tags: self.tags.clone(),
            next_version: self.next_version,
        }
    }

    /// Rebuilds a store from a snapshot (recovery).
    pub fn from_snapshot(snapshot: VersionStoreSnapshot) -> Self {
        let commits = snapshot
            .manifests
            .into_iter()
            .map(|m| (m.version, Arc::new(m)))
            .collect();
        Self {
            commits,
            head: snapshot.head,
            branches: snapshot.branches,
            tags: snapshot.tags,
            next_version: snapshot.next_version,
        }
    }

    /// All segment ids referenced by any retained version.
    pub fn all_referenced_segments(&self) -> BTreeSet<u64> {
        let mut ids = BTreeSet::new();
        for m in self.commits.values() {
            ids.extend(m.segment_ids());
        }
        ids
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::GlobalId;

    fn seg(id: u64, lo: u64, hi: u64) -> SegmentRef {
        SegmentRef {
            id,
            id_lo: lo,
            id_hi: hi,
            count: hi - lo + 1,
        }
    }

    fn dv(deleted: &[u64]) -> DeletionVector {
        let mut d = DeletionVector::default();
        for &id in deleted {
            d.insert(GlobalId::new(id));
        }
        d
    }

    fn commit(s: &mut VersionStore, segs: Vec<SegmentRef>, deleted: &[u64]) -> u64 {
        s.commit("test", None, None, segs, dv(deleted), 0).version
    }

    #[test]
    fn commit_links_parent_advances_head_and_next_version() {
        let mut s = VersionStore::new();
        assert_eq!(s.head(), None);
        let m0 = s.commit("manual", None, None, vec![seg(0, 0, 9)], dv(&[]), 0);
        assert_eq!(m0.version, 0);
        assert_eq!(m0.parent, None);
        assert_eq!(s.head(), Some(0));
        let m1 = s.commit("manual", None, None, vec![seg(1, 10, 19)], dv(&[]), 0);
        assert_eq!(m1.version, 1);
        assert_eq!(m1.parent, Some(0)); // parent-linked to previous head
        assert_eq!(s.head(), Some(1));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn resolve_covers_version_tag_branch_and_head() {
        let mut s = VersionStore::new();
        assert_eq!(s.resolve(&VersionSelector::Head), None); // empty store
        let v0 = commit(&mut s, vec![seg(0, 0, 9)], &[]);
        let v1 = commit(&mut s, vec![seg(1, 10, 19)], &[]);
        s.set_tag("rel", v0).unwrap();
        s.set_branch("dev", v0).unwrap();
        assert_eq!(s.resolve(&VersionSelector::Version(v0)), Some(v0));
        assert_eq!(s.resolve(&VersionSelector::Version(999)), None);
        assert_eq!(s.resolve(&VersionSelector::Tag("rel".into())), Some(v0));
        assert_eq!(s.resolve(&VersionSelector::Tag("nope".into())), None);
        assert_eq!(s.resolve(&VersionSelector::Branch("dev".into())), Some(v0));
        assert_eq!(s.resolve(&VersionSelector::Branch("nope".into())), None);
        assert_eq!(s.resolve(&VersionSelector::Head), Some(v1)); // head == last commit
    }

    #[test]
    fn set_tag_branch_and_diff_reject_unknown_version() {
        let mut s = VersionStore::new();
        commit(&mut s, vec![seg(0, 0, 9)], &[]);
        assert!(matches!(s.set_tag("t", 999), Err(VersionError::NoSuchVersion(999))));
        assert!(matches!(s.set_branch("b", 999), Err(VersionError::NoSuchVersion(999))));
        assert!(matches!(s.diff(0, 999), Err(VersionError::NoSuchVersion(999))));
        assert!(matches!(s.diff(999, 0), Err(VersionError::NoSuchVersion(999))));
    }

    #[test]
    fn moving_tag_and_branch_repoints_without_duplicating() {
        let mut s = VersionStore::new();
        let v0 = commit(&mut s, vec![seg(0, 0, 9)], &[]);
        let v1 = commit(&mut s, vec![seg(1, 10, 19)], &[]);
        s.set_branch("main", v0).unwrap();
        assert_eq!(s.resolve(&VersionSelector::Branch("main".into())), Some(v0));
        s.set_branch("main", v1).unwrap(); // move, not create
        assert_eq!(s.resolve(&VersionSelector::Branch("main".into())), Some(v1));
        assert_eq!(s.branches().len(), 1);
        s.set_tag("t", v0).unwrap();
        s.set_tag("t", v1).unwrap();
        assert_eq!(s.resolve(&VersionSelector::Tag("t".into())), Some(v1));
        assert_eq!(s.tags().len(), 1);
    }

    #[test]
    fn gc_retention_guards_protect_tagged_and_branch_head_versions() {
        // The data-loss safety property: a tagged or branch-pointed version older than
        // the keep_last window MUST survive GC, and its segments must NOT be orphaned.
        let mut s = VersionStore::new();
        let v0 = commit(&mut s, vec![seg(0, 0, 9)], &[]);
        let v1 = commit(&mut s, vec![seg(1, 10, 19)], &[]);
        let v2 = commit(&mut s, vec![seg(2, 20, 29)], &[]); // head
        s.set_tag("release", v0).unwrap();
        s.set_branch("b", v1).unwrap();

        // keep_last=1 alone would drop v0 and v1 — the tag/branch/head guards save them.
        let guarded = RetentionRules {
            keep_last: Some(1),
            keep_tagged: true,
            keep_branch_heads: true,
        };
        let r = s.plan_gc(&guarded);
        assert!(r.removed_versions.is_empty(), "tag/branch/head must protect v0,v1,v2");
        assert!(r.orphan_segments.is_empty());

        // Disable the guards: v0 (tag) and v1 (branch) drop; v2 stays (head + keep_last).
        let unguarded = RetentionRules {
            keep_last: Some(1),
            keep_tagged: false,
            keep_branch_heads: false,
        };
        let r2 = s.plan_gc(&unguarded);
        assert_eq!(r2.removed_versions, vec![v0, v1]);
        assert_eq!(r2.orphan_segments, vec![0, 1]); // their now-unreferenced segments
        let _ = v2;
    }

    #[test]
    fn snapshot_round_trip_preserves_tags_branches_head_next_version() {
        let mut s = VersionStore::new();
        let v0 = s
            .commit("m", Some("first".into()), Some("autotag".into()), vec![seg(0, 0, 9)], dv(&[]), 0)
            .version;
        let v1 = commit(&mut s, vec![seg(1, 10, 19)], &[]);
        s.set_branch("dev", v0).unwrap();
        s.set_tag("rel", v1).unwrap();

        // Round-trip through the real serde path (catches field drift / missing maps).
        let bytes = rmp_serde::to_vec(&s.snapshot()).unwrap();
        let snap2: VersionStoreSnapshot = rmp_serde::from_slice(&bytes).unwrap();
        let mut s2 = VersionStore::from_snapshot(snap2);

        assert_eq!(s2.len(), 2);
        assert_eq!(s2.head(), Some(v1));
        assert_eq!(s2.resolve(&VersionSelector::Branch("dev".into())), Some(v0));
        assert_eq!(s2.resolve(&VersionSelector::Tag("rel".into())), Some(v1));
        assert_eq!(s2.resolve(&VersionSelector::Tag("autotag".into())), Some(v0));
        // next_version restored: a new commit must not reuse an existing id.
        let v2 = commit(&mut s2, vec![seg(2, 20, 29)], &[]);
        assert_eq!(v2, 2);
    }

    #[test]
    fn diff_reports_added_and_removed_accounting_for_deletions() {
        let mut s = VersionStore::new();
        let v0 = commit(&mut s, vec![seg(0, 0, 4)], &[]); // live {0,1,2,3,4}
        // v1: same range with 2 deleted, plus a new segment for 5,6.
        let v1 = commit(&mut s, vec![seg(0, 0, 4), seg(1, 5, 6)], &[2]); // live {0,1,3,4,5,6}
        let d = s.diff(v0, v1).unwrap();
        assert_eq!(d.added, vec![5, 6]);
        assert_eq!(d.removed, vec![2]);
    }
}
