//! Durable collections: WAL-first writes + crash recovery + checkpointing.
//!
//! A [`DurableCollection`] wraps an in-memory [`Collection`] with a write-ahead log.
//! The contract: a mutation is appended to the WAL and fsynced **before** it is
//! acked and applied in memory, so any acked write survives a crash by being
//! replayed on the next open.
//!
//! Checkpointing keeps the WAL short and recovery fast: it seals the appendable into
//! sealed segments, persists them durably, then **switches to a fresh WAL
//! generation** referenced by an atomically-written `HEAD`. The ordering — segments
//! durable → HEAD swapped → old WAL retired — is what makes it crash-consistent:
//! a crash before `HEAD` leaves the old generation valid (the checkpoint simply
//! didn't commit); a crash after leaves the new (empty) generation, so folded ops
//! are never double-applied.
//!
//! Writes fsync on the calling thread; the server runs them on its rayon pool
//! ([`BlockingBridge`](../../vecvec_server/blocking/struct.BlockingBridge.html)), so
//! durability never blocks the async reactor. A dedicated I/O actor + group commit
//! is a later optimization.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::collection::{Collection, CollectionConfig, ScoredGlobal};
use crate::error::{CoreError, Result};
use crate::id::{GlobalId, SegmentId};
use crate::payload::{Filter, Payload, PayloadMap};
use crate::persist::atomic::{FileKind, read_framed, write_atomic};
use crate::persist::wal::{Wal, WalOp};
use crate::segment::SegmentStore;
use crate::version::DeletionVector;

const HEAD_FORMAT_VERSION: u32 = 1;
const CONFIG_FILE: &str = "config";

/// Reads the persisted [`CollectionConfig`] from a collection directory (used to
/// rediscover and reopen collections on server startup).
pub fn read_config(dir: impl AsRef<Path>) -> Result<CollectionConfig> {
    let path = dir.as_ref().join(CONFIG_FILE);
    let frame = read_framed(&path)?;
    rmp_serde::from_slice(&frame.payload).map_err(|e| CoreError::Serialization {
        detail: e.to_string(),
    })
}

fn write_config_if_absent(dir: &Path, config: &CollectionConfig) -> Result<()> {
    let path = dir.join(CONFIG_FILE);
    if !path.exists() {
        let bytes = rmp_serde::to_vec(config).map_err(|e| CoreError::Serialization {
            detail: e.to_string(),
        })?;
        write_atomic(&path, FileKind::Generic, 1, &bytes)?;
    }
    Ok(())
}

/// When to fsync the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsyncMode {
    /// fsync each write batch before acking (durable; the default).
    Sync,
    /// Don't fsync (faster; relies on the OS — acked writes may be lost on power
    /// loss, but the log is never corrupted). For ephemeral/bench use.
    Async,
}

const VERSIONS_FILE: &str = "versions";

/// The persisted pointer to the latest durable state.
#[derive(Debug, Serialize, Deserialize)]
struct CheckpointHead {
    wal_generation: u64,
    /// Segments forming the live working set as of the checkpoint.
    working_segment_ids: Vec<u64>,
    /// All segments to load on recovery (working set + everything any version refs).
    all_segment_ids: Vec<u64>,
    next_global_id: u64,
    next_segment_id: u64,
    #[serde(default)]
    deletions: crate::version::DeletionVector,
    #[serde(default)]
    payloads: PayloadMap,
}

struct WalState {
    wal: Wal,
    generation: u64,
}

/// A collection with durable, crash-recoverable storage.
pub struct DurableCollection {
    collection: Arc<Collection>,
    store: SegmentStore,
    dir: PathBuf,
    wal: Mutex<WalState>,
    fsync: FsyncMode,
    trigger: Mutex<crate::version::TriggerEvaluator>,
    clock: crate::version::SystemClock,
    /// Segment ids already written to disk (so commits don't rewrite them).
    persisted_segments: Mutex<std::collections::HashSet<SegmentId>>,
}

impl DurableCollection {
    /// Opens (and recovers, if data exists) a durable collection rooted at `dir`.
    pub fn open(dir: impl AsRef<Path>, config: CollectionConfig, fsync: FsyncMode) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| CoreError::io(&dir, e))?;
        write_config_if_absent(&dir, &config)?;
        let store = SegmentStore::new(dir.join("segments"));
        let policy = config.versioning;
        let collection = Arc::new(Collection::create(config));
        let clock = crate::version::SystemClock;

        // Recover the durable version DAG (if any), so time-travel survives restart.
        let versions_path = dir.join(VERSIONS_FILE);
        let mut referenced: std::collections::BTreeSet<u64> = Default::default();
        if versions_path.exists() {
            let frame = read_framed(&versions_path)?;
            let snapshot: crate::version::VersionStoreSnapshot =
                rmp_serde::from_slice(&frame.payload).map_err(|e| CoreError::Serialization {
                    detail: e.to_string(),
                })?;
            for m in &snapshot.manifests {
                referenced.extend(m.segment_ids());
            }
            collection.load_version_snapshot(snapshot);
        }

        // Recover the checkpoint, if any.
        let head_path = dir.join("HEAD");
        let mut persisted = std::collections::HashSet::new();
        let generation = if head_path.exists() {
            let frame = read_framed(&head_path)?;
            let head: CheckpointHead =
                rmp_serde::from_slice(&frame.payload).map_err(|e| CoreError::Serialization {
                    detail: e.to_string(),
                })?;
            // Load the working set + every segment referenced by any version.
            let mut all_ids: std::collections::BTreeSet<u64> =
                head.all_segment_ids.iter().copied().collect();
            all_ids.extend(head.working_segment_ids.iter().copied());
            all_ids.extend(referenced.iter().copied());
            let mut segments = Vec::with_capacity(all_ids.len());
            for id in &all_ids {
                segments.push(store.load(SegmentId::new(*id))?);
                persisted.insert(SegmentId::new(*id));
            }
            collection.install_recovered(segments, &head.working_segment_ids);
            collection.set_allocators(head.next_global_id, head.next_segment_id);
            collection.set_deletions(head.deletions);
            collection.set_payloads(head.payloads);
            head.wal_generation
        } else {
            // No checkpoint, but versions may still reference persisted segments.
            let mut segments = Vec::new();
            for id in &referenced {
                segments.push(store.load(SegmentId::new(*id))?);
                persisted.insert(SegmentId::new(*id));
            }
            collection.install_recovered(segments, &[]);
            0
        };

        // Replay the active WAL generation on top of the checkpoint.
        let wal = Wal::open(Self::wal_path(&dir, generation))?;
        for op in wal.read_all()? {
            apply_op(&collection, &op);
        }

        Ok(Self {
            collection,
            store,
            dir,
            wal: Mutex::new(WalState { wal, generation }),
            fsync,
            trigger: Mutex::new(crate::version::TriggerEvaluator::new(policy, &clock)),
            clock,
            persisted_segments: Mutex::new(persisted),
        })
    }

    /// Durably persists the version DAG and any segments it (or the working set)
    /// references but that aren't yet on disk. Called after every commit so versions
    /// survive a crash.
    fn persist_versions(&self) -> Result<()> {
        for id in self.collection.segment_ids_to_persist() {
            if self.persisted_segments.lock().contains(&id) {
                continue;
            }
            if let Some(seg) = self.collection.get_segment(id) {
                self.store.write_sealed(&seg)?;
                self.persisted_segments.lock().insert(id);
            }
        }
        let snapshot = self.collection.version_snapshot();
        let bytes = rmp_serde::to_vec(&snapshot).map_err(|e| CoreError::Serialization {
            detail: e.to_string(),
        })?;
        write_atomic(&self.dir.join(VERSIONS_FILE), FileKind::Manifest, 1, &bytes)
    }

    fn wal_path(dir: &Path, generation: u64) -> PathBuf {
        dir.join(format!("wal.{generation}.log"))
    }

    /// The wrapped in-memory collection (for read paths run on a compute pool).
    pub fn collection(&self) -> &Arc<Collection> {
        &self.collection
    }

    /// The collection config.
    pub fn config(&self) -> &CollectionConfig {
        self.collection.config()
    }

    /// The number of live points.
    pub fn len(&self) -> usize {
        self.collection.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.collection.is_empty()
    }

    /// A page of live points (vectors + payloads), optionally as of a past version.
    /// Returns `(page, total_live_count)`. Backs the explorer table + 2D graph view.
    pub fn scroll(
        &self,
        at: Option<&crate::version::VersionSelector>,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<crate::collection::PointRecord>, usize)> {
        self.collection.scroll(at, offset, limit)
    }

    /// A single live point by id (vector + payload), or `None` if missing/deleted.
    pub fn get_point(&self, id: u64) -> Option<crate::collection::PointRecord> {
        self.collection.get_point(GlobalId::new(id))
    }

    /// The current HEAD version, if any.
    pub fn head_version(&self) -> Option<u64> {
        self.collection.head_version()
    }

    /// Durably appends a batch of `(vector, payload)` points, returning their ids.
    /// WAL-first: logged + fsynced before applied in memory.
    pub fn upsert(&self, points: Vec<(Vec<f32>, Option<Payload>)>) -> Result<Vec<u64>> {
        let dim = self.collection.config().dim;
        for (v, _) in &points {
            if v.len() != dim {
                return Err(CoreError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let mut guard = self.wal.lock();
        let mut ids = Vec::with_capacity(points.len());
        for (v, payload) in &points {
            let id = self.collection.alloc_global_id();
            guard.wal.append(&WalOp::Upsert {
                id: id.get(),
                vector: v.clone(),
                payload: payload.clone(),
            })?;
            ids.push(id.get());
        }
        if self.fsync == FsyncMode::Sync {
            guard.wal.flush()?;
        }
        // Durable -> apply in memory (the same path recovery uses).
        for (id, (v, payload)) in ids.iter().zip(points) {
            self.collection
                .insert_with_id_and_payload(GlobalId::new(*id), &v, payload)?;
        }
        drop(guard);

        // Auto-commit if the versioning policy's trigger has fired.
        let committed = {
            let mut trigger = self.trigger.lock();
            trigger.record_writes(ids.len() as u64);
            if trigger.should_commit(&self.clock) {
                self.collection.commit("auto", None, None)?;
                trigger.note_commit(&self.clock);
                true
            } else {
                false
            }
        };
        if committed {
            self.persist_versions()?;
        }
        Ok(ids)
    }

    /// Explicitly commits the working state as a new version.
    pub fn commit(&self, message: Option<String>, tag: Option<String>) -> Result<u64> {
        let version = self.collection.commit("manual", message, tag)?;
        self.persist_versions()?;
        Ok(version)
    }

    /// Time-travel search as of a version/tag/branch.
    pub fn search_at(
        &self,
        selector: &crate::version::VersionSelector,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredGlobal>> {
        self.collection.search_at(selector, query, k, filter)
    }

    /// Recommend-by-example: build a query from positive/negative example ids.
    pub fn recommend(
        &self,
        positive: &[GlobalId],
        negative: &[GlobalId],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredGlobal>> {
        self.collection.recommend(positive, negative, k, filter)
    }

    /// Diffs two versions.
    pub fn diff(&self, from: u64, to: u64) -> Result<crate::version::Diff> {
        self.collection.diff(from, to)
    }

    /// Restores the working state to a version (a forward commit).
    pub fn restore(&self, version: u64) -> Result<u64> {
        let new_version = self.collection.restore(version)?;
        self.persist_versions()?;
        Ok(new_version)
    }

    /// Tags a version.
    pub fn create_tag(&self, name: impl Into<String>, version: u64) -> Result<()> {
        self.collection.create_tag(name, version)?;
        self.persist_versions()
    }

    /// Branches from a version.
    pub fn create_branch(&self, name: impl Into<String>, version: u64) -> Result<()> {
        self.collection.create_branch(name, version)?;
        self.persist_versions()
    }

    /// All committed versions, oldest first.
    pub fn list_versions(&self) -> Vec<Arc<crate::version::Manifest>> {
        self.collection.list_versions()
    }

    /// Merges working segments to cut fan-out, then checkpoints to persist the new
    /// working set. Returns the merged segment id (if any).
    pub fn compact(&self) -> Result<Option<u64>> {
        let merged = self.collection.compact().map(|id| id.get());
        if merged.is_some() {
            self.checkpoint()?;
        }
        Ok(merged)
    }

    /// Exports the collection to a tar archive for backup / migration. Checkpoints
    /// first so the archive is a self-contained, WAL-folded snapshot.
    pub fn export(&self, out_path: impl AsRef<Path>) -> Result<()> {
        self.checkpoint()?;
        let out = out_path.as_ref();
        let file = std::fs::File::create(out).map_err(|e| CoreError::io(out, e))?;
        let mut builder = tar::Builder::new(file);
        builder
            .append_dir_all(".", &self.dir)
            .map_err(|e| CoreError::io(&self.dir, e))?;
        builder.finish().map_err(|e| CoreError::io(out, e))?;
        Ok(())
    }

    /// Runs a GC pass with the given retention, deleting orphaned segment files.
    pub fn gc(&self, retention: &crate::version::RetentionRules) -> Result<Vec<u64>> {
        let dropped = self.collection.gc(retention);
        for id in &dropped {
            self.store.remove(*id);
            self.persisted_segments.lock().remove(id);
        }
        self.persist_versions()?;
        Ok(dropped.iter().map(|s| s.get()).collect())
    }

    /// Durably tombstones a point. Returns whether it was newly deleted.
    pub fn delete(&self, id: u64) -> Result<bool> {
        let mut guard = self.wal.lock();
        guard.wal.append(&WalOp::Delete { id })?;
        if self.fsync == FsyncMode::Sync {
            guard.wal.flush()?;
        }
        Ok(self.collection.delete(GlobalId::new(id)))
    }

    /// Searches the collection (delegates to the in-memory engine).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredGlobal>> {
        self.collection.search(query, k, filter)
    }

    /// Folds the WAL into durable sealed segments and starts a fresh WAL generation.
    ///
    /// Ordering (crash-consistent): seal in memory → persist segments durably →
    /// write the new `HEAD` atomically → retire the old WAL.
    pub fn checkpoint(&self) -> Result<()> {
        let mut guard = self.wal.lock();

        // 1. Seal the appendable into the sealed set (in memory).
        self.collection.seal();

        // 2. Persist every segment the working set OR any version references.
        let working_segment_ids = self.collection.working_segment_ids();
        let mut all_segment_ids = Vec::new();
        for id in self.collection.segment_ids_to_persist() {
            if let Some(seg) = self.collection.get_segment(id) {
                self.store.write_sealed(&seg)?;
                self.persisted_segments.lock().insert(id);
                all_segment_ids.push(id.get());
            }
        }

        // 3. Create the next (empty) WAL generation, then commit HEAD atomically.
        let new_generation = guard.generation + 1;
        let new_wal = Wal::open(Self::wal_path(&self.dir, new_generation))?;
        let head = CheckpointHead {
            wal_generation: new_generation,
            working_segment_ids,
            all_segment_ids,
            next_global_id: self.collection.next_global_id_value(),
            next_segment_id: self.collection.next_segment_id_value(),
            deletions: DeletionVector::clone(&self.collection.deletions_snapshot()),
            payloads: self.collection.payloads_snapshot(),
        };
        let head_bytes = rmp_serde::to_vec(&head).map_err(|e| CoreError::Serialization {
            detail: e.to_string(),
        })?;
        write_atomic(
            &self.dir.join("HEAD"),
            FileKind::Head,
            HEAD_FORMAT_VERSION,
            &head_bytes,
        )?;

        // 4. Switch to the new generation and retire the old WAL.
        let old_generation = guard.generation;
        guard.wal = new_wal;
        guard.generation = new_generation;
        let _ = std::fs::remove_file(Self::wal_path(&self.dir, old_generation));
        Ok(())
    }
}

/// Imports a collection from a tar archive (produced by
/// [`DurableCollection::export`]) into `dest_dir`. Open the result with
/// [`DurableCollection::open`].
pub fn import(tar_path: impl AsRef<Path>, dest_dir: impl AsRef<Path>) -> Result<()> {
    let tar_path = tar_path.as_ref();
    let dest = dest_dir.as_ref();
    std::fs::create_dir_all(dest).map_err(|e| CoreError::io(dest, e))?;
    let file = std::fs::File::open(tar_path).map_err(|e| CoreError::io(tar_path, e))?;
    tar::Archive::new(file)
        .unpack(dest)
        .map_err(|e| CoreError::io(dest, e))?;
    Ok(())
}

/// Applies a single op to the in-memory collection. The one apply path shared by
/// live writes and recovery.
fn apply_op(collection: &Collection, op: &WalOp) {
    match op {
        WalOp::Upsert {
            id,
            vector,
            payload,
        } => {
            // Recovery trusts the WAL; a dimension mismatch here would mean a
            // corrupt record that passed CRC, which we treat as non-fatal-skip.
            let _ =
                collection.insert_with_id_and_payload(GlobalId::new(*id), vector, payload.clone());
        }
        WalOp::Delete { id } => {
            collection.delete(GlobalId::new(*id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    fn vecf(dim: usize, i: usize) -> Vec<f32> {
        (0..dim)
            .map(|j| ((i * 7 + j * 3) % 100) as f32 / 50.0 - 1.0)
            .collect()
    }

    #[test]
    fn survives_crash_without_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = CollectionConfig::new("c", 8, Metric::Dot);
        let q = vecf(8, 3);
        let first_id;
        {
            let dc = DurableCollection::open(dir.path(), cfg.clone(), FsyncMode::Sync).unwrap();
            let ids = dc
                .upsert((0..20).map(|i| (vecf(8, i), None)).collect())
                .unwrap();
            first_id = ids[3];
            assert_eq!(dc.len(), 20);
            // crash: drop without checkpoint
        }
        let dc = DurableCollection::open(dir.path(), cfg, FsyncMode::Sync).unwrap();
        assert_eq!(dc.len(), 20);
        // The point survived and is searchable after recovery (k == n returns all).
        let got = dc.search(&q, 20, None).unwrap();
        assert_eq!(got.len(), 20);
        assert!(got.iter().any(|s| s.id.get() == first_id));
    }

    #[test]
    fn versions_and_time_travel_survive_restart() {
        use crate::version::VersionSelector;
        let dir = tempfile::tempdir().unwrap();
        let cfg = CollectionConfig::new("c", 8, Metric::Cosine);
        let v1;
        {
            let dc = DurableCollection::open(dir.path(), cfg.clone(), FsyncMode::Sync).unwrap();
            dc.upsert((0..30).map(|i| (vecf(8, i), None)).collect())
                .unwrap();
            v1 = dc.commit(Some("first".into()), None).unwrap();
            // delete some after the commit, then add more
            dc.delete(2).unwrap();
            dc.delete(3).unwrap();
            assert_eq!(dc.len(), 28);
            // crash (drop) — note: NO checkpoint, only commit + WAL
        }
        // Reopen: versions (and their segments) recovered from disk.
        let dc = DurableCollection::open(dir.path(), cfg, FsyncMode::Sync).unwrap();
        assert_eq!(dc.list_versions().len(), 1);
        assert_eq!(dc.len(), 28); // live reflects the deletes (replayed from WAL)
        // Time-travel to v1 still sees all 30 (snapshot isolation across restart).
        let at_v1 = dc
            .search_at(&VersionSelector::Version(v1), &vecf(8, 1), 40, None)
            .unwrap();
        assert_eq!(at_v1.len(), 30);
    }

    #[test]
    fn export_import_roundtrip() {
        let src = tempfile::tempdir().unwrap();
        let tar_dir = tempfile::tempdir().unwrap();
        let cfg = CollectionConfig::new("c", 8, Metric::Dot);
        let tar = tar_dir.path().join("backup.tar");
        {
            let dc = DurableCollection::open(src.path(), cfg.clone(), FsyncMode::Sync).unwrap();
            dc.upsert((0..50).map(|i| (vecf(8, i), None)).collect())
                .unwrap();
            dc.commit(Some("v0".into()), None).unwrap();
            dc.export(&tar).unwrap();
        }
        // Import into a fresh directory and reopen.
        let dest = tempfile::tempdir().unwrap();
        import(&tar, dest.path()).unwrap();
        let dc = DurableCollection::open(dest.path(), cfg, FsyncMode::Sync).unwrap();
        assert_eq!(dc.len(), 50);
        assert_eq!(dc.list_versions().len(), 1);
    }

    #[test]
    fn checkpoint_then_recover() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = CollectionConfig::new("c", 8, Metric::Cosine);
        {
            let dc = DurableCollection::open(dir.path(), cfg.clone(), FsyncMode::Sync).unwrap();
            dc.upsert((0..30).map(|i| (vecf(8, i), None)).collect())
                .unwrap();
            dc.checkpoint().unwrap();
            // more writes after the checkpoint
            dc.upsert((30..40).map(|i| (vecf(8, i), None)).collect())
                .unwrap();
            assert_eq!(dc.len(), 40);
        }
        let dc = DurableCollection::open(dir.path(), cfg, FsyncMode::Sync).unwrap();
        assert_eq!(dc.len(), 40); // 30 from the sealed segment + 10 replayed from WAL
    }

    proptest::proptest! {
        /// Random insert/checkpoint/insert/delete sequences must recover to exactly
        /// the acked live set after a crash (drop without a final checkpoint).
        #[test]
        fn prop_recovery_preserves_acked(
            n1 in 1usize..40,
            do_ckpt in proptest::bool::ANY,
            n2 in 0usize..40,
            ndel in 0usize..20,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let cfg = CollectionConfig::new("c", 8, Metric::Dot);
            let mut all_ids = Vec::new();
            {
                let dc = DurableCollection::open(dir.path(), cfg.clone(), FsyncMode::Sync).unwrap();
                for i in 0..n1 {
                    all_ids.push(dc.upsert(vec![(vecf(8, i), None)]).unwrap()[0]);
                }
                if do_ckpt {
                    dc.checkpoint().unwrap();
                }
                for i in 0..n2 {
                    all_ids.push(dc.upsert(vec![(vecf(8, n1 + i), None)]).unwrap()[0]);
                }
                let ndel = ndel.min(all_ids.len());
                for &id in &all_ids[..ndel] {
                    dc.delete(id).unwrap();
                }
                // crash
            }
            let dc = DurableCollection::open(dir.path(), cfg, FsyncMode::Sync).unwrap();
            let ndel = ndel.min(all_ids.len());
            proptest::prop_assert_eq!(dc.len(), all_ids.len() - ndel);
        }
    }
}
