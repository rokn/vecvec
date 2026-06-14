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
use crate::index::FilterContext;
use crate::persist::atomic::{FileKind, read_framed, write_atomic};
use crate::persist::wal::{Wal, WalOp};
use crate::segment::SegmentStore;

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

/// The persisted pointer to the latest durable state.
#[derive(Debug, Serialize, Deserialize)]
struct CheckpointHead {
    wal_generation: u64,
    sealed_ids: Vec<u64>,
    next_global_id: u64,
    next_segment_id: u64,
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
}

impl DurableCollection {
    /// Opens (and recovers, if data exists) a durable collection rooted at `dir`.
    pub fn open(dir: impl AsRef<Path>, config: CollectionConfig, fsync: FsyncMode) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| CoreError::io(&dir, e))?;
        write_config_if_absent(&dir, &config)?;
        let store = SegmentStore::new(dir.join("segments"));
        let collection = Arc::new(Collection::create(config));

        // Recover the checkpoint, if any.
        let head_path = dir.join("HEAD");
        let generation = if head_path.exists() {
            let frame = read_framed(&head_path)?;
            let head: CheckpointHead =
                rmp_serde::from_slice(&frame.payload).map_err(|e| CoreError::Serialization {
                    detail: e.to_string(),
                })?;
            let mut segments = Vec::with_capacity(head.sealed_ids.len());
            for id in &head.sealed_ids {
                segments.push(store.load(SegmentId::new(*id))?);
            }
            collection.install_sealed(segments);
            collection.set_allocators(head.next_global_id, head.next_segment_id);
            head.wal_generation
        } else {
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
        })
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

    /// Durably appends a batch of vectors, returning their assigned ids. WAL-first:
    /// logged + fsynced before applied in memory.
    pub fn upsert(&self, vectors: Vec<Vec<f32>>) -> Result<Vec<u64>> {
        let dim = self.collection.config().dim;
        for v in &vectors {
            if v.len() != dim {
                return Err(CoreError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let mut guard = self.wal.lock();
        let mut ids = Vec::with_capacity(vectors.len());
        for v in &vectors {
            let id = self.collection.alloc_global_id();
            guard.wal.append(&WalOp::Upsert {
                id: id.get(),
                vector: v.clone(),
            })?;
            ids.push(id.get());
        }
        if self.fsync == FsyncMode::Sync {
            guard.wal.flush()?;
        }
        // Durable -> apply in memory (the same path recovery uses).
        for (id, v) in ids.iter().zip(&vectors) {
            self.collection.insert_with_id(GlobalId::new(*id), v)?;
        }
        Ok(ids)
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
        filter: Option<&dyn FilterContext>,
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

        // 2. Persist every sealed segment durably.
        let sealed = self.collection.sealed_snapshot();
        let mut sealed_ids = Vec::new();
        for seg in sealed.iter() {
            self.store.write_sealed(seg)?;
            sealed_ids.push(seg.id().get());
        }

        // 3. Create the next (empty) WAL generation, then commit HEAD atomically.
        let new_generation = guard.generation + 1;
        let new_wal = Wal::open(Self::wal_path(&self.dir, new_generation))?;
        let head = CheckpointHead {
            wal_generation: new_generation,
            sealed_ids,
            next_global_id: self.collection.next_global_id_value(),
            next_segment_id: self.collection.next_segment_id_value(),
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

/// Applies a single op to the in-memory collection. The one apply path shared by
/// live writes and recovery.
fn apply_op(collection: &Collection, op: &WalOp) {
    match op {
        WalOp::Upsert { id, vector } => {
            // Recovery trusts the WAL; a dimension mismatch here would mean a
            // corrupt record that passed CRC, which we treat as non-fatal-skip.
            let _ = collection.insert_with_id(GlobalId::new(*id), vector);
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
            let ids = dc.upsert((0..20).map(|i| vecf(8, i)).collect()).unwrap();
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
    fn checkpoint_then_recover() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = CollectionConfig::new("c", 8, Metric::Cosine);
        {
            let dc = DurableCollection::open(dir.path(), cfg.clone(), FsyncMode::Sync).unwrap();
            dc.upsert((0..30).map(|i| vecf(8, i)).collect()).unwrap();
            dc.checkpoint().unwrap();
            // more writes after the checkpoint
            dc.upsert((30..40).map(|i| vecf(8, i)).collect()).unwrap();
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
                    all_ids.push(dc.upsert(vec![vecf(8, i)]).unwrap()[0]);
                }
                if do_ckpt {
                    dc.checkpoint().unwrap();
                }
                for i in 0..n2 {
                    all_ids.push(dc.upsert(vec![vecf(8, n1 + i)]).unwrap()[0]);
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
