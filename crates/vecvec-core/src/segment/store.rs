//! The sealed-segment store: durable, content-checked, mmap-loaded segment files.
//!
//! Sealed segments are written atomically as `<dir>/<id>.seg` and loaded back by
//! memory-mapping the file, validating its frame (magic + CRC), then decoding. A
//! `Weak` cache hands out a shared `Arc<SealedSegment>` so repeated loads of a hot
//! segment don't re-read or re-decode, while still letting unused segments drop. A
//! ref-count via `Weak` is also the hook GC (M10) uses to know a segment is unused.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::error::{CoreError, Result};
use crate::id::SegmentId;
use crate::persist::atomic::{FileKind, parse_framed, write_atomic};

use super::codec::{SEGMENT_FORMAT_VERSION, decode_segment, encode_segment};
use super::sealed::SealedSegment;

/// A directory of persisted sealed segments with an in-memory cache.
pub struct SegmentStore {
    dir: PathBuf,
    cache: Mutex<HashMap<SegmentId, Weak<SealedSegment>>>,
}

impl SegmentStore {
    /// Opens (or prepares) a store rooted at `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn segment_path(&self, id: SegmentId) -> PathBuf {
        self.dir.join(format!("{}.seg", id.get()))
    }

    /// Persists `segment` to disk atomically.
    pub fn write_sealed(&self, segment: &SealedSegment) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(|e| CoreError::io(&self.dir, e))?;
        let payload = encode_segment(segment)?;
        write_atomic(
            &self.segment_path(segment.id()),
            FileKind::Segment,
            SEGMENT_FORMAT_VERSION,
            &payload,
        )
    }

    /// Deletes a segment's file from disk (used by GC). Best-effort.
    pub fn remove(&self, id: SegmentId) {
        let _ = std::fs::remove_file(self.segment_path(id));
        self.cache.lock().remove(&id);
    }

    /// Loads a sealed segment by id, returning a cached `Arc` if one is live.
    pub fn load(&self, id: SegmentId) -> Result<Arc<SealedSegment>> {
        if let Some(existing) = self.cache.lock().get(&id).and_then(Weak::upgrade) {
            return Ok(existing);
        }
        let path = self.segment_path(id);
        let file = std::fs::File::open(&path).map_err(|e| CoreError::io(&path, e))?;
        // SAFETY: segment files are written once via atomic rename and never mutated
        // in place; we only ever read the mapping. Out-of-process mutation is the
        // caller's contract not to do (documented store invariant).
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| CoreError::io(&path, e))? };

        let view = parse_framed(&mmap, &path)?;
        if view.format_version > SEGMENT_FORMAT_VERSION {
            return Err(CoreError::UnsupportedVersion {
                path,
                found: view.format_version,
                supported: SEGMENT_FORMAT_VERSION,
            });
        }
        let segment = Arc::new(decode_segment(id, view.payload)?);
        self.cache.lock().insert(id, Arc::downgrade(&segment));
        Ok(segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{DistanceKernel, Metric};
    use crate::index::brute_force_topk;
    use crate::segment::AppendableSegment;
    use crate::vector::VectorStorage;
    use crate::version::DeletionVector;
    use crate::{GlobalId, HnswConfig};

    fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503))
                    .wrapping_add(i as u32 * seed);
                ((x % 10_000) as f32 / 5_000.0) - 1.0
            })
            .collect()
    }

    fn build_sealed(dim: usize, n: usize, metric: Metric, id: SegmentId) -> SealedSegment {
        let mut seg = AppendableSegment::new(dim, metric);
        for i in 0..n {
            seg.append(GlobalId::new(1000 + i as u64), &vec_of(dim, i as u32 + 1));
        }
        seg.seal(id, HnswConfig::default())
    }

    #[test]
    fn write_then_mmap_load_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let store = SegmentStore::new(dir.path());
        let dim = 24;
        let n = 500;
        let seg = build_sealed(dim, n, Metric::Cosine, SegmentId::new(3));
        store.write_sealed(&seg).unwrap();

        let loaded = store.load(SegmentId::new(3)).unwrap();
        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.id(), SegmentId::new(3));

        // The reopened segment returns the same results as the original.
        let query = vec_of(dim, 77_777);
        let dv = DeletionVector::new();
        let original = seg.search(&query, 10, &dv, None);
        let reopened = loaded.search(&query, 10, &dv, None);
        assert_eq!(original, reopened);
    }

    #[test]
    fn sealed_hnsw_recall_vs_flat() {
        let dim = 24;
        let n = 1500;
        let metric = Metric::Dot;
        let seg = build_sealed(dim, n, metric, SegmentId::new(0));

        // Oracle over the same vectors (global ids start at 1000).
        let mut storage = VectorStorage::new(dim, metric);
        for i in 0..n {
            storage.push(&vec_of(dim, i as u32 + 1));
        }
        let kernel = DistanceKernel::new(metric, dim);

        let mut hits = 0usize;
        let trials = 20usize;
        for q in 0..trials {
            let query = vec_of(dim, 500_000 + q as u32);
            let dv = DeletionVector::new();
            let got: std::collections::HashSet<u64> = seg
                .search(&query, 10, &dv, None)
                .into_iter()
                .map(|(g, _)| g.get())
                .collect();
            let truth = brute_force_topk(&storage, &kernel, &query, 10, None, None);
            for sp in truth {
                if got.contains(&(1000 + sp.id.get() as u64)) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f32 / (10 * trials) as f32;
        assert!(recall >= 0.95, "sealed HNSW recall@10 {recall} < 0.95");
    }

    #[test]
    fn cache_returns_same_arc() {
        let dir = tempfile::tempdir().unwrap();
        let store = SegmentStore::new(dir.path());
        let seg = build_sealed(8, 50, Metric::Dot, SegmentId::new(1));
        store.write_sealed(&seg).unwrap();
        let a = store.load(SegmentId::new(1)).unwrap();
        let b = store.load(SegmentId::new(1)).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn corrupt_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = SegmentStore::new(dir.path());
        let seg = build_sealed(8, 50, Metric::Dot, SegmentId::new(2));
        store.write_sealed(&seg).unwrap();
        // Flip a byte in the payload region.
        let path = dir.path().join("2.seg");
        let mut raw = std::fs::read(&path).unwrap();
        let mid = raw.len() / 2;
        raw[mid] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();
        assert!(store.load(SegmentId::new(2)).is_err());
    }

    #[test]
    fn load_rejects_future_format_version() {
        // The forward-compatibility guard: a segment written by a newer binary (format
        // version > what we support) must be rejected with a typed UnsupportedVersion
        // error, not silently mis-decoded.
        let dir = tempfile::tempdir().unwrap();
        let store = SegmentStore::new(dir.path());
        let seg = build_sealed(8, 50, Metric::Dot, SegmentId::new(7));
        // Write a structurally-valid frame (real payload + CRC) but stamp a format
        // version one past the supported maximum.
        let payload = encode_segment(&seg).unwrap();
        write_atomic(
            &dir.path().join("7.seg"),
            FileKind::Segment,
            SEGMENT_FORMAT_VERSION + 1,
            &payload,
        )
        .unwrap();

        match store.load(SegmentId::new(7)) {
            Err(CoreError::UnsupportedVersion {
                found, supported, ..
            }) => {
                assert_eq!(found, SEGMENT_FORMAT_VERSION + 1);
                assert_eq!(supported, SEGMENT_FORMAT_VERSION);
            }
            Err(other) => panic!("expected UnsupportedVersion, got {other:?}"),
            Ok(_) => panic!("expected UnsupportedVersion, but load succeeded"),
        }
    }
}
