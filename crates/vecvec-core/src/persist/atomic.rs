//! Crash-safe atomic file writes.
//!
//! Every durable on-disk artifact in vecvec (manifests, sealed segments, deletion
//! vectors, the `HEAD` pointer) is written through [`write_atomic`], which gives
//! the **all-or-nothing replacement** guarantee the durability design depends on:
//! after a crash a target file is observed as *either* its previous contents *or*
//! the fully-written new contents — never a torn mix.
//!
//! It achieves this with the standard sequence:
//! 1. write the new bytes to a temporary file in the **same directory**,
//! 2. `fsync` the temp file (so its data is durable),
//! 3. atomically `rename` it onto the target (atomic within a filesystem),
//! 4. `fsync` the containing **directory** (so the rename itself is durable — a
//!    rename is atomic but not durable on ext4/xfs without this).
//!
//! Files are framed with a magic, a format version, a kind tag, and a CRC-32 over
//! the header+payload, plus a trailing end-magic so truncated tails are detected.
//! [`read_framed`] validates all of that and returns a structured error (never a
//! panic, never silently-wrong bytes) on any corruption.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::error::{CoreError, Result};

const MAGIC: [u8; 4] = *b"VVEC";
const END_MAGIC: [u8; 4] = *b"VEND";
const HEADER_LEN: usize = 20; // magic(4) + version(4) + kind(4) + payload_len(8)
const FOOTER_LEN: usize = 8; // crc32(4) + end_magic(4)

/// The kind of artifact stored in a framed file. The framing layer treats the kind
/// as an opaque tag (it round-trips it but never rejects an unknown one), so older
/// builds can still read newer files structurally; callers interpret it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u32)]
pub enum FileKind {
    /// Unspecified / generic payload.
    Generic = 0,
    /// A version manifest.
    Manifest = 1,
    /// A sealed segment.
    Segment = 2,
    /// A per-version deletion vector.
    DeletionVector = 3,
    /// The collection `HEAD` pointer.
    Head = 4,
    /// A checkpoint snapshot.
    Snapshot = 5,
}

impl FileKind {
    /// The raw tag written into the file header.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

/// A decoded framed file: its declared format version, kind tag, and payload bytes.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The caller-defined payload format version recorded in the header.
    pub format_version: u32,
    /// The raw [`FileKind`] tag (kept as a `u32` for forward compatibility).
    pub kind: u32,
    /// The payload bytes (header/footer stripped, integrity already verified).
    pub payload: Vec<u8>,
}

/// Atomically writes `payload` to `path`, replacing any existing file.
///
/// On success the bytes are durable and the target file contains exactly the new
/// framed contents. On any failure the previous file (if any) is left untouched.
pub fn write_atomic(
    path: &Path,
    kind: FileKind,
    format_version: u32,
    payload: &[u8],
) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..8].copy_from_slice(&format_version.to_le_bytes());
    header[8..12].copy_from_slice(&kind.as_u32().to_le_bytes());
    header[12..20].copy_from_slice(&(payload.len() as u64).to_le_bytes());

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(payload);
    let crc = hasher.finalize();

    let mut footer = [0u8; FOOTER_LEN];
    footer[0..4].copy_from_slice(&crc.to_le_bytes());
    footer[4..8].copy_from_slice(&END_MAGIC);

    // 1. Write to a temp file in the same directory.
    let mut tmp = tempfile::NamedTempFile::new_in(parent).map_err(|e| CoreError::io(parent, e))?;
    let tmp_path = tmp.path().to_path_buf();
    tmp.write_all(&header)
        .and_then(|()| tmp.write_all(payload))
        .and_then(|()| tmp.write_all(&footer))
        .and_then(|()| tmp.flush())
        .map_err(|e| CoreError::io(tmp_path.clone(), e))?;

    // 2. fsync the temp file's contents.
    tmp.as_file()
        .sync_all()
        .map_err(|e| CoreError::io(tmp_path, e))?;

    // 3. Atomically rename onto the target.
    tmp.persist(path)
        .map_err(|e| CoreError::io(path, e.error))?;

    // 4. fsync the directory so the rename itself is durable.
    fsync_dir(parent)
}

/// Reads and validates a framed file written by [`write_atomic`].
///
/// Returns a structured [`CoreError`] (`BadMagic`, `Corrupt`, `ChecksumMismatch`)
/// on any integrity problem rather than panicking or returning wrong bytes.
pub fn read_framed(path: &Path) -> Result<Frame> {
    let bytes = std::fs::read(path).map_err(|e| CoreError::io(path, e))?;

    if bytes.len() < HEADER_LEN + FOOTER_LEN {
        return Err(CoreError::Corrupt {
            path: path.into(),
            detail: format!("file too short: {} bytes", bytes.len()),
        });
    }

    let magic: [u8; 4] = bytes[0..4].try_into().expect("4 bytes");
    if magic != MAGIC {
        return Err(CoreError::BadMagic {
            path: path.into(),
            expected: MAGIC,
            found: magic,
        });
    }

    let format_version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
    let kind = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    let payload_len = u64::from_le_bytes(bytes[12..20].try_into().expect("8 bytes"));

    let expected_total = HEADER_LEN as u64 + payload_len + FOOTER_LEN as u64;
    if bytes.len() as u64 != expected_total {
        return Err(CoreError::Corrupt {
            path: path.into(),
            detail: format!(
                "length mismatch: header declares {payload_len}-byte payload \
                 (expected {expected_total} total) but file is {} bytes",
                bytes.len()
            ),
        });
    }

    let payload_end = HEADER_LEN + payload_len as usize;
    let stored_crc = u32::from_le_bytes(
        bytes[payload_end..payload_end + 4]
            .try_into()
            .expect("4 bytes"),
    );
    let end_magic: [u8; 4] = bytes[payload_end + 4..payload_end + 8]
        .try_into()
        .expect("4 bytes");
    if end_magic != END_MAGIC {
        return Err(CoreError::Corrupt {
            path: path.into(),
            detail: "missing end magic (truncated tail)".into(),
        });
    }

    let computed = crc32fast::hash(&bytes[0..payload_end]);
    if computed != stored_crc {
        return Err(CoreError::ChecksumMismatch {
            path: path.into(),
            expected: stored_crc,
            computed,
        });
    }

    Ok(Frame {
        format_version,
        kind,
        payload: bytes[HEADER_LEN..payload_end].to_vec(),
    })
}

fn fsync_dir(dir: &Path) -> Result<()> {
    let f = File::open(dir).map_err(|e| CoreError::io(dir, e))?;
    f.sync_all().map_err(|e| CoreError::io(dir, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, payload: &[u8]) {
        write_atomic(path, FileKind::Generic, 1, payload).unwrap();
    }

    #[test]
    fn roundtrips_payload_kind_and_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.bin");
        write_atomic(&path, FileKind::Manifest, 7, b"hello vecvec").unwrap();

        let frame = read_framed(&path).unwrap();
        assert_eq!(frame.payload, b"hello vecvec");
        assert_eq!(frame.kind, FileKind::Manifest.as_u32());
        assert_eq!(frame.format_version, 7);
    }

    #[test]
    fn empty_payload_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        write(&path, b"");
        assert_eq!(read_framed(&path).unwrap().payload, b"");
    }

    #[test]
    fn replacement_is_all_or_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.bin");
        write(&path, b"AAAA");
        assert_eq!(read_framed(&path).unwrap().payload, b"AAAA");
        write(&path, b"BBBBBBBB");
        assert_eq!(read_framed(&path).unwrap().payload, b"BBBBBBBB");
    }

    /// Models a crash mid-write: a temp file is fully written in the target's
    /// directory but the process dies before the rename. The live target must
    /// still read as its previous contents — never torn.
    #[test]
    fn crash_before_rename_leaves_old_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.bin");
        write(&path, b"OLD-CONTENTS");

        // Simulate a write of "NEW" that gets as far as a synced temp file but is
        // never persisted (renamed) onto the target.
        {
            let mut tmp = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
            tmp.write_all(b"\x00\x01\x02 partial new frame bytes")
                .unwrap();
            tmp.as_file().sync_all().unwrap();
            // Dropped here without `.persist(&path)` — the "crash".
        }

        // The target is untouched.
        assert_eq!(read_framed(&path).unwrap().payload, b"OLD-CONTENTS");
    }

    #[test]
    fn detects_bit_flip_in_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.bin");
        write(&path, b"important payload");

        let mut raw = std::fs::read(&path).unwrap();
        raw[HEADER_LEN] ^= 0xFF; // flip first payload byte
        std::fs::write(&path, &raw).unwrap();

        assert!(matches!(
            read_framed(&path),
            Err(CoreError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn detects_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.bin");
        write(&path, b"0123456789");

        let raw = std::fs::read(&path).unwrap();
        std::fs::write(&path, &raw[..raw.len() - 3]).unwrap();

        assert!(matches!(read_framed(&path), Err(CoreError::Corrupt { .. })));
    }

    #[test]
    fn detects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.bin");
        std::fs::write(&path, vec![0u8; 64]).unwrap();
        assert!(matches!(
            read_framed(&path),
            Err(CoreError::BadMagic { .. })
        ));
    }

    proptest::proptest! {
        #[test]
        fn prop_roundtrip(payload in proptest::collection::vec(proptest::num::u8::ANY, 0..4096), ver in 0u32..1000, kind in 0u32..6) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("p.bin");
            // Reconstruct a FileKind from the raw tag for the API; Generic for any.
            let fk = match kind { 1 => FileKind::Manifest, 2 => FileKind::Segment, 3 => FileKind::DeletionVector, 4 => FileKind::Head, 5 => FileKind::Snapshot, _ => FileKind::Generic };
            write_atomic(&path, fk, ver, &payload).unwrap();
            let frame = read_framed(&path).unwrap();
            proptest::prop_assert_eq!(&frame.payload, &payload);
            proptest::prop_assert_eq!(frame.format_version, ver);
            proptest::prop_assert_eq!(frame.kind, fk.as_u32());
        }

        /// Any single-byte corruption must be detected: `read_framed` either errors
        /// or returns the *exact* original payload — it never yields different bytes.
        #[test]
        fn prop_any_byte_flip_is_caught(payload in proptest::collection::vec(proptest::num::u8::ANY, 1..512), flip_at in 0usize..600) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("f.bin");
            write_atomic(&path, FileKind::Generic, 1, &payload).unwrap();

            let mut raw = std::fs::read(&path).unwrap();
            let idx = flip_at % raw.len();
            raw[idx] ^= 0x80;
            std::fs::write(&path, &raw).unwrap();

            // Either an error, or — if the flip somehow lands harmlessly — the
            // exact original payload. Never different bytes.
            if let Ok(frame) = read_framed(&path) {
                proptest::prop_assert_eq!(frame.payload, payload);
            }
        }
    }
}
