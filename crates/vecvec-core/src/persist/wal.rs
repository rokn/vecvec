//! A minimal append-only write-ahead log.
//!
//! Every mutation is appended here and made durable (`fsync`) **before** the client
//! is acked and before it is applied in memory, so an acked write survives a crash:
//! recovery replays the log. Records are individually CRC'd and length-framed, so a
//! crash mid-write leaves a *torn tail* that is detected and truncated on open
//! rather than corrupting earlier records.
//!
//! This is deliberately simple (one file, whole-file scan on open). Checkpoints keep
//! it short by folding the log into sealed segments and switching to a fresh log
//! generation (see [`DurableCollection`](crate::durable::DurableCollection)).

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

/// A logged mutation. The single source of truth for both live apply and recovery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalOp {
    /// Insert a vector (and optional payload) under a specific global id.
    Upsert {
        /// The collection-global id assigned to this point.
        id: u64,
        /// The vector (in stored form).
        vector: Vec<f32>,
        /// Optional JSON payload.
        #[serde(default)]
        payload: Option<crate::payload::Payload>,
    },
    /// Tombstone a point by global id.
    Delete {
        /// The global id to delete.
        id: u64,
    },
}

const RECORD_HEADER_LEN: usize = 8; // payload_len(u32) + crc32(u32)

/// An append-only write-ahead log file.
pub struct Wal {
    path: PathBuf,
    file: File,
    count: u64,
}

impl Wal {
    /// Opens (creating if absent) the log at `path`, truncating any torn tail.
    /// Returns the log positioned for appends.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        let (ops, valid_end) = read_records(&path)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| CoreError::io(&path, e))?;
        // Drop a torn tail so future appends never sit behind garbage.
        file.set_len(valid_end)
            .map_err(|e| CoreError::io(&path, e))?;
        let mut wal = Self {
            path,
            file,
            count: ops.len() as u64,
        };
        wal.file
            .seek(SeekFrom::End(0))
            .map_err(|e| CoreError::io(&wal.path, e))?;
        Ok(wal)
    }

    /// The number of records currently in the log.
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Appends `op` (without fsync — call [`Wal::flush`] to make it durable).
    pub fn append(&mut self, op: &WalOp) -> Result<()> {
        let payload = rmp_serde::to_vec(op).map_err(|e| CoreError::Serialization {
            detail: e.to_string(),
        })?;
        let crc = crc32fast::hash(&payload);
        let mut framed = Vec::with_capacity(RECORD_HEADER_LEN + payload.len());
        framed.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        framed.extend_from_slice(&crc.to_le_bytes());
        framed.extend_from_slice(&payload);
        self.file
            .write_all(&framed)
            .map_err(|e| CoreError::io(&self.path, e))?;
        self.count += 1;
        Ok(())
    }

    /// Flushes and fsyncs the log so all appended records are durable.
    pub fn flush(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|e| CoreError::io(&self.path, e))?;
        self.file
            .sync_data()
            .map_err(|e| CoreError::io(&self.path, e))
    }

    /// Reads every valid record (stopping at a torn tail).
    pub fn read_all(&self) -> Result<Vec<WalOp>> {
        Ok(read_records(&self.path)?.0)
    }
}

/// Parses all valid records from a log file, returning them plus the byte offset of
/// the end of the last valid record (everything after is a torn tail).
fn read_records(path: &Path) -> Result<(Vec<WalOp>, u64)> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(CoreError::io(path, e)),
    };

    let mut ops = Vec::new();
    let mut off = 0usize;
    while off < bytes.len() {
        if off + RECORD_HEADER_LEN > bytes.len() {
            break; // torn header
        }
        let len = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4")) as usize;
        let crc = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().expect("4"));
        let payload_start = off + RECORD_HEADER_LEN;
        let payload_end = payload_start + len;
        if payload_end > bytes.len() {
            break; // torn payload
        }
        let payload = &bytes[payload_start..payload_end];
        if crc32fast::hash(payload) != crc {
            break; // corrupt / torn
        }
        match rmp_serde::from_slice::<WalOp>(payload) {
            Ok(op) => ops.push(op),
            Err(_) => break,
        }
        off = payload_end;
    }
    Ok((ops, off as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_flush_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&WalOp::Upsert {
                id: 1,
                vector: vec![1.0, 2.0],
                payload: None,
            })
            .unwrap();
            wal.append(&WalOp::Delete { id: 1 }).unwrap();
            wal.flush().unwrap();
            assert_eq!(wal.len(), 2);
        }
        let wal = Wal::open(&path).unwrap();
        let ops = wal.read_all().unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[1], WalOp::Delete { id: 1 });
        assert_eq!(wal.len(), 2);
    }

    #[test]
    fn torn_tail_is_truncated_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            for i in 0..5 {
                wal.append(&WalOp::Upsert {
                    id: i,
                    vector: vec![i as f32],
                    payload: None,
                })
                .unwrap();
            }
            wal.flush().unwrap();
        }
        // Append garbage simulating a half-written record after a crash.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[0xFF, 0x00, 0x00, 0x00, 0xDE, 0xAD]).unwrap();
        }
        let wal = Wal::open(&path).unwrap();
        assert_eq!(wal.len(), 5); // the 5 good records survive; the torn tail is dropped
        // The truncation persists: reopening still sees exactly 5.
        let wal2 = Wal::open(&path).unwrap();
        assert_eq!(wal2.read_all().unwrap().len(), 5);
    }
}
