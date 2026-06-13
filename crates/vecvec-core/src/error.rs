//! Crate-wide error type.
//!
//! Subsystems add their own richer error enums over time, but they all convert
//! into [`CoreError`] at the crate boundary so callers (and the server) handle a
//! single type. Persistence/recovery is the dominant source of fallible operations
//! at M0, so the early variants reflect on-disk integrity.

use std::path::PathBuf;

/// The result type used throughout `vecvec-core`.
pub type Result<T> = std::result::Result<T, CoreError>;

/// The unified error type for `vecvec-core`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CoreError {
    /// An underlying I/O failure, with the path that caused it where known.
    #[error("io error{}: {source}", path_suffix(.path))]
    Io {
        /// The path involved, if the failure was tied to a specific file.
        path: Option<PathBuf>,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A file's magic bytes didn't match the expected framing — not a vecvec file,
    /// or a wrong file kind.
    #[error("bad magic in {path:?}: expected {expected:?}, found {found:?}")]
    BadMagic {
        /// The file that failed the check.
        path: PathBuf,
        /// The magic we expected.
        expected: [u8; 4],
        /// The magic we found.
        found: [u8; 4],
    },

    /// A file used a format version this build can't read.
    #[error(
        "unsupported format version in {path:?}: file is v{found}, this build supports up to v{supported}"
    )]
    UnsupportedVersion {
        /// The file that failed the check.
        path: PathBuf,
        /// The version recorded in the file.
        found: u32,
        /// The newest version this build understands.
        supported: u32,
    },

    /// A CRC mismatch — the file is truncated or corrupted.
    #[error("checksum mismatch in {path:?}: expected {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// The file that failed the check.
        path: PathBuf,
        /// The CRC stored in the file's footer.
        expected: u32,
        /// The CRC we computed over the bytes.
        computed: u32,
    },

    /// A file was structurally malformed (too short, inconsistent length fields).
    #[error("corrupt file {path:?}: {detail}")]
    Corrupt {
        /// The file that failed the check.
        path: PathBuf,
        /// A human-readable description of what was wrong.
        detail: String,
    },
}

impl CoreError {
    /// Builds an [`CoreError::Io`] that remembers the path involved.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        CoreError::Io {
            path: Some(path.into()),
            source,
        }
    }
}

impl From<std::io::Error> for CoreError {
    fn from(source: std::io::Error) -> Self {
        CoreError::Io { path: None, source }
    }
}

fn path_suffix(path: &Option<PathBuf>) -> String {
    match path {
        Some(p) => format!(" ({})", p.display()),
        None => String::new(),
    }
}
