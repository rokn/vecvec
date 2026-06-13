//! Durable persistence primitives.
//!
//! At M0 this is just [`atomic`] — crash-safe atomic file replacement, the
//! foundation every on-disk artifact (manifests, sealed segments, the `HEAD`
//! pointer) is written through. The WAL, checkpointing, and recovery layers
//! described in `BuildPlan.md` build on top of it in later milestones.

pub mod atomic;

pub use atomic::{FileKind, Frame, read_framed, write_atomic};
