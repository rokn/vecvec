//! Sealed-segment on-disk encoding.
//!
//! A sealed segment serializes to a self-describing payload (vectors + id map +
//! HNSW graph + tombstones + the config needed to reopen it). The payload is framed
//! with magic + CRC by [`write_atomic`](crate::persist::atomic::write_atomic) at the
//! store layer, so corruption is caught before decoding.
//!
//! The payload uses MessagePack (`rmp-serde`) today; the schema is centralized in
//! [`SegmentData`] so a future move to a zero-copy format (rkyv) for true mmap
//! residency is a localized change behind this seam.

use std::sync::Arc;

use crate::distance::Metric;
use crate::error::{CoreError, Result};
use crate::id::{GlobalId, SegmentId};
use crate::index::HnswConfig;
use crate::index::HnswIndex;
use crate::index::hnsw::GraphLayers;
use crate::vector::VectorStorage;

use super::id_map::IdMap;
use super::sealed::SealedSegment;

/// The on-disk segment format version (bumped on any incompatible schema change).
pub(crate) const SEGMENT_FORMAT_VERSION: u32 = 1;

/// The serializable mirror of a sealed segment.
#[derive(serde::Serialize, serde::Deserialize)]
struct SegmentData {
    dim: u32,
    metric: u8,
    // HNSW config (so the graph can be reopened and queried with the same params).
    m: u32,
    m_max0: u32,
    ef_construction: u32,
    ef_search: u32,
    seed: u64,
    keep_pruned: bool,
    // Vectors (flat, row-major) and the local→global id map.
    vectors: Vec<f32>,
    global_ids: Vec<u64>,
    // Tombstoned local ids.
    deleted: Vec<u32>,
    // Sealed graph.
    entry: Option<u32>,
    max_level: u32,
    levels: Vec<u8>,
    l0_offsets: Vec<u32>,
    l0_links: Vec<u32>,
    upper: Vec<Vec<Vec<u32>>>,
    upper_index: Vec<u32>,
}

fn metric_to_u8(m: Metric) -> u8 {
    match m {
        Metric::Cosine => 0,
        Metric::Dot => 1,
        Metric::Euclidean => 2,
    }
}

fn metric_from_u8(b: u8) -> Result<Metric> {
    match b {
        0 => Ok(Metric::Cosine),
        1 => Ok(Metric::Dot),
        2 => Ok(Metric::Euclidean),
        other => Err(CoreError::Serialization {
            detail: format!("unknown metric tag {other}"),
        }),
    }
}

/// Encodes a sealed segment to its on-disk payload bytes.
pub(crate) fn encode_segment(seg: &SealedSegment) -> Result<Vec<u8>> {
    let index = seg.index();
    let vectors = index.vectors();
    let graph = index.graph();
    let cfg = index.config();
    let deleted: Vec<u32> = index.deleted().snapshot().iter().collect();

    let data = SegmentData {
        dim: vectors.dim() as u32,
        metric: metric_to_u8(vectors.metric()),
        m: cfg.m as u32,
        m_max0: cfg.m_max0 as u32,
        ef_construction: cfg.ef_construction as u32,
        ef_search: cfg.ef_search as u32,
        seed: cfg.seed,
        keep_pruned: cfg.keep_pruned,
        vectors: vectors.as_flat().to_vec(),
        global_ids: seg.id_map().global_ids().iter().map(|g| g.get()).collect(),
        deleted,
        entry: graph.entry,
        max_level: graph.max_level as u32,
        levels: graph.levels.clone(),
        l0_offsets: graph.l0_offsets.clone(),
        l0_links: graph.l0_links.clone(),
        upper: graph.upper.clone(),
        upper_index: graph.upper_index.clone(),
    };

    rmp_serde::to_vec(&data).map_err(|e| CoreError::Serialization {
        detail: e.to_string(),
    })
}

/// Decodes a sealed segment payload (already integrity-checked by the frame).
pub(crate) fn decode_segment(id: SegmentId, bytes: &[u8]) -> Result<SealedSegment> {
    let data: SegmentData = rmp_serde::from_slice(bytes).map_err(|e| CoreError::Serialization {
        detail: e.to_string(),
    })?;

    let metric = metric_from_u8(data.metric)?;
    let dim = data.dim as usize;
    let vectors = Arc::new(VectorStorage::from_flat(dim, metric, data.vectors));
    let ids = IdMap::from_global_ids(data.global_ids.into_iter().map(GlobalId::new).collect());
    let config = HnswConfig {
        m: data.m as usize,
        m_max0: data.m_max0 as usize,
        ef_construction: data.ef_construction as usize,
        ef_search: data.ef_search as usize,
        seed: data.seed,
        keep_pruned: data.keep_pruned,
        quantization: true,
    };
    let graph = GraphLayers {
        entry: data.entry,
        max_level: data.max_level as usize,
        levels: data.levels,
        l0_offsets: data.l0_offsets,
        l0_links: data.l0_links,
        upper: data.upper,
        upper_index: data.upper_index,
    };
    let index = HnswIndex::from_parts(vectors, config, graph, &data.deleted);
    Ok(SealedSegment::from_index(id, Arc::new(index), ids))
}
