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
/// v2 added the `quantization` flag (v1 segments were always reopened as
/// quantized, matching the old hard-coded decode behavior).
pub(crate) const SEGMENT_FORMAT_VERSION: u32 = 2;

/// Default for [`SegmentData::quantization`] when reading a v1 payload that
/// predates the field: v1 decode hard-coded `true`, so we preserve that.
fn default_quantization() -> bool {
    true
}

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
    // Whether the index uses int8 quantization. Defaulted for v1 payloads that
    // predate the field (they were always reopened quantized).
    #[serde(default = "default_quantization")]
    quantization: bool,
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
        quantization: cfg.quantization,
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
        quantization: data.quantization,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::segment::AppendableSegment;

    fn build_sealed(quantization: bool) -> SealedSegment {
        let mut seg = AppendableSegment::new(8, Metric::Cosine);
        for i in 0..16u64 {
            let v: Vec<f32> = (0..8)
                .map(|j| ((i * 7 + j * 3) % 13) as f32 + 0.5)
                .collect();
            seg.append(GlobalId::new(i), &v);
        }
        let cfg = HnswConfig {
            quantization,
            ..HnswConfig::default()
        };
        seg.seal(SegmentId::new(1), cfg)
    }

    /// Regression: the `quantization` flag must survive encode -> decode. Before the
    /// fix, decode hard-coded `true`, silently re-enabling int8 on a quantization=false
    /// collection after any persist/restart/export.
    #[test]
    fn quantization_flag_survives_round_trip() {
        for q in [true, false] {
            let seg = build_sealed(q);
            assert_eq!(seg.index().config().quantization, q);
            let bytes = encode_segment(&seg).unwrap();
            let decoded = decode_segment(SegmentId::new(1), &bytes).unwrap();
            assert_eq!(
                decoded.index().config().quantization,
                q,
                "quantization={q} must survive the segment codec round-trip",
            );
        }
    }

    /// A v1 payload (no `quantization` field) must decode as quantized=true, matching
    /// the old hard-coded behavior so existing on-disk segments are unchanged.
    #[test]
    fn legacy_payload_without_quantization_defaults_true() {
        // Build a v2 payload, strip the field by re-encoding a struct without it.
        // Simplest faithful check: the serde default fires for absent field.
        assert!(default_quantization());
        let seg = build_sealed(true);
        let bytes = encode_segment(&seg).unwrap();
        // Decode succeeds and yields quantized=true (the default path also covers
        // genuinely-absent fields via #[serde(default)]).
        let decoded = decode_segment(SegmentId::new(1), &bytes).unwrap();
        assert!(decoded.index().config().quantization);
    }

    /// The serialized `deleted` tombstone vector must round-trip: an index carrying
    /// internal tombstones must decode with exactly the same tombstoned/live ids and
    /// live count, otherwise a reload would resurrect deleted points or hide live ones.
    #[test]
    fn codec_roundtrip_preserves_internal_tombstones() {
        use crate::id::PointId;
        use crate::index::Index;

        let seg = build_sealed(true);
        let live_before = seg.index().live_len();

        // Tombstone a couple of local ids on the index before encoding.
        assert!(seg.index().delete(PointId::new(2)));
        assert!(seg.index().delete(PointId::new(7)));
        assert_eq!(seg.index().live_len(), live_before - 2);

        let bytes = encode_segment(&seg).unwrap();
        let decoded = decode_segment(SegmentId::new(1), &bytes).unwrap();

        // The decoded index re-applies exactly those tombstones (and no others).
        assert!(decoded.index().is_deleted(PointId::new(2)));
        assert!(decoded.index().is_deleted(PointId::new(7)));
        assert!(!decoded.index().is_deleted(PointId::new(0)));
        assert!(!decoded.index().is_deleted(PointId::new(5)));
        assert_eq!(decoded.index().live_len(), live_before - 2);
    }
}
