//! `vecvec-server` — the database server.
//!
//! M3 vertical slice: a tokio + tonic gRPC server exposing `Collections.Create`,
//! `Points.Upsert` (client-streaming) and `Query.Query` over the in-RAM
//! [`Service`], with gRPC health and reflection. CPU-bound work runs off the
//! reactor via the [`BlockingBridge`](blocking::BlockingBridge). REST, the full
//! API, durability, and observability arrive in later milestones.

pub mod blocking;
pub mod grpc;
pub mod registry;
pub mod rest;
pub mod service;

pub use service::Service;

use std::sync::Arc;

use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use vecvec_proto::pb::collections_server::CollectionsServer;
use vecvec_proto::pb::points_server::PointsServer;
use vecvec_proto::pb::query_server::QueryServer;
use vecvec_proto::pb::versioning_server::VersioningServer;

use crate::grpc::Api;

/// A boxed error usable across `.await` points.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// How often the background maintenance task auto-compacts + checkpoints
/// collections. This also bounds the resolution of the compaction time trigger.
const MAINTENANCE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// Builds the auto-compaction policy from the environment:
/// - `VECVEC_COMPACT_MAX_SEGMENTS` — compact at this many working segments
///   (default `8`; set `0` to disable the segment-count trigger).
/// - `VECVEC_COMPACT_INTERVAL_SECS` — compact this often (default `300`; set `0`
///   to disable the time trigger).
///
/// An unset var uses the default; an explicit `0` (or unparseable value) disables
/// that trigger. Both disabled = manual compaction only.
pub fn compaction_policy_from_env() -> vecvec_core::CompactionPolicy {
    let max_segments = match std::env::var("VECVEC_COMPACT_MAX_SEGMENTS") {
        Ok(s) => s.parse::<usize>().ok().filter(|&n| n > 0),
        Err(_) => Some(8),
    };
    let interval_ms = match std::env::var("VECVEC_COMPACT_INTERVAL_SECS") {
        Ok(s) => s
            .parse::<u64>()
            .ok()
            .filter(|&n| n > 0)
            .map(|secs| secs * 1000),
        Err(_) => Some(300_000),
    };
    vecvec_core::CompactionPolicy {
        max_segments,
        interval_ms,
    }
}

/// Serves the gRPC API (collections + points + query, with health and reflection)
/// on `listener` until the process is shut down.
pub async fn serve(service: Arc<Service>, listener: TcpListener) -> Result<(), BoxError> {
    // Background maintenance: periodically auto-compact (if a collection's trigger
    // fired) and checkpoint every collection (folding the WAL into sealed segments,
    // keeping recovery fast). Cancelled on shutdown.
    let maintenance = {
        let service = service.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(MAINTENANCE_INTERVAL);
            ticker.tick().await; // skip the immediate first tick
            loop {
                ticker.tick().await;
                for collection in service.registry().snapshot() {
                    let _ = tokio::task::spawn_blocking(move || {
                        // A compaction checkpoints itself, so only checkpoint
                        // separately when nothing compacted. Both fold the WAL.
                        match collection.maybe_compact() {
                            Ok(Some(_)) => {}
                            _ => {
                                let _ = collection.checkpoint();
                            }
                        }
                    })
                    .await;
                }
            }
        })
    };

    let api = Api::new(service);

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<CollectionsServer<Api>>()
        .await;
    health_reporter.set_serving::<PointsServer<Api>>().await;
    health_reporter.set_serving::<QueryServer<Api>>().await;
    health_reporter.set_serving::<VersioningServer<Api>>().await;

    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(vecvec_proto::FILE_DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
        .build_v1()?;

    let result = Server::builder()
        .add_service(health_service)
        .add_service(reflection)
        .add_service(CollectionsServer::new(api.clone()))
        .add_service(PointsServer::new(api.clone()))
        .add_service(QueryServer::new(api.clone()))
        .add_service(VersioningServer::new(api))
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await;

    maintenance.abort();
    result?;
    Ok(())
}

/// Serves the REST/JSON gateway on `listener` until shut down (the same `Service`
/// the gRPC surface uses).
pub async fn serve_rest(service: Arc<Service>, listener: TcpListener) -> Result<(), BoxError> {
    axum::serve(listener, rest::router(service)).await?;
    Ok(())
}
