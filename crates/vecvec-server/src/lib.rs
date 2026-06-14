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

/// Serves the gRPC API (collections + points + query, with health and reflection)
/// on `listener` until the process is shut down.
pub async fn serve(service: Arc<Service>, listener: TcpListener) -> Result<(), BoxError> {
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

    Server::builder()
        .add_service(health_service)
        .add_service(reflection)
        .add_service(CollectionsServer::new(api.clone()))
        .add_service(PointsServer::new(api.clone()))
        .add_service(QueryServer::new(api.clone()))
        .add_service(VersioningServer::new(api))
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await?;
    Ok(())
}
