//! `vecvec-server` binary entry point (M3 vertical slice).

use std::sync::Arc;

use vecvec_core::FsyncMode;
use vecvec_server::{Service, compaction_policy_from_env, serve, serve_rest};

#[tokio::main]
async fn main() -> Result<(), vecvec_server::BoxError> {
    let grpc_addr =
        std::env::var("VECVEC_GRPC_ADDR").unwrap_or_else(|_| "127.0.0.1:6334".to_string());
    let rest_addr =
        std::env::var("VECVEC_REST_ADDR").unwrap_or_else(|_| "127.0.0.1:6333".to_string());
    let data_dir = std::env::var("VECVEC_DATA_DIR").unwrap_or_else(|_| "vecvec-data".to_string());
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Recover any existing collections before serving.
    let compaction = compaction_policy_from_env();
    let service = Arc::new(Service::open_with_compaction(
        &data_dir,
        cpus,
        cpus * 8,
        FsyncMode::Sync,
        compaction,
    )?);

    let grpc_listener = tokio::net::TcpListener::bind(&grpc_addr).await?;
    let rest_listener = tokio::net::TcpListener::bind(&rest_addr).await?;
    println!(
        "vecvec-server {} — grpc://{} · rest http://{} (data: {})",
        vecvec_core::VERSION,
        grpc_listener.local_addr()?,
        rest_listener.local_addr()?,
        data_dir,
    );
    println!(
        "auto-compaction: max_segments={:?}, interval_secs={:?}",
        compaction.max_segments,
        compaction.interval_ms.map(|ms| ms / 1000),
    );

    let rest_service = service.clone();
    let rest = tokio::spawn(async move { serve_rest(rest_service, rest_listener).await });
    let grpc_result = serve(service, grpc_listener).await;
    rest.abort();
    grpc_result
}
