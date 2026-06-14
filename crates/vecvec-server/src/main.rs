//! `vecvec-server` binary entry point (M3 vertical slice).

use std::sync::Arc;

use vecvec_core::FsyncMode;
use vecvec_server::{Service, serve};

#[tokio::main]
async fn main() -> Result<(), vecvec_server::BoxError> {
    let addr = std::env::var("VECVEC_GRPC_ADDR").unwrap_or_else(|_| "127.0.0.1:6334".to_string());
    let data_dir = std::env::var("VECVEC_DATA_DIR").unwrap_or_else(|_| "vecvec-data".to_string());
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Recover any existing collections before serving.
    let service = Arc::new(Service::open(&data_dir, cpus, cpus * 8, FsyncMode::Sync)?);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!(
        "vecvec-server {} listening on grpc://{} (data: {}) ",
        vecvec_core::VERSION,
        listener.local_addr()?,
        data_dir,
    );
    serve(service, listener).await
}
