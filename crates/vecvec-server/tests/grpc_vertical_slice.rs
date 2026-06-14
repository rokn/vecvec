//! M3 exit test: drive the full vertical slice over real gRPC —
//! create collection → client-streaming upsert → query — and assert the server's
//! results equal the brute-force oracle computed in-process with `vecvec-core`.

use std::sync::Arc;
use std::time::Duration;

use tonic::transport::Channel;
use vecvec_core::{DistanceKernel, FsyncMode, Metric, VectorStorage, brute_force_topk};
use vecvec_proto::pb;
use vecvec_server::{Service, serve};

fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

/// Brute-force top-k over the same vectors, keyed by the ids the server assigns
/// (0..n in insertion order).
fn oracle(dim: usize, metric: Metric, n: usize, query: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut storage = VectorStorage::new(dim, metric);
    for i in 0..n {
        storage.push(&vec_of(dim, i as u32 + 1));
    }
    let kernel = DistanceKernel::new(metric, dim);
    brute_force_topk(&storage, &kernel, query, k, None, None)
        .into_iter()
        .map(|sp| (sp.id.get() as u64, sp.score))
        .collect()
}

async fn connect(addr: std::net::SocketAddr) -> Channel {
    for _ in 0..50 {
        if let Ok(ch) = Channel::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
        {
            return ch;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("server did not come up at {addr}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_upsert_query_over_grpc() {
    let dim = 64usize;
    let n = 500usize;
    let metric = Metric::Cosine;

    // Start the server on an ephemeral port.
    let data = tempfile::tempdir().unwrap();
    let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { serve(service, listener).await });

    let channel = connect(addr).await;

    // 0. Health reports SERVING for the Query service.
    let mut health = tonic_health::pb::health_client::HealthClient::new(channel.clone());
    let status = health
        .check(tonic_health::pb::HealthCheckRequest {
            service: "vecvec.v1.Query".into(),
        })
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(
        status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32
    );

    // 1. Create a collection.
    let mut collections = pb::collections_client::CollectionsClient::new(channel.clone());
    let created = collections
        .create(pb::CreateCollectionRequest {
            name: "embeddings".into(),
            dim: dim as u32,
            metric: pb::Metric::Cosine as i32,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(created.name, "embeddings");

    // 2. Client-streaming upsert in several batches.
    let mut points = pb::points_client::PointsClient::new(channel.clone());
    let batches: Vec<pb::UpsertRequest> = (0..n)
        .collect::<Vec<_>>()
        .chunks(100)
        .map(|chunk| pb::UpsertRequest {
            collection: "embeddings".into(),
            vectors: chunk
                .iter()
                .map(|&i| pb::Vector {
                    values: vec_of(dim, i as u32 + 1),
                })
                .collect(),
        })
        .collect();
    let upserted = points
        .upsert(tokio_stream::iter(batches))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(upserted.inserted, n as u64);
    assert_eq!(upserted.ids.len(), n);
    assert_eq!(upserted.ids.first().copied(), Some(0));

    // 3. Query, and compare to the oracle.
    let mut query_client = pb::query_client::QueryClient::new(channel);
    let q = vec_of(dim, 9_999);
    let resp = query_client
        .query(pb::QueryRequest {
            collection: "embeddings".into(),
            vector: q.clone(),
            k: 10,
        })
        .await
        .unwrap()
        .into_inner();

    let got: Vec<(u64, f32)> = resp.results.iter().map(|r| (r.id, r.score)).collect();
    let want = oracle(dim, metric, n, &q, 10);
    assert_eq!(got.len(), 10);
    assert_eq!(
        got, want,
        "gRPC query results must equal the brute-force oracle"
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn errors_map_to_grpc_status_codes() {
    let data = tempfile::tempdir().unwrap();
    let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { serve(service, listener).await });
    let channel = connect(addr).await;

    let mut collections = pb::collections_client::CollectionsClient::new(channel.clone());
    collections
        .create(pb::CreateCollectionRequest {
            name: "c".into(),
            dim: 4,
            metric: pb::Metric::Dot as i32,
        })
        .await
        .unwrap();

    // Duplicate create -> ALREADY_EXISTS.
    let dup = collections
        .create(pb::CreateCollectionRequest {
            name: "c".into(),
            dim: 4,
            metric: pb::Metric::Dot as i32,
        })
        .await
        .unwrap_err();
    assert_eq!(dup.code(), tonic::Code::AlreadyExists);

    // Query a missing collection -> NOT_FOUND.
    let mut query_client = pb::query_client::QueryClient::new(channel.clone());
    let missing = query_client
        .query(pb::QueryRequest {
            collection: "nope".into(),
            vector: vec![0.0; 4],
            k: 5,
        })
        .await
        .unwrap_err();
    assert_eq!(missing.code(), tonic::Code::NotFound);

    // Wrong-dimension query -> INVALID_ARGUMENT.
    let bad_dim = query_client
        .query(pb::QueryRequest {
            collection: "c".into(),
            vector: vec![0.0; 3],
            k: 5,
        })
        .await
        .unwrap_err();
    assert_eq!(bad_dim.code(), tonic::Code::InvalidArgument);

    server.abort();
}

/// M6: data written through one server instance is recovered by a fresh instance
/// pointed at the same data directory.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_survives_server_restart() {
    let data = tempfile::tempdir().unwrap();
    let dim = 32usize;
    let n = 200usize;

    // First instance: create + upsert, then shut down.
    {
        let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { serve(service, listener).await });
        let channel = connect(addr).await;

        pb::collections_client::CollectionsClient::new(channel.clone())
            .create(pb::CreateCollectionRequest {
                name: "emb".into(),
                dim: dim as u32,
                metric: pb::Metric::Cosine as i32,
            })
            .await
            .unwrap();
        let batch = pb::UpsertRequest {
            collection: "emb".into(),
            vectors: (0..n)
                .map(|i| pb::Vector {
                    values: vec_of(dim, i as u32 + 1),
                })
                .collect(),
        };
        let resp = pb::points_client::PointsClient::new(channel)
            .upsert(tokio_stream::iter(vec![batch]))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.inserted, n as u64);
        server.abort();
    }

    // Second instance over the same data dir: the collection + points are recovered.
    {
        let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { serve(service, listener).await });
        let channel = connect(addr).await;

        let resp = pb::query_client::QueryClient::new(channel)
            .query(pb::QueryRequest {
                collection: "emb".into(),
                vector: vec_of(dim, 7),
                k: n as u32,
            })
            .await
            .unwrap()
            .into_inner();
        // All n points recovered and searchable.
        assert_eq!(resp.results.len(), n);
        server.abort();
    }
}
