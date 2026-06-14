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
                    payload: None,
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
            at: None,
            filter: None,
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
            at: None,
            filter: None,
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
            at: None,
            filter: None,
        })
        .await
        .unwrap_err();
    assert_eq!(bad_dim.code(), tonic::Code::InvalidArgument);

    server.abort();
}

/// M9: payload + filtered query over gRPC.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filtered_query_over_grpc() {
    let data = tempfile::tempdir().unwrap();
    let dim = 8usize;
    let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { serve(service, listener).await });
    let channel = connect(addr).await;

    pb::collections_client::CollectionsClient::new(channel.clone())
        .create(pb::CreateCollectionRequest {
            name: "f".into(),
            dim: dim as u32,
            metric: pb::Metric::Dot as i32,
        })
        .await
        .unwrap();

    // 100 points, each tagged with bucket = id % 5.
    let batch = pb::UpsertRequest {
        collection: "f".into(),
        vectors: (0..100u32)
            .map(|i| pb::Vector {
                values: vec_of(dim, i + 1),
                payload: Some(format!("{{\"bucket\":{}}}", i % 5)),
            })
            .collect(),
    };
    pb::points_client::PointsClient::new(channel.clone())
        .upsert(tokio_stream::iter(vec![batch]))
        .await
        .unwrap();

    // Query filtered to bucket == 2.
    let resp = pb::query_client::QueryClient::new(channel)
        .query(pb::QueryRequest {
            collection: "f".into(),
            vector: vec_of(dim, 9_999),
            k: 10,
            at: None,
            filter: Some(r#"{"must":[{"key":"bucket","match":2}]}"#.into()),
        })
        .await
        .unwrap()
        .into_inner();

    assert_eq!(resp.results.len(), 10);
    // Server-assigned ids are insertion order, so id % 5 == 2 <=> bucket 2.
    assert!(resp.results.iter().all(|r| r.id % 5 == 2));

    server.abort();
}

/// M7: the versioning differentiator over gRPC — commit, time-travel query, diff,
/// tag, restore.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn versioning_over_grpc() {
    let data = tempfile::tempdir().unwrap();
    let dim = 16usize;
    let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { serve(service, listener).await });
    let channel = connect(addr).await;

    let mut collections = pb::collections_client::CollectionsClient::new(channel.clone());
    let mut points = pb::points_client::PointsClient::new(channel.clone());
    let mut versioning = pb::versioning_client::VersioningClient::new(channel.clone());

    collections
        .create(pb::CreateCollectionRequest {
            name: "emb".into(),
            dim: dim as u32,
            metric: pb::Metric::Cosine as i32,
        })
        .await
        .unwrap();

    let upsert = |from: usize, n: usize| pb::UpsertRequest {
        collection: "emb".into(),
        vectors: (from..from + n)
            .map(|i| pb::Vector {
                values: vec_of(dim, i as u32 + 1),
                payload: None,
            })
            .collect(),
    };

    // 30 points -> commit v0.
    points
        .upsert(tokio_stream::iter(vec![upsert(0, 30)]))
        .await
        .unwrap();
    let v0 = versioning
        .commit(pb::CommitRequest {
            collection: "emb".into(),
            message: Some("first".into()),
            tag: None,
        })
        .await
        .unwrap()
        .into_inner()
        .version;

    // 10 more -> commit v1.
    points
        .upsert(tokio_stream::iter(vec![upsert(30, 10)]))
        .await
        .unwrap();
    let v1 = versioning
        .commit(pb::CommitRequest {
            collection: "emb".into(),
            message: None,
            tag: None,
        })
        .await
        .unwrap()
        .into_inner()
        .version;

    let versions = versioning
        .list_versions(pb::ListVersionsRequest {
            collection: "emb".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(versions.versions.len(), 2);
    assert_eq!(versions.head, Some(v1));

    let q = vec_of(dim, 7);
    let at = |v: u64| {
        Some(pb::VersionRef {
            selector: Some(pb::version_ref::Selector::Version(v)),
        })
    };
    let count_at = |at_ref| {
        let mut query = pb::query_client::QueryClient::new(channel.clone());
        let q = q.clone();
        async move {
            query
                .query(pb::QueryRequest {
                    collection: "emb".into(),
                    vector: q,
                    k: 50,
                    at: at_ref,
                    filter: None,
                })
                .await
                .unwrap()
                .into_inner()
                .results
                .len()
        }
    };

    assert_eq!(count_at(None).await, 40); // live
    assert_eq!(count_at(at(v0)).await, 30); // time-travel to v0
    assert_eq!(count_at(at(v1)).await, 40);

    // Diff v0 -> v1: 10 added, 0 removed.
    let diff = versioning
        .diff(pb::DiffRequest {
            collection: "emb".into(),
            from: v0,
            to: v1,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(diff.added.len(), 10);
    assert_eq!(diff.removed.len(), 0);

    // Tag v0 and query as of the tag.
    versioning
        .create_tag(pb::TagRequest {
            collection: "emb".into(),
            name: "base".into(),
            version: v0,
        })
        .await
        .unwrap();
    let by_tag = pb::query_client::QueryClient::new(channel.clone())
        .query(pb::QueryRequest {
            collection: "emb".into(),
            vector: q.clone(),
            k: 50,
            at: Some(pb::VersionRef {
                selector: Some(pb::version_ref::Selector::Tag("base".into())),
            }),
            filter: None,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(by_tag.results.len(), 30);

    // Restore to v0: live now reflects 30 points.
    versioning
        .restore(pb::RestoreRequest {
            collection: "emb".into(),
            version: v0,
        })
        .await
        .unwrap();
    assert_eq!(count_at(None).await, 30);

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
                    payload: None,
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
                at: None,
                filter: None,
            })
            .await
            .unwrap()
            .into_inner();
        // All n points recovered and searchable.
        assert_eq!(resp.results.len(), n);
        server.abort();
    }
}
