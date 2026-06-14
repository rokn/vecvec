//! M12 REST gateway test: create → upsert → query → commit → versions over HTTP/JSON,
//! exercised in-process via `tower::oneshot` (no network).

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use vecvec_core::FsyncMode;
use vecvec_server::{Service, rest};

fn vec_of(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

async fn post(router: Router, uri: &str, body: String) -> (StatusCode, serde_json::Value) {
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

async fn get(router: Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let resp = router
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_lifecycle() {
    let data = tempfile::tempdir().unwrap();
    let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
    let router = rest::router(service);
    let dim = 8;

    // create
    let (s, _) = post(
        router.clone(),
        "/collections/c",
        r#"{"dim":8,"metric":"dot"}"#.into(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // upsert 20 points (every 4th tagged group=1)
    let points: Vec<String> = (0..20u32)
        .map(|i| {
            format!(
                r#"{{"vector":{:?},"payload":{{"group":{}}}}}"#,
                vec_of(dim, i + 1),
                i % 4
            )
        })
        .collect();
    let (s, j) = post(
        router.clone(),
        "/collections/c/points",
        format!(r#"{{"points":[{}]}}"#, points.join(",")),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["inserted"], 20);

    // query top-5
    let (s, j) = post(
        router.clone(),
        "/collections/c/query",
        format!(r#"{{"vector":{:?},"k":5}}"#, vec_of(dim, 99)),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["results"].as_array().unwrap().len(), 5);

    // filtered query (group == 1)
    let (s, j) = post(
        router.clone(),
        "/collections/c/query",
        format!(
            r#"{{"vector":{:?},"k":3,"filter":{{"must":[{{"key":"group","match":1}}]}}}}"#,
            vec_of(dim, 99)
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(j["results"].as_array().unwrap().iter().all(|r| {
        // group==1 ids are 1,5,9,13,17 -> id % 4 == 1
        r["id"].as_u64().unwrap() % 4 == 1
    }));

    // commit + list versions
    let (s, j) = post(router.clone(), "/collections/c/commit", "{}".into()).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["version"], 0);
    let (s, j) = get(router.clone(), "/collections/c/versions").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["versions"].as_array().unwrap().len(), 1);

    // missing collection -> 404
    let (s, _) = post(
        router.clone(),
        "/collections/missing/query",
        format!(r#"{{"vector":{:?},"k":1}}"#, vec_of(dim, 1)),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);

    // healthz
    let resp = router
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
