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

async fn del(router: Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let resp = router
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(uri)
                .body(Body::empty())
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_routes_coverage() {
    // Exercises the REST routes the lifecycle test doesn't: stats, list, scroll,
    // get/delete point, recommend, diff, tag/branch, restore, and drop (incl. the
    // on-disk deletion).
    let data = tempfile::tempdir().unwrap();
    let service = Arc::new(Service::open(data.path(), 2, 8, FsyncMode::Sync).unwrap());
    let router = rest::router(service);
    let dim = 8;

    let (s, _) = post(router.clone(), "/collections/c", r#"{"dim":8,"metric":"dot"}"#.into()).await;
    assert_eq!(s, StatusCode::OK);
    let points: Vec<String> = (0..12u32)
        .map(|i| format!(r#"{{"vector":{:?},"payload":{{"g":{}}}}}"#, vec_of(dim, i + 1), i % 3))
        .collect();
    let (s, j) = post(
        router.clone(),
        "/collections/c/points",
        format!(r#"{{"points":[{}]}}"#, points.join(",")),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["inserted"], 12);
    // commit v0
    let (s, j) = post(router.clone(), "/collections/c/commit", "{}".into()).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["version"], 0);

    // stats
    let (s, j) = get(router.clone(), "/collections/c").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["count"], 12);
    assert_eq!(j["dim"], 8);

    // list collections
    let (s, j) = get(router.clone(), "/collections").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["collections"].as_array().unwrap().len(), 1);

    // scroll (paged)
    let (s, j) = get(router.clone(), "/collections/c/points?offset=0&limit=5").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["points"].as_array().unwrap().len(), 5);
    assert_eq!(j["total"], 12);

    // get point 0; missing -> 404
    let (s, j) = get(router.clone(), "/collections/c/points/0").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["id"], 0);
    let (s, _) = get(router.clone(), "/collections/c/points/999").await;
    assert_eq!(s, StatusCode::NOT_FOUND);

    // recommend from a positive example
    let (s, j) = post(
        router.clone(),
        "/collections/c/recommend",
        r#"{"positive":[0],"negative":[],"k":3}"#.into(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["results"].as_array().unwrap().len(), 3);

    // delete a point, then commit v1
    let (s, j) = del(router.clone(), "/collections/c/points/0").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["deleted"], true);
    let (s, _) = post(router.clone(), "/collections/c/commit", "{}".into()).await;
    assert_eq!(s, StatusCode::OK);

    // diff v0 -> v1 shows id 0 removed
    let (s, j) = get(router.clone(), "/collections/c/diff?from=0&to=1").await;
    assert_eq!(s, StatusCode::OK);
    assert!(j["removed"].as_array().unwrap().iter().any(|v| v.as_u64() == Some(0)));

    // tag + branch over v0
    let (s, _) = post(router.clone(), "/collections/c/tags", r#"{"name":"rel","version":0}"#.into()).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _) = post(router.clone(), "/collections/c/branches", r#"{"name":"b","version":0}"#.into()).await;
    assert_eq!(s, StatusCode::OK);

    // restore v0 (brings point 0 back); returns a new version
    let (s, j) = post(router.clone(), "/collections/c/restore", r#"{"version":0}"#.into()).await;
    assert_eq!(s, StatusCode::OK);
    assert!(j["version"].as_u64().unwrap() >= 2);
    let (s, _) = get(router.clone(), "/collections/c/points/0").await;
    assert_eq!(s, StatusCode::OK, "restore should bring point 0 back");

    // drop the collection: response + the on-disk directory must be gone
    let coll_dir = data.path().join("collections").join("c");
    assert!(coll_dir.exists());
    let (s, j) = del(router.clone(), "/collections/c").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(j["dropped"], "c");
    assert!(!coll_dir.exists(), "drop_collection must delete on-disk data");
    let (s, _) = get(router.clone(), "/collections/c").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
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
