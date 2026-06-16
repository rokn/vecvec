//! REST/JSON gateway.
//!
//! A thin axum gateway that mirrors the gRPC surface 1:1 by calling the **same**
//! [`Service`] objects — not a proxy and no self-RPC. Runs on its own port (REST
//! 6333, gRPC 6334 by default, matching Qdrant). Errors map to HTTP status codes
//! that parallel the gRPC codes.

use std::str::FromStr;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, response::Response};
use serde::Deserialize;
use serde_json::json;
use vecvec_core::version::VersionSelector;
use vecvec_core::{Metric, PointRecord};

use crate::service::{CollectionStats, Service, ServiceError};

/// Builds the REST router over a shared [`Service`].
pub fn router(service: Arc<Service>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/collections", get(list_collections))
        .route(
            "/collections/{name}",
            post(create_collection)
                .get(collection_stats)
                .delete(drop_collection),
        )
        .route(
            "/collections/{name}/points",
            post(upsert).get(scroll_points),
        )
        .route(
            "/collections/{name}/points/{id}",
            get(get_point).delete(delete_point),
        )
        .route("/collections/{name}/query", post(query))
        .route("/collections/{name}/recommend", post(recommend))
        .route("/collections/{name}/commit", post(commit))
        .route("/collections/{name}/versions", get(list_versions))
        .route("/collections/{name}/diff", get(diff))
        .route("/collections/{name}/tags", post(create_tag))
        .route("/collections/{name}/branches", post(create_branch))
        .route("/collections/{name}/restore", post(restore))
        .layer(middleware::from_fn(cors))
        .with_state(service)
}

/// Permissive CORS for local browser tooling: adds `Access-Control-*` headers and
/// short-circuits preflight `OPTIONS` requests. Dependency-free so it doesn't pull a
/// new crate into the server build.
async fn cors(req: Request, next: Next) -> Response {
    let preflight = req.method() == Method::OPTIONS;
    let mut res = if preflight {
        Response::new(Body::empty())
    } else {
        next.run(req).await
    };
    let h = res.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    h.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        "GET,POST,DELETE,OPTIONS".parse().unwrap(),
    );
    h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, "*".parse().unwrap());
    res
}

fn stats_json(s: &CollectionStats) -> serde_json::Value {
    json!({
        "name": s.name,
        "dim": s.dim,
        "metric": s.metric.as_str(),
        "count": s.count,
        "head_version": s.head_version,
    })
}

fn point_json(p: &PointRecord) -> serde_json::Value {
    json!({
        "id": p.id.get(),
        "vector": p.vector,
        "payload": p.payload,
    })
}

fn err(e: ServiceError) -> Response {
    let code = match &e {
        ServiceError::NotFound(_) => StatusCode::NOT_FOUND,
        ServiceError::AlreadyExists(_) => StatusCode::CONFLICT,
        ServiceError::Core(vecvec_core::CoreError::DimensionMismatch { .. }) => {
            StatusCode::BAD_REQUEST
        }
        ServiceError::Core(vecvec_core::CoreError::Version { .. }) => StatusCode::NOT_FOUND,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (code, Json(json!({ "error": e.to_string() }))).into_response()
}

fn bad_request(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": msg.into() })),
    )
        .into_response()
}

#[derive(Deserialize)]
struct CreateReq {
    dim: usize,
    metric: String,
}

async fn create_collection(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<CreateReq>,
) -> Response {
    let metric = match Metric::from_str(&req.metric) {
        Ok(m) => m,
        Err(e) => return bad_request(e.to_string()),
    };
    match svc.create_collection(name.clone(), req.dim, metric) {
        Ok(()) => Json(json!({ "name": name })).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct PointReq {
    vector: Vec<f32>,
    #[serde(default)]
    payload: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct UpsertReq {
    points: Vec<PointReq>,
}

async fn upsert(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<UpsertReq>,
) -> Response {
    let points = req
        .points
        .into_iter()
        .map(|p| (p.vector, p.payload))
        .collect();
    match svc.upsert(name, points).await {
        Ok(ids) => Json(json!({ "inserted": ids.len(), "ids": ids })).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct AtReq {
    #[serde(default)]
    version: Option<u64>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    branch: Option<String>,
}

impl AtReq {
    fn selector(self) -> Option<VersionSelector> {
        if let Some(v) = self.version {
            Some(VersionSelector::Version(v))
        } else if let Some(t) = self.tag {
            Some(VersionSelector::Tag(t))
        } else {
            self.branch.map(VersionSelector::Branch)
        }
    }
}

#[derive(Deserialize)]
struct QueryReq {
    vector: Vec<f32>,
    k: usize,
    #[serde(default)]
    at: Option<AtReq>,
    #[serde(default)]
    filter: Option<vecvec_core::Filter>,
    #[serde(default)]
    include_payloads: bool,
}

fn results_json(results: Vec<(u64, f32, Option<serde_json::Value>)>) -> Response {
    let results: Vec<_> = results
        .into_iter()
        .map(|(id, score, payload)| json!({ "id": id, "score": score, "payload": payload }))
        .collect();
    Json(json!({ "results": results })).into_response()
}

async fn query(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<QueryReq>,
) -> Response {
    let at = req.at.and_then(AtReq::selector);
    match svc
        .query(name, req.vector, req.k, at, req.filter, req.include_payloads)
        .await
    {
        Ok(results) => results_json(results),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct RecommendReq {
    #[serde(default)]
    positive: Vec<u64>,
    #[serde(default)]
    negative: Vec<u64>,
    k: usize,
    #[serde(default)]
    filter: Option<vecvec_core::Filter>,
    #[serde(default)]
    include_payloads: bool,
}

async fn recommend(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<RecommendReq>,
) -> Response {
    match svc
        .recommend(
            name,
            req.positive,
            req.negative,
            req.k,
            req.filter,
            req.include_payloads,
        )
        .await
    {
        Ok(results) => results_json(results),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct CommitReq {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    tag: Option<String>,
}

async fn commit(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<CommitReq>,
) -> Response {
    match svc.commit(name, req.message, req.tag).await {
        Ok(version) => Json(json!({ "version": version })).into_response(),
        Err(e) => err(e),
    }
}

async fn list_versions(State(svc): State<Arc<Service>>, Path(name): Path<String>) -> Response {
    match svc.list_versions(&name) {
        Ok((manifests, head)) => {
            let versions: Vec<_> = manifests
                .iter()
                .map(|m| {
                    json!({
                        "version": m.version,
                        "parent": m.parent,
                        "created_at_ms": m.created_at_ms,
                        "trigger": m.trigger,
                        "message": m.message,
                    })
                })
                .collect();
            Json(json!({ "versions": versions, "head": head })).into_response()
        }
        Err(e) => err(e),
    }
}

async fn list_collections(State(svc): State<Arc<Service>>) -> Response {
    let collections: Vec<_> = svc.list_collections().iter().map(stats_json).collect();
    Json(json!({ "collections": collections })).into_response()
}

async fn collection_stats(State(svc): State<Arc<Service>>, Path(name): Path<String>) -> Response {
    match svc.collection_stats(&name) {
        Ok(s) => Json(stats_json(&s)).into_response(),
        Err(e) => err(e),
    }
}

async fn drop_collection(State(svc): State<Arc<Service>>, Path(name): Path<String>) -> Response {
    match svc.drop_collection(&name) {
        Ok(()) => Json(json!({ "dropped": name })).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct ScrollParams {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    version: Option<u64>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    branch: Option<String>,
}

fn default_limit() -> usize {
    2000
}

impl ScrollParams {
    fn selector(&self) -> Option<VersionSelector> {
        if let Some(v) = self.version {
            Some(VersionSelector::Version(v))
        } else if let Some(t) = &self.tag {
            Some(VersionSelector::Tag(t.clone()))
        } else {
            self.branch.clone().map(VersionSelector::Branch)
        }
    }
}

async fn scroll_points(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Query(params): Query<ScrollParams>,
) -> Response {
    let at = params.selector();
    match svc.scroll(name, at, params.offset, params.limit).await {
        Ok((points, total)) => {
            let points: Vec<_> = points.iter().map(point_json).collect();
            Json(json!({ "points": points, "total": total })).into_response()
        }
        Err(e) => err(e),
    }
}

async fn get_point(
    State(svc): State<Arc<Service>>,
    Path((name, id)): Path<(String, u64)>,
) -> Response {
    match svc.get_point(&name, id) {
        Ok(Some(p)) => Json(point_json(&p)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("point {id} not found") })),
        )
            .into_response(),
        Err(e) => err(e),
    }
}

async fn delete_point(
    State(svc): State<Arc<Service>>,
    Path((name, id)): Path<(String, u64)>,
) -> Response {
    match svc.delete(name, id).await {
        Ok(deleted) => Json(json!({ "deleted": deleted, "id": id })).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct DiffParams {
    from: u64,
    to: u64,
}

async fn diff(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Query(params): Query<DiffParams>,
) -> Response {
    match svc.diff(&name, params.from, params.to) {
        Ok((added, removed)) => Json(json!({ "added": added, "removed": removed })).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct RefReq {
    name: String,
    version: u64,
}

async fn create_tag(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<RefReq>,
) -> Response {
    match svc.create_tag(name, req.name.clone(), req.version).await {
        Ok(()) => Json(json!({ "tag": req.name, "version": req.version })).into_response(),
        Err(e) => err(e),
    }
}

async fn create_branch(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<RefReq>,
) -> Response {
    match svc.create_branch(name, req.name.clone(), req.version).await {
        Ok(()) => Json(json!({ "branch": req.name, "version": req.version })).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct RestoreReq {
    version: u64,
}

async fn restore(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<RestoreReq>,
) -> Response {
    match svc.restore(name, req.version).await {
        Ok(version) => Json(json!({ "version": version })).into_response(),
        Err(e) => err(e),
    }
}
