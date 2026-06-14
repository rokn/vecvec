//! REST/JSON gateway.
//!
//! A thin axum gateway that mirrors the gRPC surface 1:1 by calling the **same**
//! [`Service`] objects — not a proxy and no self-RPC. Runs on its own port (REST
//! 6333, gRPC 6334 by default, matching Qdrant). Errors map to HTTP status codes
//! that parallel the gRPC codes.

use std::str::FromStr;
use std::sync::Arc;

use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, response::Response};
use serde::Deserialize;
use serde_json::json;
use vecvec_core::Metric;
use vecvec_core::version::VersionSelector;

use crate::service::{Service, ServiceError};

/// Builds the REST router over a shared [`Service`].
pub fn router(service: Arc<Service>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/collections/{name}", post(create_collection))
        .route("/collections/{name}/points", post(upsert))
        .route("/collections/{name}/query", post(query))
        .route("/collections/{name}/recommend", post(recommend))
        .route("/collections/{name}/commit", post(commit))
        .route("/collections/{name}/versions", get(list_versions))
        .with_state(service)
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
}

fn results_json(results: Vec<(u64, f32)>) -> Response {
    let results: Vec<_> = results
        .into_iter()
        .map(|(id, score)| json!({ "id": id, "score": score }))
        .collect();
    Json(json!({ "results": results })).into_response()
}

async fn query(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<QueryReq>,
) -> Response {
    let at = req.at.and_then(AtReq::selector);
    match svc.query(name, req.vector, req.k, at, req.filter).await {
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
}

async fn recommend(
    State(svc): State<Arc<Service>>,
    Path(name): Path<String>,
    Json(req): Json<RecommendReq>,
) -> Response {
    match svc
        .recommend(name, req.positive, req.negative, req.k, req.filter)
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
