//! gRPC adapters: thin translation between the `vecvec.v1` wire types and the
//! [`Service`] facade. No business logic lives here — just pb ⇄ core conversion and
//! error mapping.

use std::sync::Arc;

use tonic::{Request, Response, Status, Streaming};
use vecvec_core::Metric;
use vecvec_core::version::VersionSelector;
use vecvec_proto::pb;

use crate::service::{Service, ServiceError};

/// The object that implements every gRPC service, sharing one [`Service`].
#[derive(Clone)]
pub struct Api {
    svc: Arc<Service>,
}

impl Api {
    /// Wraps a shared service.
    pub fn new(svc: Arc<Service>) -> Self {
        Self { svc }
    }
}

fn map_metric(m: i32) -> Result<Metric, Status> {
    match pb::Metric::try_from(m) {
        Ok(pb::Metric::Cosine) => Ok(Metric::Cosine),
        Ok(pb::Metric::Dot) => Ok(Metric::Dot),
        Ok(pb::Metric::Euclidean) => Ok(Metric::Euclidean),
        Ok(pb::Metric::Unspecified) => Err(Status::invalid_argument("metric must be specified")),
        Err(_) => Err(Status::invalid_argument("unknown metric")),
    }
}

fn to_status(e: ServiceError) -> Status {
    match &e {
        ServiceError::NotFound(_) => Status::not_found(e.to_string()),
        ServiceError::AlreadyExists(_) => Status::already_exists(e.to_string()),
        ServiceError::Core(vecvec_core::CoreError::DimensionMismatch { .. }) => {
            Status::invalid_argument(e.to_string())
        }
        ServiceError::Core(vecvec_core::CoreError::Version { .. }) => {
            Status::not_found(e.to_string())
        }
        ServiceError::Core(_) | ServiceError::Bridge(_) => Status::internal(e.to_string()),
    }
}

fn version_selector(at: Option<pb::VersionRef>) -> Option<VersionSelector> {
    at.and_then(|vr| vr.selector).map(|sel| match sel {
        pb::version_ref::Selector::Version(v) => VersionSelector::Version(v),
        pb::version_ref::Selector::Tag(t) => VersionSelector::Tag(t),
        pb::version_ref::Selector::Branch(b) => VersionSelector::Branch(b),
    })
}

#[tonic::async_trait]
impl pb::collections_server::Collections for Api {
    async fn create(
        &self,
        request: Request<pb::CreateCollectionRequest>,
    ) -> Result<Response<pb::CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let metric = map_metric(req.metric)?;
        if req.dim == 0 {
            return Err(Status::invalid_argument("dim must be > 0"));
        }
        self.svc
            .create_collection(req.name.clone(), req.dim as usize, metric)
            .map_err(to_status)?;
        Ok(Response::new(pb::CreateCollectionResponse {
            name: req.name,
        }))
    }
}

#[tonic::async_trait]
impl pb::points_server::Points for Api {
    async fn upsert(
        &self,
        request: Request<Streaming<pb::UpsertRequest>>,
    ) -> Result<Response<pb::UpsertResponse>, Status> {
        let mut stream = request.into_inner();
        let mut collection: Option<String> = None;
        let mut points: Vec<(Vec<f32>, Option<vecvec_core::Payload>)> = Vec::new();
        while let Some(batch) = stream.message().await? {
            if collection.is_none() && !batch.collection.is_empty() {
                collection = Some(batch.collection);
            }
            for v in batch.vectors {
                let payload = match v.payload {
                    Some(json) => Some(
                        serde_json::from_str(&json)
                            .map_err(|e| Status::invalid_argument(format!("bad payload: {e}")))?,
                    ),
                    None => None,
                };
                points.push((v.values, payload));
            }
        }
        let collection =
            collection.ok_or_else(|| Status::invalid_argument("no collection specified"))?;
        let ids = self
            .svc
            .upsert(collection, points)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::UpsertResponse {
            inserted: ids.len() as u64,
            ids,
        }))
    }

    async fn write_batch(
        &self,
        request: Request<pb::WriteBatchRequest>,
    ) -> Result<Response<pb::WriteBatchResponse>, Status> {
        let req = request.into_inner();
        if req.collection.is_empty() {
            return Err(Status::invalid_argument("no collection specified"));
        }
        let mut upserts: Vec<(Vec<f32>, Option<vecvec_core::Payload>)> =
            Vec::with_capacity(req.upserts.len());
        for v in req.upserts {
            let payload = match v.payload {
                Some(json) => Some(
                    serde_json::from_str(&json)
                        .map_err(|e| Status::invalid_argument(format!("bad payload: {e}")))?,
                ),
                None => None,
            };
            upserts.push((v.values, payload));
        }
        let commit = req.commit.then_some((req.message, req.tag));
        let out = self
            .svc
            .write_batch(req.collection, upserts, req.deletes, commit)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::WriteBatchResponse {
            ids: out.ids,
            deleted: out.deleted,
            version: out.version,
        }))
    }
}

#[tonic::async_trait]
impl pb::query_server::Query for Api {
    async fn query(
        &self,
        request: Request<pb::QueryRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        let req = request.into_inner();
        let at = version_selector(req.at);
        let filter = match req.filter {
            Some(json) => Some(
                serde_json::from_str::<vecvec_core::Filter>(&json)
                    .map_err(|e| Status::invalid_argument(format!("bad filter: {e}")))?,
            ),
            None => None,
        };
        let results = self
            .svc
            .query(req.collection, req.vector, req.k as usize, at, filter)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::QueryResponse {
            results: results
                .into_iter()
                .map(|(id, score)| pb::ScoredPoint { id, score })
                .collect(),
        }))
    }

    async fn recommend(
        &self,
        request: Request<pb::RecommendRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        let req = request.into_inner();
        let filter = match req.filter {
            Some(json) => Some(
                serde_json::from_str::<vecvec_core::Filter>(&json)
                    .map_err(|e| Status::invalid_argument(format!("bad filter: {e}")))?,
            ),
            None => None,
        };
        let results = self
            .svc
            .recommend(
                req.collection,
                req.positive,
                req.negative,
                req.k as usize,
                filter,
            )
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::QueryResponse {
            results: results
                .into_iter()
                .map(|(id, score)| pb::ScoredPoint { id, score })
                .collect(),
        }))
    }
}

#[tonic::async_trait]
impl pb::versioning_server::Versioning for Api {
    async fn list_versions(
        &self,
        request: Request<pb::ListVersionsRequest>,
    ) -> Result<Response<pb::ListVersionsResponse>, Status> {
        let req = request.into_inner();
        let (manifests, head) = self.svc.list_versions(&req.collection).map_err(to_status)?;
        let versions = manifests
            .iter()
            .map(|m| pb::VersionInfo {
                version: m.version,
                parent: m.parent,
                created_at_ms: m.created_at_ms,
                trigger: m.trigger.clone(),
                message: m.message.clone(),
            })
            .collect();
        Ok(Response::new(pb::ListVersionsResponse { versions, head }))
    }

    async fn commit(
        &self,
        request: Request<pb::CommitRequest>,
    ) -> Result<Response<pb::CommitResponse>, Status> {
        let req = request.into_inner();
        let version = self
            .svc
            .commit(req.collection, req.message, req.tag)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::CommitResponse { version }))
    }

    async fn create_tag(
        &self,
        request: Request<pb::TagRequest>,
    ) -> Result<Response<pb::RefResponse>, Status> {
        let req = request.into_inner();
        self.svc
            .create_tag(req.collection, req.name, req.version)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::RefResponse {}))
    }

    async fn create_branch(
        &self,
        request: Request<pb::BranchRequest>,
    ) -> Result<Response<pb::RefResponse>, Status> {
        let req = request.into_inner();
        self.svc
            .create_branch(req.collection, req.name, req.version)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::RefResponse {}))
    }

    async fn diff(
        &self,
        request: Request<pb::DiffRequest>,
    ) -> Result<Response<pb::DiffResponse>, Status> {
        let req = request.into_inner();
        let (added, removed) = self
            .svc
            .diff(&req.collection, req.from, req.to)
            .map_err(to_status)?;
        Ok(Response::new(pb::DiffResponse { added, removed }))
    }

    async fn restore(
        &self,
        request: Request<pb::RestoreRequest>,
    ) -> Result<Response<pb::RestoreResponse>, Status> {
        let req = request.into_inner();
        let version = self
            .svc
            .restore(req.collection, req.version)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::RestoreResponse { version }))
    }
}
