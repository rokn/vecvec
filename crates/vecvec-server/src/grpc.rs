//! gRPC adapters: thin translation between the `vecvec.v1` wire types and the
//! [`Service`] facade. No business logic lives here — just pb ⇄ core conversion and
//! error mapping.

use std::sync::Arc;

use tonic::{Request, Response, Status, Streaming};
use vecvec_core::Metric;
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
        ServiceError::Core(_) | ServiceError::Bridge(_) => Status::internal(e.to_string()),
    }
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
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        while let Some(batch) = stream.message().await? {
            if collection.is_none() && !batch.collection.is_empty() {
                collection = Some(batch.collection);
            }
            for v in batch.vectors {
                vectors.push(v.values);
            }
        }
        let collection =
            collection.ok_or_else(|| Status::invalid_argument("no collection specified"))?;
        let ids = self
            .svc
            .upsert(collection, vectors)
            .await
            .map_err(to_status)?;
        Ok(Response::new(pb::UpsertResponse {
            inserted: ids.len() as u64,
            ids,
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
        let results = self
            .svc
            .query(req.collection, req.vector, req.k as usize)
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
